# This file is part of tad-mctc.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2024 Grimme Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Neighbour search: triple (three-body) enumeration
===================================================

Three-body terms (e.g. the Axilrod-Teller-Muto dispersion contribution in
``tad-dftd3``/``tad-dftd4``) need every candidate triple ``(i, j, k)`` --
``j`` the centre atom, ``i`` and ``k`` its neighbours, each at an image
shift for a periodic system -- whose three
pairwise distances all fall inside a cutoff, not just pairs. There is no
three-body physics in this module at all -- it is pure index math over a
:class:`.NeighborList`'s real pairs, factored out here (rather than
hand-copied into every three-body consumer) precisely so that copy never
has to happen, or be fixed, twice.

:func:`triples_from_neighborlist` is the public entry point: given a
:class:`.NeighborList` (built at ``cutoff``, or wider via ``skin``, per its
own reuse contract) and the :class:`~tad_mctc.io.structure.Structure` it
describes -- the list itself stores no positions, since it is meant to be
built once and reused across many evaluations -- it yields :class:`TripleChunk` chunks of
real atom indices and image shifts, each with at most ``chunk_size`` rows,
never materialising the full candidate-triple list at once. That chunking is not
an implementation detail a caller can ignore: the candidate-triple count
grows as ``sum(degree * (degree - 1) / 2)`` over all centre atoms, with
``degree`` the *upward* degree of :func:`_oriented_csr`, i.e. as
``cutoff ** 6`` in three dimensions, so an unchunked list can exceed memory
long before the underlying neighbour list does.

Every triple is yielded exactly once (see :func:`_oriented_csr`), with
the smallest of its three points as centre, so a consumer that shares the
energy of a triple equally between its three atoms adds ``E / 3`` to
each of ``idx_i``, ``idx_j`` and ``idx_k``. The lattice enters only to
decide the cutoff; the consumer takes its own, live lattice for the
distances.

Internally, :func:`triples_from_neighborlist` dispatches on
``positions.device.type``: a per-centre-atom Python loop
(:func:`_iter_triple_chunks_loop`) on CPU, where each iteration is cheap and
there is no device-sync cost to amortise, and a vectorized enumeration
over all centres at once (:func:`_iter_triple_chunks_vectorized`)
everywhere else (i.e. CUDA), where the loop's per-atom device syncs and
kernel launches would otherwise dominate. Both are proven, by test, to enumerate
the exact same triple set for the same input -- see
``test/test_neighbor/test_triples.py``.

:class:`TripleList` is the other form, for reuse over many evaluations
(e.g. the steps of a molecular dynamics): every triangle of a list's pairs,
enumerated once, each with its three sides as slots of the list. A
consumer computes what depends on a pair (a distance, a coefficient) once
per slot and only looks it up per triple, and masks the triples at its own
cutoff.

List *construction* (this whole module) is data-dependent --
``bincount``/``argsort`` for the CSR adjacency, a Python loop over centre
atoms or ``searchsorted``/``cumsum`` for the flat enumeration, boolean
masking against the cutoff, chunk-buffer flushing based on a running Python
int -- so every function here runs under ``torch.no_grad()``, outside any
traced or transformed region. A consumer is expected to treat the yielded index and shift tensors as
opaque, fixed-index-set input to its own differentiable physics -- exactly
the same split :mod:`.list` draws between building and consuming a
:class:`.NeighborList`.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import TYPE_CHECKING, NamedTuple

import torch

from ..tree import Node, child
from ..typing import Tensor
from ._distance_kernels import _image_translation, split_lattice
from ._tiles import _is_forward, _ragged_runs
from .list import NeighborList, _molecular_shift

if TYPE_CHECKING:
    from ..io.structure import Structure

__all__ = [
    "TripleChunk",
    "TripleIndex",
    "TripleList",
    "triples_from_neighborlist",
]


class TripleChunk(NamedTuple):
    """
    One chunk of three-body triples, ``n`` rows. A triple is a centre atom
    ``idx_j`` -- the smallest point of the triple, see
    :func:`_oriented_csr` -- and two of its neighbours ``idx_i`` and
    ``idx_k``.

    The shifts are integer lattice translations of the neighbours relative
    to the centre, with the sign of :func:`.pair_distance_squared`: the
    vector from the centre to the neighbour is
    ``x_i - x_j + shift_i @ lattice``, and likewise for ``k``. For a
    molecular list both are all-zero views of one row.
    """

    idx_i: Tensor
    """``(n,)``, first neighbour of the centre."""
    idx_j: Tensor
    """``(n,)``, centre atom."""
    idx_k: Tensor
    """``(n,)``, second neighbour of the centre."""
    shift_i: Tensor
    """``(n, 3)``, image of ``idx_i`` relative to the centre."""
    shift_k: Tensor
    """``(n, 3)``, image of ``idx_k`` relative to the centre."""


def _within_cutoff(
    centre: Tensor, left: Tensor, right: Tensor, cutoff_squared: float
) -> Tensor:
    """
    Whether all three sides of each candidate triple -- ``centre-left``,
    ``centre-right`` and the third side ``left-right`` -- are inside the
    cutoff. Shared by both triple-chunk generators.

    The caller gathers the positions with ``index_select``, not
    ``positions[idx]``, as in :func:`.pair_distance_squared`.

    Parameters
    ----------
    centre, left, right : Tensor
        Positions of the three atoms, ``(n_triples, 3)``. ``centre`` may
        be a single ``(3,)`` row shared by all triples.
    cutoff_squared : float
        The squared cutoff.

    Returns
    -------
    Tensor
        Boolean mask of shape ``(n_triples,)``.
    """
    distance_ij_squared = (centre - left).pow(2).sum(-1)
    distance_jk_squared = (centre - right).pow(2).sum(-1)
    distance_ik_squared = (left - right).pow(2).sum(-1)
    return (
        (distance_ij_squared <= cutoff_squared)
        & (distance_jk_squared <= cutoff_squared)
        & (distance_ik_squared <= cutoff_squared)
    )


def _translated(
    positions: Tensor, shift: Tensor, cell: Tensor | None
) -> Tensor:
    """
    ``positions`` moved by the integer lattice translation ``shift``.

    Parameters
    ----------
    positions : Tensor
        ``(n, 3)``.
    shift : Tensor
        ``(n, 3)``, integer.
    cell : Tensor | None
        One ``(3, 3)`` cell for all rows, one ``(n, 3, 3)`` cell per row, or
        ``None`` for a molecule, which is returned unchanged.
    """
    if cell is None:
        return positions
    return positions + _image_translation(shift, cell)


def _oriented_csr(
    nat: int, idx_i: Tensor, idx_j: Tensor, shift: Tensor | None
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Oriented CSR adjacency from real (unpadded) pairs. Shared by both the
    loop and vectorized triple-chunk generators below.

    A point is an atom together with an image shift, ``(atom, shift)``,
    and points are compared lexicographically; the centre of a block is
    ``(atom, 0)``. A stored entry ``(i, j, S)`` is a half list's pair, and
    stands for the neighbour ``(j, +S)`` of ``i`` as well as for the
    neighbour ``(i, -S)`` of ``j`` (for ``i == j``, the images ``+S`` and
    ``-S`` of the same atom). Exactly one of the two is *upward*, i.e. a
    larger point than its centre, and only that one enters the CSR.

    Every pair of a centre's upward neighbours is then one candidate
    triple, and every triple -- an unordered set of three points up to a
    common lattice translation -- has exactly one smallest point, so it is
    a candidate of exactly one centre, once. For a molecular list,
    ``shift=None``, the shifts are zero and the order reduces to the atom
    index.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor, Tensor]
        ``(offset, neighbours, neighbour_shift, degree)`` -- atom ``a``'s
        upward neighbours are ``neighbours[offset[a] : offset[a] +
        degree[a]]``, at the image ``neighbour_shift[...]`` relative to
        ``a``.
    """
    return _oriented_csr_entries(nat, idx_i, idx_j, shift)[:4]


def _oriented_csr_entries(
    nat: int, idx_i: Tensor, idx_j: Tensor, shift: Tensor | None
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    :func:`_oriented_csr`, and in addition the entry of the pairs each
    neighbour comes from, ``(n_pairs,)``: neighbour ``n`` of the CSR is
    the pair ``entry[n]`` of `idx_i`, `idx_j` and `shift`.
    """
    if shift is None:
        upward = idx_j > idx_i
    else:
        # An `i == j` entry always has a non-zero shift, for which
        # `_is_forward` is the lexicographic `shift > 0`.
        shift = shift.to(torch.long)
        upward = (idx_j > idx_i) | ((idx_j == idx_i) & _is_forward(shift))

    src = torch.where(upward, idx_i, idx_j)
    dst = torch.where(upward, idx_j, idx_i)

    src_sorted, order = torch.sort(src, stable=True)
    neighbours = dst[order]
    if shift is None:
        neighbour_shift = _molecular_shift(neighbours.shape[0], src.device)
    else:
        dst_shift = torch.where(upward.unsqueeze(-1), shift, -shift)
        neighbour_shift = dst_shift[order]

    degree = torch.bincount(src_sorted, minlength=nat)
    offset = torch.cat([degree.new_zeros(1), torch.cumsum(degree, dim=0)])[:nat]
    return offset, neighbours, neighbour_shift, degree, order


def _triangle_rows(degree: Tensor) -> tuple[Tensor, Tensor]:
    """
    Number the candidate triples of every centre atom by rows.

    A centre with ``d`` neighbours has one candidate triple per pair of
    its local neighbour slots ``(a, b)``, ``a < b``: the strict upper
    triangle of a ``d x d`` block, whose row ``a`` holds ``d - 1 - a``
    entries. The rows of all centres, one centre after the other, number
    the candidate triples with one flat index, in the order that
    ``torch.triu_indices`` lists each centre's slot pairs.

    It lives in tad-mctc rather than in tad-dftd3's ATM term because other
    three-body consumers (e.g. tad-dftd4) need the identical numbering.

    Parameters
    ----------
    degree : Tensor
        ``(n_centres,)``, the number of neighbours of each centre.

    Returns
    -------
    tuple[Tensor, Tensor]
        ``first_row``, ``(n_centres,)``, the number of each centre's row
        ``0``; and ``row_offset``, ``(n_rows + 1,)``, the flat index of
        each row's first entry, ending with the number of entries.
    """
    rows_per_centre = (degree - 1).clamp_min(0)
    first_row = torch.cumsum(rows_per_centre, dim=0) - rows_per_centre

    row_centre, row_in_centre = _ragged_runs(rows_per_centre)
    row_length = rows_per_centre[row_centre] - row_in_centre

    row_offset = torch.cat(
        [row_length.new_zeros(1), torch.cumsum(row_length, dim=0)]
    )
    return first_row, row_offset


def _triangular_indices(
    flat: Tensor, row_offset: Tensor, first_row: Tensor | int
) -> tuple[Tensor, Tensor]:
    """
    Invert the numbering of :func:`_triangle_rows`: flat index ->
    ``(a, b)`` with ``a < b``, both local neighbour slots within the
    block of the index's centre atom. A search in integers only, so it is
    exact for every degree and on every device.

    Parameters
    ----------
    flat : Tensor
        ``(n,)``, flat candidate-triple indices.
    row_offset : Tensor
        ``(n_rows + 1,)``, from :func:`_triangle_rows`.
    first_row : Tensor | int
        ``(n,)``, the number of row ``0`` of each index's centre, or ``0``
        when ``row_offset`` numbers the rows of a single centre.

    Returns
    -------
    tuple[Tensor, Tensor]
        ``(a, b)``, each ``(n,)``.
    """
    # `right=True`: the row whose entries hold `flat`, i.e. the last row
    # with `row_offset[row] <= flat`. No row is empty, so it is unique.
    row = torch.searchsorted(row_offset, flat, right=True) - 1
    a = row - first_row
    b = a + 1 + (flat - row_offset[row])
    return a, b


@torch.no_grad()
def _iter_triple_chunks_loop(
    nat: int,
    real_i: Tensor,
    real_j: Tensor,
    real_shift: Tensor,
    positions: Tensor,
    lattice: Tensor | None,
    atoms_per_system: int,
    cutoff: float,
    chunk_size: int | None = None,
) -> Iterator[TripleChunk]:
    """
    CPU-favored generator: a per-centre-atom Python loop over a CSR
    adjacency built from the symmetrised pairs. For every centre atom
    with at least two neighbours, every unordered pair of neighbours is a
    candidate triple. The two edges ``i-j`` and ``j-k`` are only
    guaranteed inside ``cutoff`` when the neighbour list itself was built
    at exactly this ``cutoff`` with zero skin; a skinned or wider-cutoff
    list (its whole point is reuse across steps without rebuilding) hands
    ``j`` neighbours out to a larger radius, so all three sides -- ``i-j``,
    ``j-k``, and the third side ``i-k``, which is never implied by
    construction at all -- are checked explicitly here.

    Bounding the working set means never holding more than ``chunk_size``
    candidate rows' worth of intermediate tensors alive at once, across
    *both* directions this can blow up:

    - across centres: candidates from already-processed centres are
      buffered and flushed (yielded) exactly when the buffer would reach
      ``chunk_size``, rather than accumulated for every centre atom
      before any is handed back to the caller;
    - within a single centre: a centre's own candidate count is
      ``C(degree, 2)``, and it is only ever materialised via the cheap
      ``torch.triu_indices`` when the *buffer's current remaining room*
      covers that whole block in one go (the realistic case at
      chemically-plausible degrees, once the buffer is empty or nearly
      so); whenever a centre's block does not fit in what remains of the
      buffer -- whether because the block alone exceeds ``chunk_size``,
      or merely because the buffer already holds candidates from earlier
      centres -- the next sub-range is sized to exactly fill the buffer
      to ``chunk_size`` and built via :func:`_triangular_indices` (shared
      with the vectorized path) instead, so no allocation, buffered or
      otherwise, ever exceeds ``chunk_size`` rows -- not even by a
      partially-filled buffer plus one more full centre block.

    ``chunk_size=None`` disables chunking (a single flush at the end).

    Data-dependent (``bincount``, a Python loop over centre atoms,
    boolean masking, buffer flushing based on a running Python int), so
    this always runs under ``torch.no_grad()``, outside any traced or
    transformed region.

    Yields
    ------
    TripleChunk
        Chunks of real atom indices only, each
        with at most ``chunk_size`` rows (fewer once the cutoff mask is
        applied) -- concatenating every yielded chunk reproduces exactly
        what an unchunked call (``chunk_size=None``) returns in one chunk.
    """
    device = real_i.device
    offset, neighbours, neighbour_shift, degree = _oriented_csr(
        nat, real_i, real_j, None if lattice is None else real_shift
    )
    shared_lattice, system_lattices = split_lattice(lattice)
    cutoff_squared = cutoff * cutoff

    left_parts: list[Tensor] = []
    centre_parts: list[Tensor] = []
    right_parts: list[Tensor] = []
    left_shift_parts: list[Tensor] = []
    right_shift_parts: list[Tensor] = []
    buffered = 0

    def flush() -> TripleChunk:
        nonlocal buffered
        left = torch.cat(left_parts)
        centre = torch.cat(centre_parts)
        right = torch.cat(right_parts)
        if lattice is None:
            zero = _molecular_shift(left.shape[0], device)
            left_shift = right_shift = zero
        else:
            left_shift = torch.cat(left_shift_parts)
            right_shift = torch.cat(right_shift_parts)
        left_parts.clear()
        centre_parts.clear()
        right_parts.clear()
        left_shift_parts.clear()
        right_shift_parts.clear()
        buffered = 0
        return TripleChunk(left, centre, right, left_shift, right_shift)

    # Read once, rather than one `.item()` per atom.
    degree_of = degree.tolist()
    offset_of = offset.tolist()

    for centre_atom in range(nat):
        centre_degree = degree_of[centre_atom]
        if centre_degree < 2:
            continue

        start = offset_of[centre_atom]
        block = neighbours[start : start + centre_degree]
        block_shift = neighbour_shift[start : start + centre_degree]
        cell = (
            system_lattices[centre_atom // atoms_per_system]
            if system_lattices is not None
            else shared_lattice
        )
        n_local = centre_degree * (centre_degree - 1) // 2

        sub_start = 0
        while sub_start < n_local:
            # Each sub-chunk is sized to exactly fill the buffer up to
            # chunk_size (not just capped at chunk_size per centre) --
            # otherwise a partially-filled buffer plus one more
            # chunk_size-sized centre block could momentarily hold up to
            # ~2 * chunk_size rows before the next flush. This is what
            # makes both bounds -- across centres, and within one centre
            # whose own block alone would exceed chunk_size -- exact
            # rather than a loose factor-of-two bound.
            if chunk_size is None:
                step = n_local - sub_start
            else:
                remaining = chunk_size - buffered
                step = min(max(remaining, 1), n_local - sub_start)
            sub_stop = sub_start + step

            if sub_start == 0 and sub_stop == n_local:
                # Common case: this centre's whole block fits in the
                # buffer's remaining room in one go. `triu_indices` is a
                # cheap, purpose-built index generator -- use it directly
                # rather than building this centre's row table for
                # `_triangular_indices` on the common path.
                local_a, local_b = torch.triu_indices(
                    centre_degree, centre_degree, offset=1, device=device
                )
            else:
                # Either this centre's own block exceeds chunk_size, or
                # it merely doesn't fit in the buffer's current remaining
                # room: invert this sub-range of its triangular numbering
                # directly, reusing the same helper the vectorized path
                # uses, instead of allocating the full
                # `2 x C(d, 2)` index tensor and slicing it (which would
                # not bound anything).
                sub_local = torch.arange(sub_start, sub_stop, device=device)
                _, row_offset = _triangle_rows(
                    degree[centre_atom : centre_atom + 1]
                )
                local_a, local_b = _triangular_indices(sub_local, row_offset, 0)

            left = block[local_a]
            right = block[local_b]
            left_shift = block_shift[local_a]
            right_shift = block_shift[local_b]

            # `positions[centre_atom]` is a single row (a view, not a
            # gather) that broadcasts against the neighbours, which are
            # moved to their image next to the centre.
            keep = _within_cutoff(
                positions[centre_atom],
                _translated(positions.index_select(0, left), left_shift, cell),
                _translated(
                    positions.index_select(0, right), right_shift, cell
                ),
                cutoff_squared,
            )
            left, right = left[keep], right[keep]

            if left.shape[0] > 0:
                left_parts.append(left)
                centre_parts.append(torch.full_like(left, centre_atom))
                right_parts.append(right)
                left_shift_parts.append(left_shift[keep])
                right_shift_parts.append(right_shift[keep])
                buffered += left.shape[0]

            if chunk_size is not None and buffered >= chunk_size:
                yield flush()

            sub_start = sub_stop

    if buffered > 0:
        yield flush()


class TripleIndex:
    """
    The candidate triples of a neighbour list as one flat range
    ``[0, total)``, any slice of which can be turned into a
    :class:`TripleChunk` on its own.

    This is the vectorized enumeration behind
    :func:`_iter_triple_chunks_vectorized`, exposed as an object: building
    it costs ``O(n_pairs)`` (the oriented CSR and the row table), and
    ``chunk(start, stop)`` is a pure function of the slice, the positions
    and the lattice. A consumer can therefore *regenerate* the indices of a
    chunk during the backward pass (inside a ``torch.utils.checkpoint``
    region) instead of keeping them alive from the forward pass, which
    drops the retained index memory from ``O(total triples)`` to
    ``O(n_pairs)``.

    The positions and lattice are read at the time of each call, so a
    regenerated chunk is the chunk of the current geometry. Everything runs
    under ``torch.no_grad()``.

    Parameters
    ----------
    nat : int
        Total number of atoms of the flattened batch.
    real_i, real_j, real_shift : Tensor
        The real (unpadded) pairs of the list and their image shifts.
    atoms_per_system : int
        Atoms per system of the padded batch.
    cutoff : float
        Three-body cutoff.
    periodic : bool
        Whether the list is periodic (then chunks carry the shifts).
    """

    @torch.no_grad()
    def __init__(
        self,
        nat: int,
        real_i: Tensor,
        real_j: Tensor,
        real_shift: Tensor,
        atoms_per_system: int,
        cutoff: float,
        *,
        periodic: bool,
    ) -> None:
        self.device = real_i.device
        self.cutoff_squared = cutoff * cutoff
        self.periodic = periodic
        self.atoms_per_system = atoms_per_system
        self.offset, self.neighbours, self.neighbour_shift, degree = (
            _oriented_csr(nat, real_i, real_j, real_shift if periodic else None)
        )

        # Number of candidate triples contributed by each centre atom, and
        # the cumulative offset at which each centre's block starts in the
        # flat enumeration.
        ntri = degree * (degree - 1) // 2
        self.tri_offset = torch.cat([ntri.new_zeros(1), torch.cumsum(ntri, 0)])
        self.total = int(self.tri_offset[-1].item())
        self.first_row, self.row_offset = _triangle_rows(degree)

    @torch.no_grad()
    def chunk(
        self,
        start: int,
        stop: int,
        positions: Tensor,
        lattice: Tensor | None,
    ) -> TripleChunk:
        """
        The triples of the candidate slice ``[start, stop)`` that pass the
        cutoff (possibly none), for the given geometry.

        Parameters
        ----------
        start, stop : int
            Slice of ``[0, total)``.
        positions : Tensor
            ``(nat, 3)`` flattened positions.
        lattice : Tensor | None
            ``None`` for a molecule, else the cell(s), ``(3, 3)`` or
            ``(B, 3, 3)``.
        """
        shared_lattice, system_lattices = split_lattice(lattice)
        flat = torch.arange(start, stop, device=self.device)
        # `right=True` recovers, for each flat index, the centre whose
        # block it falls into: `tri_offset[c] <= flat < tri_offset[c+1]`.
        centre = torch.searchsorted(self.tri_offset, flat, right=True) - 1

        local_a, local_b = _triangular_indices(
            flat, self.row_offset, self.first_row[centre]
        )
        slot_a = self.offset[centre] + local_a
        slot_b = self.offset[centre] + local_b
        left, right = self.neighbours[slot_a], self.neighbours[slot_b]
        left_shift = self.neighbour_shift[slot_a]
        right_shift = self.neighbour_shift[slot_b]

        # Each triple takes the cell of its centre's system.
        cell = (
            system_lattices.index_select(0, centre // self.atoms_per_system)
            if system_lattices is not None
            else shared_lattice
        )
        keep = _within_cutoff(
            positions.index_select(0, centre),
            _translated(positions.index_select(0, left), left_shift, cell),
            _translated(positions.index_select(0, right), right_shift, cell),
            self.cutoff_squared,
        )
        left, centre, right = left[keep], centre[keep], right[keep]

        if not self.periodic:
            zero = _molecular_shift(left.shape[0], self.device)
            return TripleChunk(left, centre, right, zero, zero)
        return TripleChunk(
            left, centre, right, left_shift[keep], right_shift[keep]
        )


@torch.no_grad()
def _iter_triple_chunks_vectorized(
    nat: int,
    real_i: Tensor,
    real_j: Tensor,
    real_shift: Tensor,
    positions: Tensor,
    lattice: Tensor | None,
    atoms_per_system: int,
    cutoff: float,
    chunk_size: int | None = None,
) -> Iterator[TripleChunk]:
    """
    CUDA-favored generator: trades the loop path's per-centre-atom device
    syncs and kernel launches for one flat, whole-tensor enumeration over
    all centres at once, at the cost of two ``searchsorted`` calls per
    candidate triple to invert the flat numbering below -- a large net
    win on GPU, a net loss on CPU, hence the device dispatch in
    :func:`triples_from_neighborlist`.

    Every centre atom's ``C(degree, 2)`` unordered neighbour pairs are
    numbered ``0 .. ntri(centre) - 1``; concatenating those ranges across
    all centres gives one flat running index over every candidate triple
    (:func:`_triangle_rows`). The CSR adjacency and the row table are
    ``O(n_pairs)`` (not ``O(total triples)``), so they are built outside
    the chunk loop; only the flat-index range itself -- and everything
    derived from it (``searchsorted``, :func:`_triangular_indices`, the
    neighbour gather, the cutoff mask) -- is chunked to
    ``chunk_size``-sized slices (``chunk_size=None`` disables chunking,
    i.e. one slice covering the whole flat range).

    ``torch.searchsorted`` against the cumulative-sum offsets
    (``tri_offset``) recovers, for each flat index in the current chunk's
    range, which centre it belongs to; :func:`_triangular_indices` then
    finds the index's row, and with it the pair of local neighbour slots
    -- both operate correctly on a flat range that starts mid-block (not
    just at 0), since ``searchsorted`` only depends on where the range's
    values fall relative to the offsets, not on where the range itself
    starts;
    ``test_triple_chunks_vectorized_matches_unchunked`` in
    ``test/test_neighbor/test_triples.py`` exercises this directly with
    adversarial (e.g. ``chunk_size=1``) chunk boundaries.

    Like the loop path, all three sides -- ``i-j``, ``j-k``, ``i-k`` --
    are checked against ``cutoff`` explicitly before a candidate triple is
    kept.

    Data-dependent (``cumsum``, ``searchsorted``, a ``.item()`` for the
    total candidate-triple count, boolean masking), so -- like the loop
    path -- this always runs under ``torch.no_grad()``, outside any traced
    or transformed region: list construction, not consumption.

    Yields
    ------
    TripleChunk
        Chunks of real atom indices only, with
        ``idx_j`` the centre atom -- same convention as
        :func:`_iter_triple_chunks_loop` -- each with at most
        ``chunk_size`` rows (fewer once the cutoff mask is applied).
    """
    index = TripleIndex(
        nat,
        real_i,
        real_j,
        real_shift,
        atoms_per_system,
        cutoff,
        periodic=lattice is not None,
    )
    if index.total == 0:
        return
    step = index.total if chunk_size is None else max(chunk_size, 1)

    for start in range(0, index.total, step):
        chunk = index.chunk(
            start, min(start + step, index.total), positions, lattice
        )
        # Like the loop path, never yield a chunk the cutoff emptied.
        if chunk.idx_i.shape[0] > 0:
            yield chunk


@torch.no_grad()
def triples_from_neighborlist(
    nbl: NeighborList,
    structure: Structure,
    cutoff: float,
    *,
    chunk_size: int | None = 2_000_000,
) -> Iterator[TripleChunk]:
    """
    Enumerate candidate three-body triples ``(i, j, k)`` -- ``j`` the
    centre atom, ``i`` and ``k`` its neighbours, each at an image shift for
    a periodic list -- from a :class:`.NeighborList`'s real pairs, in
    fixed-size chunks. Every triple is yielded once, with the smallest of
    its three points as centre (see :func:`_oriented_csr`).

    ``nbl`` stores index data only, no positions (it is meant to be built
    once and reused across many evaluations), so the positions come from
    ``structure``: the system ``nbl`` was built from, or a later state of
    it that has drifted only slightly, exactly as
    :meth:`.NeighborList.stale` governs list reuse. The two real edges
    ``i-j``/``j-k`` are only guaranteed inside ``cutoff`` when ``nbl`` was
    built at exactly this ``cutoff`` with zero skin; a skinned or
    wider-cutoff list is reused on purpose (that is what ``skin`` is for),
    so this function checks all three sides -- ``i-j``, ``j-k``, and the
    third side ``i-k``, never implied by construction at all -- explicitly
    against ``cutoff`` before keeping a candidate triple.

    A batched ``structure`` gives triples in the list's flattened atom
    numbering, ``b * nat + i`` (see :class:`.NeighborList`). A list never
    pairs atoms of two systems, so no triple spans two systems either.

    The full candidate-triple list is never materialised at once: this
    returns a generator, chunked to at most ``chunk_size`` rows per yield,
    so that a caller's own working set stays bounded regardless of how
    large ``sum(degree * (degree - 1) / 2)`` (which grows as
    ``cutoff ** 6``) gets. The arguments are checked by the call itself,
    before the first chunk is asked for. Dispatches internally on the
    device of ``structure.positions``: a per-centre Python loop on CPU
    (:func:`_iter_triple_chunks_loop`), a vectorized enumeration over all
    centres at once everywhere else
    (:func:`_iter_triple_chunks_vectorized`). Both are proven, by test, to
    enumerate the exact same triple set for the same input -- see
    ``test_triple_list_loop_matches_vectorized`` in
    ``test/test_neighbor/test_triples.py``.

    This always runs under ``torch.no_grad()``, outside any traced or
    transformed region: it is list construction, not consumption. A caller
    consumes each yielded chunk with its own fixed-index-set,
    differentiable physics (e.g. ``tad_dftd3.damping.atm._atm_chunk_energy``),
    never branching on a chunk's contents.

    Parameters
    ----------
    nbl : NeighborList
        A pre-built neighbour list, molecular or periodic.
    structure : Structure
        The system ``nbl`` describes, single or batched. Its ``positions``
        and, for a periodic list, its ``lattice`` enter the enumeration.
    cutoff : float
        Three-body real-space cutoff. A triple contributes only if all of
        ``r_ij``, ``r_jk``, ``r_ik`` are inside it.
    chunk_size : int | None, optional
        Upper bound on how many candidate-triple rows are yielded per
        chunk. ``None`` disables chunking (a single chunk holding every
        candidate triple). Defaults to ``2_000_000``.

    Returns
    -------
    Iterator[TripleChunk]
        Chunks of real atom indices only -- no
        padding, unlike :class:`.NeighborList` itself -- each with at most
        ``chunk_size`` rows (fewer once the cutoff mask is applied).
        Concatenating every yielded chunk reproduces exactly what
        ``chunk_size=None`` returns in its one chunk.

    Raises
    ------
    ValueError
        ``nbl`` is not compatible with ``structure`` and ``cutoff`` (see
        :meth:`.NeighborList.check_compatible`), e.g. it was built for
        atoms of another shape or at a smaller cutoff.
    """
    nbl.check_compatible(structure, cutoff)

    # Flattened, the positions line up with a batched list's atom indices.
    positions = structure.positions.reshape(-1, 3)
    nat = positions.shape[0]
    real_i, real_j, real_shift = nbl.real_entries()
    # The triple search sorts, counts and gathers, and yields `int64`
    # triples: it works on `int64` indices (whole list, not chunked).
    real_i = real_i.long()
    real_j = real_j.long()
    lattice = structure.lattice if nbl.periodic else None
    atoms_per_system = structure.positions.shape[-2]

    iterate = (
        _iter_triple_chunks_loop
        if positions.device.type == "cpu"
        else _iter_triple_chunks_vectorized
    )
    return iterate(
        nat,
        real_i,
        real_j,
        real_shift,
        positions,
        lattice,
        atoms_per_system,
        cutoff,
        chunk_size,
    )


class _PairLookup:
    """
    The slot of a pair of a :class:`.NeighborList`, from its two atoms and
    the image shift of the second: a sorted table of integer keys, one per
    pair, searched with ``searchsorted``.

    A pair ``(a, b, S)`` and ``(b, a, -S)`` are the same pair, so both are
    first turned to point forward: ``a < b``, or ``a == b`` with ``S``
    forward (see :func:`._tiles._is_forward`). The shifts enter the key in
    a mixed radix sized to the shifts of the list, so a key is exact, and
    a shift beyond them is not a pair of the list.
    """

    def __init__(
        self,
        nat: int,
        slot: Tensor,
        idx_i: Tensor,
        idx_j: Tensor,
        shift: Tensor,
    ) -> None:
        self.nat = nat
        low, high, s = self._forward(idx_i, idx_j, shift)

        self.low = s.amin(0) if s.shape[0] > 0 else s.new_zeros(3)
        self.high = s.amax(0) if s.shape[0] > 0 else s.new_zeros(3)
        self.radix = self.high - self.low + 1

        largest = nat * nat * math.prod(self.radix.tolist())
        if largest >= 2**63:
            raise ValueError(
                f"The pairs of {nat} atoms with shifts from "
                f"{self.low.tolist()} to {self.high.tolist()} cannot be "
                "numbered with 64-bit integers; fold the positions into the "
                "cell before building the list."
            )

        self.keys, order = torch.sort(self._key(low, high, s))
        self.slot = slot[order]

    @staticmethod
    def _forward(
        idx_i: Tensor, idx_j: Tensor, shift: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        flip = (idx_i > idx_j) | ((idx_i == idx_j) & ~_is_forward(shift))
        first = torch.where(flip, idx_j, idx_i)
        second = torch.where(flip, idx_i, idx_j)
        return first, second, torch.where(flip.unsqueeze(-1), -shift, shift)

    def _key(self, first: Tensor, second: Tensor, shift: Tensor) -> Tensor:
        key = first * self.nat + second
        for axis in range(3):
            key = key * self.radix[axis] + (shift[:, axis] - self.low[axis])
        return key

    def find(
        self, idx_i: Tensor, idx_j: Tensor, shift: Tensor
    ) -> tuple[Tensor, Tensor]:
        """
        The slot of each pair ``(idx_i, idx_j, shift)``, and whether it is
        a pair of the list at all (where it is not, the slot is arbitrary).
        """
        first, second, s = self._forward(idx_i, idx_j, shift)
        inside = ((s >= self.low) & (s <= self.high)).all(-1)
        s = torch.minimum(torch.maximum(s, self.low), self.high)

        key = self._key(first, second, s)
        at = torch.searchsorted(self.keys, key).clamp(
            max=self.keys.shape[0] - 1
        )
        return self.slot[at], inside & (self.keys[at] == key)


class TripleList(Node):
    """
    Every triangle of the pairs of a :class:`.NeighborList`: three atoms (or
    images of atoms) of which each two are a pair of the list. Each triple
    holds its three atoms and its three sides, as slots of the list.

    It is built once, eagerly, from the list alone (:meth:`from_neighborlist`)
    and valid exactly as long as the list is: at the positions the list was
    built from, its pairs are every pair within ``cutoff + skin``, so the
    triples are every triple with all three sides within that reach; while
    no atom has moved by more than ``skin / 2`` (see :meth:`stale`), they
    still hold every triple with all sides within ``cutoff``. A consumer
    computes the quantities of the pairs once over the slots of
    :attr:`pairs` and masks each triple at its own cutoff by the distances
    of its sides, which only ever need the list's own data.

    Each triple is listed once, with the smallest of its three points as the
    centre ``idx_j`` (see :func:`_oriented_csr`), like
    :func:`triples_from_neighborlist`, and like the list, a batch is one flat
    system. This holds every triple within the reach, which grow as its
    sixth power per atom, at six integers each.

    A frozen :class:`~tad_mctc.tree.Node` of integer index data, without a
    gradient.

    Attributes
    ----------
    pairs : NeighborList
        The list the triples are built from.
    idx_i, idx_j, idx_k : Tensor
        ``(n_triples,)``, the atoms of each triple, ``idx_j`` the centre.
    side_ij, side_jk, side_ik : Tensor
        ``(n_triples,)``, the slot of :attr:`pairs` of each side.
    """

    pairs: NeighborList = child()
    idx_i: Tensor = child()
    idx_j: Tensor = child()
    idx_k: Tensor = child()
    side_ij: Tensor = child()
    side_jk: Tensor = child()
    side_ik: Tensor = child()

    @classmethod
    @torch.no_grad()
    def from_neighborlist(
        cls, nbl: NeighborList, *, chunk_size: int = 2_000_000
    ) -> TripleList:
        """
        The triangles of the pairs of `nbl`.

        Only the list enters, not a geometry: a candidate triple is two
        neighbours of a centre, and it is kept if the third side, between
        the two neighbours, is a pair of the list as well.

        Parameters
        ----------
        nbl : NeighborList
            The pairs.
        chunk_size : int, optional
            Upper bound on the candidate triples tested at once. Defaults
            to ``2_000_000``.

        Returns
        -------
        TripleList
            Every triangle of `nbl`, once.

        Raises
        ------
        ValueError
            If the atoms and image shifts of `nbl` are too many to number
            its pairs with 64-bit integers.
        """
        slot = nbl.mask.nonzero().squeeze(-1)
        real_i = nbl.idx_i.index_select(0, slot)
        real_j = nbl.idx_j.index_select(0, slot)
        nat = math.prod(nbl.numbers_shape)

        if nbl.periodic:
            real_shift = nbl.shift.index_select(0, slot).to(torch.long)
        else:
            real_shift = torch.zeros(
                slot.shape[0], 3, dtype=torch.long, device=slot.device
            )

        offset, neighbours, neighbour_shift, degree, entry = (
            _oriented_csr_entries(
                nat, real_i, real_j, real_shift if nbl.periodic else None
            )
        )
        neighbour_slot = slot[entry]
        lookup = _PairLookup(nat, slot, real_i, real_j, real_shift)

        # Candidate triples of all centres numbered by one flat index, as in
        # `TripleIndex`.
        ntri = degree * (degree - 1) // 2
        tri_offset = torch.cat([ntri.new_zeros(1), torch.cumsum(ntri, 0)])
        total = int(tri_offset[-1].item())
        first_row, row_offset = _triangle_rows(degree)

        parts: list[tuple[Tensor, ...]] = []
        step = max(chunk_size, 1)
        for start in range(0, total, step):
            flat = torch.arange(
                start, min(start + step, total), device=slot.device
            )
            centre = torch.searchsorted(tri_offset, flat, right=True) - 1
            local_a, local_b = _triangular_indices(
                flat, row_offset, first_row[centre]
            )
            slot_a = offset[centre] + local_a
            slot_b = offset[centre] + local_b
            left, right = neighbours[slot_a], neighbours[slot_b]

            # from the first neighbour to the second: `x_k + S_k - x_i - S_i`
            shift = neighbour_shift[slot_b] - neighbour_shift[slot_a]
            side_ik, found = lookup.find(left, right, shift)

            parts.append(
                (
                    left[found],
                    centre[found],
                    right[found],
                    neighbour_slot[slot_a][found],
                    neighbour_slot[slot_b][found],
                    side_ik[found],
                )
            )

        if parts:
            fields = [torch.cat(column) for column in zip(*parts)]
        else:
            fields = [slot.new_zeros(0) for _ in range(6)]

        return cls(
            pairs=nbl,
            idx_i=fields[0],
            idx_j=fields[1],
            idx_k=fields[2],
            side_ij=fields[3],
            side_jk=fields[4],
            side_ik=fields[5],
        )

    @property
    def reach(self) -> float:
        """The longest side of the triples at the positions the list was
        built from: its ``cutoff + skin``."""
        return self.pairs.cutoff + self.pairs.skin

    def check_compatible(self, structure: Structure, cutoff: float) -> None:
        """
        Raise unless these triples are compatible with `structure` and a
        consumer's `cutoff`, see :meth:`.NeighborList.check_compatible`.
        """
        self.pairs.check_compatible(structure, cutoff)

    def stale(self, structure: Structure) -> Tensor:
        """Whether the list, and with it these triples, should be rebuilt,
        see :meth:`.NeighborList.stale`."""
        return self.pairs.stale(structure)
