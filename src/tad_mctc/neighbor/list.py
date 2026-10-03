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
Neighbour search: the padded neighbour list
===========================================

A :class:`.NeighborList` holds integer index data only, so it carries no
gradient and can be built once and reused across many evaluations. It is
built by :func:`build_neighborlist` (one cutoff) or
:func:`build_neighborlists` (several cutoffs, one shared tile traversal).

The search never tests every atom pair. It runs in three steps:

1. Atoms are grouped into spatially bounded tiles of at most ``tile``
   atoms, cut out of a fine grid of bins. A cut along a Morton (Z-order)
   curve would give a smaller *median* tile, but the curve jumps at
   octree boundaries, so a few tiles end up spanning most of the box, and
   the search stencil must be wide enough for the *largest* tile. Cutting
   from bins bounds every tile by one bin diagonal, so the stencil does
   not grow with system size.
2. Tile pairs within the cutoff are found by an exact bounding-box test,
   which rules out most of the system before any atom distance is
   computed.
3. Only atom pairs from surviving tile pairs are checked exactly, then
   compacted into the padded list.

The tiles are internal (:mod:`._tiles`), since they work on ghost pools
and concatenated batches rather than a :class:`.Structure`; the ``tile``
argument of :func:`build_neighborlist` is the only tuning knob.

The module has two halves. *Construction*, including the tile search
(:mod:`._tiles`), is data-dependent (`nonzero`, boolean-mask
indexing, a Python-level padding decision) and always runs under
``torch.no_grad()``; the resulting :class:`.NeighborList` is then meant to
be *consumed* by fixed-shape tensor algebra that never branches on its
contents, so that it survives autograd, ``vmap`` and
``torch.compile(fullgraph=True)``. See ``ncoord/common.py`` for the
consumer side of that split.

Padded slots are routed to atom index ``nat`` -- one past the last real
atom -- rather than to a real atom index. The consumer appends one
phantom row to ``positions`` (a constant with no gradient path) before
indexing with ``idx_i``/``idx_j``, so a padded slot can never contaminate
a derivative. Indexing a real atom instead would make that atom's
gradient depend on how much padding happened to land on it, which is an
implementation detail of the list, not physics.

Example
-------
>>> import torch
>>> from tad_mctc.io.structure import Structure
>>> from tad_mctc.neighbor.list import build_neighborlist
>>>
>>> structure = Structure(
...     numbers=torch.tensor([8, 1, 1, 2]),
...     positions=torch.tensor([
...         [0.0, 0.0, 0.0],
...         [1.0, 0.0, 0.0],
...         [0.0, 1.0, 0.0],
...         [10.0, 10.0, 10.0],
...     ]),
... )
>>> nbl = build_neighborlist(structure, cutoff=2.0, tile=2)
>>> bool(nbl.mask.sum() > 0)
True
>>> bool((nbl.idx_i[~nbl.mask] == structure.numbers.shape[0]).all())
True
"""

from __future__ import annotations

import contextlib
import math
from typing import TYPE_CHECKING, Literal, NamedTuple, Protocol

import torch

from ..autograd.checks import is_functorch_tensor, is_vmapped
from ..autograd.unwrap import unwrap_gradtracking
from ..batch import real_atoms
from ..typing import DD, Self, Tensor, TensorLike
from . import _distance_kernels, _native
from ._distance_kernels import DistanceKernelName
from ._tiles import Tiles, _integer_box, _is_forward, _ragged_runs, tile_pairs
from .images import (
    _image_rings,
    _validate_lattice_periodic,
    wrap_to_central_cell,
)

if TYPE_CHECKING:
    from collections.abc import Generator

    from ..io.structure import Structure

__all__ = [
    "NeighborList",
    "build_neighborlist",
    "build_neighborlists",
    "estimate_neighborlist_memory",
]


# Pair counts are rounded up to a multiple of this many slots, so that a
# trajectory whose pair count wobbles from step to step reuses one
# compiled graph instead of recompiling on every rebuild.
_CAPACITY_BUCKET = 4096

# A batch of cells is searched in chunks of systems whose estimated pair
# count stays below this, so the peak memory of the search does not grow
# with the number of cells. It is a memory budget, not a speed setting, and
# never changes the result. The search holds integer index data only, so
# its peak per pair of the finished list is the same on every device and
# float dtype: 83 bytes on CPU and CUDA, float32 and float64, for batches
# of 864-atom cells at 15-40 Bohr. Only on CUDA at 10 Bohr, where fixed
# costs weigh more, does it rise to about 150 bytes. This budget thus keeps
# a chunk's peak near 170-300 MB. Smaller chunks only cost the per-search
# overhead of a few milliseconds.
_PAIRS_PER_CELL_SEARCH = 2_000_000


class _StageHook(Protocol):
    """Context manager factory entered around one labelled step of a
    neighbour-list build (see :func:`_build_neighborlists`)."""

    def __call__(
        self, label: str
    ) -> contextlib.AbstractContextManager[object]: ...


def _no_stage_hook(label: str) -> contextlib.AbstractContextManager[None]:
    """The default stage hook: runs each step as it is."""
    return contextlib.nullcontext()


class NeighborList(TensorLike):
    """
    Fixed-capacity, padded neighbour list.

    Holds integer index data only, so it carries no gradient and can be
    built once and reused across many evaluations.

    Consumer rule: each entry stands for the pair in both directions,
    ``(idx_i, idx_j, +shift)`` and ``(idx_j, idx_i, -shift)``, and each
    pair is stored once. Which of its two atoms is ``idx_i`` is not
    specified. For a self-image entry (``idx_i == idx_j``, only possible
    in a periodic list), the two directions are the two distinct
    neighbours on either side of the atom, both held by the one stored
    entry -- see :func:`._ghost_pool`. A consumer must add its
    per-pair contribution to both ``idx_i`` and ``idx_j`` unconditionally,
    with no special case for ``idx_i == idx_j``, or a self-image
    neighbour silently contributes only half of what it should.

    Padding atoms (``numbers == 0``) have no pairs, in a single structure
    as in a batch. A list for a batched structure (``numbers`` of shape
    ``(..., nat)``) indexes its padded atoms flattened: atom ``i`` of
    system ``b`` is ``b * nat + i``, and a consumer finds the system of a
    pair as ``idx_i // nat``.

    Attributes
    ----------
    idx_i : Tensor
        ``(capacity,)``, dtype :data:`_IDX_DTYPE` (``torch.long``), first
        atom of each pair. Padding is routed to the index one past the
        last atom (``nat``, or the flattened batch's atom count). Fixed at ``torch.long`` because
        ``torch.autograd``'s backward pass runs pair-indexed
        ``index_add``/``gather`` calls through this tensor, and
        ``gather`` requires an ``int64`` index -- see :data:`_IDX_DTYPE`.
    idx_j : Tensor
        ``(capacity,)``, dtype :data:`_IDX_DTYPE`, second atom of each
        pair.
    shift : Tensor
        ``(capacity, 3)``, dtype :data:`_SHIFT_DTYPE` (``torch.int16``),
        integer image shift. For a molecular list it is all zeros, stored
        as a zero-stride view of a single row (see
        :func:`_molecular_shift`), so it takes no memory per pair. Never
        reaches ``gather``/``index_add`` (it is read, not used as an
        index), so it carries none of ``idx_i``/``idx_j``'s ``int64``
        constraint -- see :data:`_SHIFT_DTYPE` for its range.
    mask : Tensor
        ``(capacity,)``, ``True`` for a real pair.
    periodic : bool
        Whether this list was built with a lattice. A plain Python
        ``bool``, fixed at construction, so a consumer can branch on it
        (as :meth:`check_compatible` does to decide whether a lattice is
        required) without reading any tensor's value. The lattice itself
        is never stored publicly: consumers take it as a call-time
        argument (see :mod:`tad_mctc.ncoord.common`), so that
        differentiating with respect to it sees the argument, not a
        build-time snapshot with no gradient history.
    periodic_axes : Tensor | None
        Boolean mask of the axes the list was built periodic along, the
        structure's own ``periodic`` (``(3,)`` or per system
        ``(..., 3)``), ``None`` for a molecular list. The pairs' ``shift`` already
        encodes these axes (including the fold into the central cell), so
        the list only describes a structure with exactly this mask.
    cutoff : float
        Real-space cutoff this list was built for.
    skin : float
        Extra radius searched beyond ``cutoff``, so that a small drift in
        ``positions`` does not immediately invalidate the list.
    numbers_shape : tuple[int, ...]
        Shape of the ``numbers`` of the structure the list was built
        from, ``(nat,)`` or ``(..., nat)`` for a batch. A plain Python
        tuple, so :meth:`check_compatible` compares it under any
        transform.
    overflow : bool
        ``True`` when an explicitly requested ``capacity`` was too small
        to hold every real pair. The list is then truncated silently
        *for physics*, but not for the caller: check this flag and
        rebuild with a larger capacity rather than trusting the energy.
    """

    __slots__ = [
        "idx_i",
        "idx_j",
        "shift",
        "mask",
        "periodic",
        "cutoff",
        "skin",
        "overflow",
        "_build_positions",
        "periodic_axes",
        "numbers_shape",
        "_build_lattice",
    ]

    def __init__(
        self,
        idx_i: Tensor,
        idx_j: Tensor,
        shift: Tensor,
        mask: Tensor,
        build_positions: Tensor,
        cutoff: float,
        skin: float,
        overflow: bool,
        lattice: Tensor | None = None,
        periodic_axes: Tensor | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(device, dtype)

        self.idx_i = idx_i
        self.idx_j = idx_j
        self.shift = shift
        self.mask = mask
        # Copied, not referenced: an MD step that updates the caller's
        # positions in place (`positions += velocity * dt`) would otherwise
        # move this snapshot along with them, and `stale()` would never
        # see any drift.
        self._build_positions = build_positions.detach().clone()
        self.cutoff = cutoff
        self.skin = skin
        self.overflow = overflow
        # Derived, not a separate constructor argument: a periodic list is
        # exactly one built with a lattice, and the two must never disagree.
        self.periodic = lattice is not None
        self.numbers_shape = tuple(build_positions.shape[:-1])
        # Detached and copied, like `_build_positions`: this snapshot exists
        # only for `stale()`'s own cell-change check, never as a gradient
        # path, and must not follow an in-place change of the caller's
        # lattice. Consumers receive the lattice as a call-time argument
        # instead (`tad_mctc.ncoord.common`), which is what makes `jacrev`
        # with respect to it see a live argument rather than this snapshot.
        self._build_lattice = (
            None if lattice is None else lattice.detach().clone()
        )
        self.periodic_axes = periodic_axes

    def to(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Self:
        """
        Copy this list to a new device and/or floating dtype.

        Overrides :meth:`TensorLike.to`, which would cast every slot to
        the requested dtype, including the integer index data
        (``idx_i``, ``idx_j``, ``shift``) and the boolean ``mask``. Those
        slots move device only; ``dtype`` applies to the floating-point
        slots (the build-time lattice and position snapshots).

        Parameters
        ----------
        device : torch.device | None, optional
            Device to move every slot to. ``None`` keeps the current
            device.
        dtype : torch.dtype | None, optional
            Floating dtype for the floating-point slots. ``None`` keeps
            the current dtype.

        Returns
        -------
        NeighborList
            A copy on the requested device and dtype.
        """
        target_device = device if device is not None else self.device
        target_dtype = dtype if dtype is not None else self.dtype
        if self.device == target_device and self.dtype == target_dtype:
            return self

        lattice = (
            None
            if self._build_lattice is None
            else self._build_lattice.to(
                device=target_device, dtype=target_dtype
            )
        )
        periodic_axes = (
            None
            if self.periodic_axes is None
            else self.periodic_axes.to(device=target_device)
        )

        # Moving a zero-stride view copies it into a full tensor, so a
        # molecular list's shift is created anew on the target device.
        shift = (
            self.shift.to(device=target_device)
            if self.periodic
            else _molecular_shift(self.shift.shape[0], target_device)
        )

        return type(self)(
            idx_i=self.idx_i.to(device=target_device),
            idx_j=self.idx_j.to(device=target_device),
            shift=shift,
            mask=self.mask.to(device=target_device),
            build_positions=self._build_positions.to(
                device=target_device, dtype=target_dtype
            ),
            cutoff=self.cutoff,
            skin=self.skin,
            overflow=self.overflow,
            lattice=lattice,
            periodic_axes=periodic_axes,
            device=target_device,
            dtype=target_dtype,
        )

    def type(self, dtype: torch.dtype) -> Self:
        """
        Copy this list to a new floating dtype, keeping its device.

        See :meth:`to` for which slots ``dtype`` actually applies to.

        Parameters
        ----------
        dtype : torch.dtype
            Floating dtype for the floating-point slots.

        Returns
        -------
        NeighborList
            A copy with the requested dtype.
        """
        return self.to(dtype=dtype)

    def check_compatible(self, structure: Structure, cutoff: float) -> None:
        """
        Raise unless this list is compatible with ``structure`` and
        ``cutoff``, i.e. summing over it cannot silently drop or add a
        pair.

        The periodic axes must match ``structure.periodic`` exactly, not
        just cover it: each pair's ``shift`` already contains the fold
        into the central cell along every axis the list was built
        periodic along, so extra axes add images that do not exist.

        The axis comparison reads the values of ``structure.periodic``,
        so it is skipped under ``torch.compile``, ``vmap`` and ``jacrev``.
        Call this once eagerly before transforming an evaluation. Every
        other check reads plain Python values and always runs.

        Parameters
        ----------
        structure : Structure
            The system the list is about to be used with.
        cutoff : float
            Real-space cutoff of the consumer, e.g. a model's ``cutoff``.

        Raises
        ------
        ValueError
            ``structure.numbers`` has a different shape than at build
            time; the list is periodic while ``structure`` has no lattice,
            or the reverse; ``self.cutoff`` is smaller than ``cutoff``;
            ``self.overflow`` is ``True``; or the list's periodic axes
            differ from ``structure.periodic``.
        """
        if tuple(structure.numbers.shape) != self.numbers_shape:
            raise ValueError(
                f"`nbl` was built for atoms of shape {self.numbers_shape}, "
                "but `structure.numbers` has shape "
                f"{tuple(structure.numbers.shape)}; its indices would point "
                "at the wrong atoms. Rebuild it from this `Structure`."
            )

        if self.periodic and structure.lattice is None:
            raise ValueError(
                "`nbl` was built with a lattice, but `structure.lattice` is "
                "`None`; use the `Structure` the list was built from."
            )
        if not self.periodic and structure.lattice is not None:
            raise ValueError(
                "`structure.lattice` is set, but `nbl` is a molecular list "
                "(built without a lattice); rebuild it from this "
                "`Structure`."
            )

        if self.cutoff < cutoff:
            raise ValueError(
                f"`nbl.cutoff` ({self.cutoff}) is smaller than the "
                f"consumer's cutoff ({cutoff}); such a list is missing "
                "pairs. Rebuild it with at least this cutoff."
            )

        if self.overflow:
            raise ValueError(
                "`nbl.overflow` is `True`: the explicit `capacity` it was "
                "built with was too small to hold every real pair, so its "
                "list is silently truncated. Rebuild it with a "
                "larger `capacity`, or omit `capacity` to let it auto-size."
            )

        if not self.periodic:
            return

        # `Structure` fills in a mask whenever it has a lattice, and a
        # periodic list always records its axes.
        periodic = structure.periodic
        assert periodic is not None and self.periodic_axes is not None
        if is_functorch_tensor(periodic):
            return

        list_axes = self.periodic_axes.to(periodic.device)
        if (periodic != list_axes).any():
            raise ValueError(
                f"`nbl.periodic_axes` ({list_axes.tolist()}) differ from "
                f"the structure's periodic axes ({periodic.tolist()}); the "
                "list would silently add or drop periodic images. Rebuild "
                "it from this `Structure`."
            )

    @torch.no_grad()
    def stale(self, structure: Structure) -> Tensor:
        """
        Whether this list should be rebuilt: any atom has drifted more
        than ``skin / 2`` since it was built, or, for a periodic list,
        the cell has changed at all.

        With ``skin == 0.0``, this always returns ``True``: there is no
        buffer to spend, so every evaluation should rebuild.

        Parameters
        ----------
        structure : Structure
            The current state of the system the list was built from, with
            the same atoms. Its ``positions`` are compared with those at
            build time, and for a periodic list its ``lattice`` too (a
            changed cell moves every pair distance, even where atoms have
            not moved in fractional coordinates).

        Returns
        -------
        Tensor
            Scalar boolean, one for the whole batch of a batched list.

        Raises
        ------
        ValueError
            The list is periodic and ``structure.lattice`` is ``None``.
        """
        positions = structure.positions
        if self.skin == 0.0:
            return torch.tensor(True, device=positions.device)

        displacement = positions - self._build_positions
        lattice_changed = torch.tensor(False, device=positions.device)

        if self.periodic:
            lattice = structure.lattice
            if lattice is None:
                raise ValueError(
                    "`structure.lattice` is `None`, but this neighbour list "
                    "is periodic; pass the current state of the `Structure` "
                    "it was built from."
                )
            assert (
                self.periodic_axes is not None
                and self._build_lattice is not None
            ), "a periodic list always records its axes and lattice"
            # `displacement` is deliberately *not* wrapped to its minimum
            # image: `shift` counts cells relative to the caller's own
            # positions, so an atom re-wrapped into the cell (moved by a
            # whole lattice vector) leaves every stored shift of its pairs
            # off by that vector. The skin guarantee holds for the raw
            # displacement a consumer actually sees, so re-wrapping costs
            # one rebuild; a trajectory that never re-wraps its atoms
            # keeps the full benefit of the skin.
            # Conservative: any change to the cell counts as stale, since a
            # strained cell changes every pair distance, not just the ones
            # belonging to atoms that moved in fractional coordinates.
            lattice_changed = torch.any(lattice != self._build_lattice)

        drift = displacement.norm(dim=-1).max()
        return (drift > self.skin / 2) | lattice_changed

    def real_entries(self) -> tuple[Tensor, Tensor, Tensor]:
        """
        The real (unpadded) entries of the list.

        Data-dependent in size, so for construction-side code only, not
        for a ``vmap``-ed or compiled consumer.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            ``idx_i`` and ``idx_j``, both ``(n_real,)``, and ``shift``,
            ``(n_real, 3)``. A molecular list's shift is again a zero-stride
            view of one zero row.
        """
        # One `nonzero` shared by every gather, instead of one per masked
        # index.
        real = self.mask.nonzero().squeeze(-1)
        idx_i = self.idx_i.index_select(0, real)
        idx_j = self.idx_j.index_select(0, real)
        if self.periodic:
            shift = self.shift.index_select(0, real)
        else:
            shift = _molecular_shift(real.shape[0], self.device)
        return idx_i, idx_j, shift


# Dtype of `idx_i`/`idx_j`. It must stay `torch.long`: the backward pass
# threads the indices through `gather` (via `index_add`'s backward), and
# `gather` requires `int64` indices. A narrower dtype works in forward mode
# but raises `RuntimeError: gather(): Expected dtype int64 for index` under
# `torch.autograd`, `jacrev` and `vmap`.
_IDX_DTYPE = torch.long

# Dtype of `shift`. Never used as a `gather`/`index_add` index -- only
# read and added into positions -- so it carries none of `_IDX_DTYPE`'s
# `int64` constraint and can be as small as the values it holds allow.
# A shift is relative to the caller's own, unwrapped positions, so it
# counts the image rings the cutoff reaches *plus* how many cells apart
# the two atoms are written. The second part grows without bound in an
# unwrapped MD trajectory, where atoms diffuse through many cells, and
# `torch.int8` (up to 127 cells) is too narrow for it. `torch.int16`
# covers 32767 cells; `_neighbor_list` raises for anything beyond that
# rather than truncating a shift silently.
_SHIFT_DTYPE = torch.int16

# Bytes per capacity slot: `idx_i` + `idx_j` (`_IDX_DTYPE`, 8 B each) and
# `mask` (torch.bool, 1 B), plus `shift` (`_SHIFT_DTYPE`, 3 x 2 B) for a
# periodic list only -- a molecular list's all-zero shift is a zero-stride
# view (see `_molecular_shift`). Exact and independent of
# `positions.dtype`, since none of these is a floating dtype. Kept in sync
# with `_IDX_DTYPE`/`_SHIFT_DTYPE` by hand, so that they stay plain,
# checkable constants.
_MOLECULAR_BYTES_PER_SLOT = 17
_PERIODIC_BYTES_PER_SLOT = 23


def _molecular_shift(capacity: int, device: torch.device) -> Tensor:
    """
    The all-zero ``(capacity, 3)`` shift of a molecular list, as a
    zero-stride view of one row: it reads like a full tensor but stores
    one row, instead of one per pair.
    """
    zero_row = torch.zeros(1, 3, dtype=_SHIFT_DTYPE, device=device)
    return zero_row.expand(capacity, 3)


def estimate_neighborlist_memory(capacity: int, *, periodic: bool) -> int:
    """
    Exact size, in bytes, of a padded neighbor list's own tensors.

    ``capacity`` is the padded list size -- the number of
    ``idx_i``/``idx_j``/``shift``/``mask`` slots (e.g. an already-built
    list's ``nbl.idx_i.shape[0]``, or a size chosen up front via
    ``build_neighborlist(..., capacity=...)``), not the number of real
    pairs (``nbl.mask.sum()``): padded slots are allocated too.

    Covers only ``NeighborList``'s own ``capacity``-length tensors, not
    the memory ``build_neighborlist`` transiently uses while constructing
    them, nor whatever a consumer (e.g. ``cn_eeq``) allocates on top of
    the list afterwards -- both depend on the consumer and on
    implementation details that can change independently of this exact
    arithmetic. It also excludes the build-time position snapshot
    (``NeighborList``'s private ``_build_positions``), which scales with
    ``nat``, not ``capacity``.

    This function does not predict ``capacity`` from ``positions`` and
    ``cutoff``. Pairs-per-atom depends on conformer shape, not just atom
    count and cutoff: a near-linear chain thinner than the cutoff sees a
    long cylinder of neighbours along its whole length, an order of
    magnitude more pairs per atom than a globular structure at the same
    cutoff would. Measure or bound ``capacity`` first (build once at a
    representative size, or use a prior run's ``idx_i.shape[0]``) rather
    than guessing it here.

    Parameters
    ----------
    capacity : int
        Padded list capacity, i.e. ``idx_i.shape[0]``.
    periodic : bool
        Whether the list is periodic (``nbl.periodic``). Only a periodic
        list stores a shift per pair.

    Returns
    -------
    int
        Exact combined size of ``idx_i``, ``idx_j``, ``shift`` and
        ``mask`` at this capacity, in bytes, apart from the constant
        six bytes of a molecular list's single zero shift row.

    Raises
    ------
    ValueError
        If ``capacity`` is negative.

    Example
    -------
    >>> from tad_mctc.neighbor.list import estimate_neighborlist_memory
    >>> estimate_neighborlist_memory(11_735_040, periodic=False)
    199495680
    >>> estimate_neighborlist_memory(11_735_040, periodic=True)
    269905920
    """
    if capacity < 0:
        raise ValueError(f"`capacity` must be >= 0, got {capacity}")

    if periodic:
        return capacity * _PERIODIC_BYTES_PER_SLOT
    return capacity * _MOLECULAR_BYTES_PER_SLOT


def _separate_systems(
    positions: Tensor, batch: Tensor, cutoff: float
) -> tuple[Tensor, Tensor]:
    """
    Shift each system apart along one axis by more than any pair spans,
    so that a single (unbatched) :class:`.Tiles` built on the
    concatenated, shifted cloud cannot place atoms from two different
    systems in the same bin, and :func:`.tile_pairs`'s exact bounding-box
    screen rejects every cross-system tile pair on its own.

    Batching by folding a system id straight into the bin key would need
    :class:`.Tiles` and :func:`.tile_pairs` to know about system ids at
    all. Shifting the systems apart in space first needs neither: it
    reuses both entirely unchanged. Two systems in a batch can overlap in
    space in the caller's own coordinates -- binning that concatenated
    cloud directly would invent pairs between molecules that merely
    happen to sit on top of each other.

    Only used to decide *which* tile pairs to search; the exact
    atom-pair distance filter that follows always uses the caller's
    original, unshifted ``positions``, so the shift never reaches a real
    distance calculation.

    The shifted coordinates are float64 whatever the dtype of
    ``positions``. They grow with the number of systems, and the tile
    screen compares them directly: in float32, a batch of 20000 systems
    reaches coordinates of about 4e7 Bohr, spaced 4 Bohr apart, and a
    real pair whose ends round apart past the cutoff would be dropped.

    This shift's entire correctness argument rests on one invariant that
    it cannot enforce by itself: once :class:`.Tiles` bins the shifted
    cloud, no bin may contain atoms from two different systems. So the
    *guaranteed minimum gap* between two adjacent systems is returned
    alongside the coordinates, and :func:`_tiles_of_separated_systems`
    passes it to :class:`.Tiles` as a ``max_bin_width`` constraint: without it,
    :class:`.Tiles`' coarsening loop could widen a bin past the gap and
    silently merge atoms from two different systems into it.

    Writing ``extent`` for the whole cloud's own spread along axis 0, and
    ``gap`` for the value returned here, the shift is
    ``separation = extent + gap`` and the guarantee is

    ``gap_k = separation + min_{k+1} - max_k >= separation - extent == gap``

    (both ``min_{k+1}`` and ``max_k``, in unshifted coordinates, lie
    inside the cloud's own axis-0 range, so their difference is at most
    ``extent``). Equality is reachable: system ``k`` sitting entirely at
    the cloud's axis-0 maximum and system ``k + 1`` entirely at its
    minimum. Two details of that formula are load-bearing:

    * ``extent`` is this axis's own spread, **not** the largest spread
      over all three axes. Only axis 0 is ever shifted along, so only
      axis 0 bounds ``max_k - min_{k+1}``; using the three-axis maximum
      would let one atom far out along an unrelated axis inflate the
      shift, and with it the bin count forced by the constraint, without
      bound.
    * ``gap`` is floored at ``extent`` rather than being simply
      ``2 * cutoff``. Correctness only needs ``gap >= 2 * cutoff``, but
      cost needs ``gap`` not to be small next to ``extent``: the
      constrained axis is forced to hold about
      ``n_systems * (extent + gap) / gap`` bins, which is bounded by
      ``2 * n_systems`` exactly when ``gap >= extent``, and grows without
      limit as ``gap / extent`` shrinks. Widening the shift to buy a
      wider permitted bin is the cheaper trade.

    Parameters
    ----------
    positions : Tensor
        Cartesian coordinates, shape ``(nat, 3)``, of every system
        concatenated together.
    batch : Tensor
        ``(nat,)``, the system index owning each atom.
    cutoff : float
        The largest cutoff this search will use. Must be positive: at
        zero there is no gap left to guarantee (see
        :func:`_check_batched_search_radius`).

    Returns
    -------
    tuple[Tensor, Tensor]
        Shifted coordinates, shape ``(nat, 3)``, float64, for binning
        only, and the guaranteed minimum gap between two adjacent systems
        along axis 0, a float64 scalar tensor, always ``>= 2 * cutoff``.
    """
    positions = positions.to(torch.float64)

    # A batch of padding atoms only (or an empty ghost pool) has no
    # extent to reduce over; `max()` of an empty tensor raises.
    if positions.shape[0] == 0:
        extent = positions.new_zeros(())
    else:
        extent = positions[:, 0].max() - positions[:, 0].min()

    gap = torch.clamp(extent, min=2.0 * cutoff)
    separation = extent + gap

    shift_axis = torch.zeros(3, dtype=positions.dtype, device=positions.device)
    shift_axis[0] = separation

    system = batch.unsqueeze(-1).to(positions.dtype)
    return positions + system * shift_axis, gap


class _Padding(NamedTuple):
    """
    How to pad the pairs of a list: to ``capacity`` slots (``None``: the pair count
    rounded up to a multiple of :data:`_CAPACITY_BUCKET`), with ``value``,
    the phantom atom index, in every padded slot.
    """

    capacity: int | None
    value: int


class _Pairs(NamedTuple):
    """
    The atom pairs of one threshold.

    ``idx_i`` and ``idx_j`` start with ``min(n_found, len(idx_i))`` real
    pairs, followed by padding if the search was asked to pad. ``n_found``
    is how many pairs the search found; more than ``len(idx_i)`` means a
    fixed capacity was too small and the pairs are truncated.
    """

    idx_i: Tensor
    idx_j: Tensor
    n_found: int


def _capacity_for(n_found: int, capacity: int | None) -> int:
    """
    Number of slots of a padded list that holds ``n_found`` pairs:
    ``capacity`` if fixed, otherwise ``n_found`` rounded up to a multiple
    of :data:`_CAPACITY_BUCKET`. ``_native_pairs.cpp`` applies the same
    rule.
    """
    if capacity is not None:
        return capacity
    return -(-n_found // _CAPACITY_BUCKET) * _CAPACITY_BUCKET


def _join_pairs(
    fragments_i: list[Tensor],
    fragments_j: list[Tensor],
    padding: _Padding | None,
    device: torch.device,
) -> _Pairs:
    """
    Concatenate fragments of a list's pairs into one :class:`_Pairs`, padded
    to its capacity if ``padding`` is given.

    The padding is part of the same concatenation, so every pair is copied
    once. Each fragment list is emptied as soon as it is joined, to lower
    the peak memory of a large build.
    """
    n_found = sum(int(fragment.shape[0]) for fragment in fragments_i)
    empty = torch.zeros(0, dtype=_IDX_DTYPE, device=device)

    if padding is None:
        tail, n_kept = empty, n_found
    else:
        capacity = _capacity_for(n_found, padding.capacity)
        n_kept = min(n_found, capacity)
        tail = torch.full(
            (capacity - n_kept,), padding.value, dtype=_IDX_DTYPE, device=device
        )

    def join(fragments: list[Tensor]) -> Tensor:
        # `empty` keeps `torch.cat` valid when there are no fragments.
        if n_kept < n_found:
            # Truncated to a too-small fixed capacity, so there is no tail.
            joined = torch.cat([empty, *fragments])[:n_kept]
        else:
            joined = torch.cat([empty, *fragments, tail])
        fragments.clear()
        return joined

    idx_i = join(fragments_i)
    idx_j = join(fragments_j)
    return _Pairs(idx_i, idx_j, n_found)


def _atom_pairs_within_thresholds(
    tiles: Tiles,
    tile_a: Tensor,
    tile_b: Tensor,
    reference_positions: Tensor,
    thresholds: tuple[float, ...],
    max_block: int = 2_000_000,
    distance_kernel: DistanceKernelName | None = None,
    pair_filter: Literal["native", "python"] | None = None,
    anchor_atoms: Tensor | None = None,
    padding: _Padding | None = None,
) -> list[_Pairs]:
    """
    Exact atom pairs for every threshold in ``thresholds``, from
    one shared, chunked scan over the candidate tile pairs
    ``(tile_a, tile_b)``.

    Every threshold is checked against the same distance matrix inside
    each chunk, since a candidate tile pair surviving the search that
    produced ``tile_a``/``tile_b`` (at the largest threshold) is a
    superset of what any smaller threshold needs; recomputing distances
    per threshold would throw that sharing away.

    Two independent choices are involved, never conflated with each other:
    which implementation performs the filter-and-compact step at all
    (``pair_filter`` -- native C++ or pure Python), and, only when the
    Python implementation runs, which formula it uses to compute a tile
    pair's distances (``distance_kernel`` -- see
    :mod:`._distance_kernels`). On CPU, with both left at their automatic
    default and float32 or float64 positions, this delegates to
    :func:`tad_mctc.neighbor._native.atom_pairs_within_thresholds_native`
    first -- an optional, OpenMP-parallel C++ implementation of exactly
    this filter-and-compact step, falling back to the pure-Python path
    below whenever the native extension is unavailable. See
    :mod:`tad_mctc.neighbor._native`'s module docstring for why, and for
    the exact-output-order contract the two paths must both honour.

    Parameters
    ----------
    tiles : Tiles
        Tiles built by :class:`.Tiles`.
    tile_a, tile_b : Tensor
        Candidate tile pairs from :func:`.tile_pairs`, ``a <= b``.
    reference_positions : Tensor
        Positions used for the exact distance filter. This is the
        caller's original geometry, even when ``tiles`` was built on a
        shifted copy for batching.
    thresholds : tuple[float, ...]
        One cutoff (already including any skin) per requested list.
    max_block : int, optional
        Roughly how many atom-pair entries to materialise per chunk. Only
        used by the pure-Python path; the native path has no chunk-sized
        intermediate to bound. The default is a memory bound, not a tuned
        optimum: a chunk's squared distances and boolean masks take some
        20 bytes per entry in float64, a few tens of MB per chunk. It
        leans towards the GPU, where this path always runs and time grows
        steeply below it with the number of chunks. On a CPU the path is
        only the fallback for a missing native extension, and smaller
        chunks run faster there.
    distance_kernel : DistanceKernelName | None, optional
        Force a specific :class:`._distance_kernels.DistanceKernel` by
        name (``"triton"``, ``"baddbmm"``, ``"broadcast"``) instead of
        the automatic, device-based choice. ``None``
        (default) is the automatic choice; see
        :func:`._distance_kernels.select_kernel`. Only consulted when the
        Python implementation actually runs; see ``pair_filter`` for how
        that is decided.
    pair_filter : Literal["native", "python"] | None, optional
        Force which implementation performs the filter-and-compact step,
        instead of the automatic, CPU/availability-based choice. ``None``
        (default) is the automatic choice. ``"python"`` forces the
        pure-Python path without ever probing native availability (so it
        never pays that extension's first-use JIT-compile cost).
        ``"native"`` forces the native path and raises ``ValueError`` if
        it is not applicable -- off CPU, for positions of a dtype other
        than float32 or float64, or combined with an explicit
        ``distance_kernel``, which the native path has no way to honour
        since it computes distances itself rather than going through
        :mod:`._distance_kernels` at all. Never silently substituted for
        the other, mirroring how :func:`._distance_kernels.select_kernel`
        already refuses an inapplicable ``force`` rather than picking a
        different kernel. Private in practice (no public caller passes
        it): for tests that need both implementations' output in the same
        process, e.g. to assert they agree.
    anchor_atoms : Tensor | None, optional
        ``(n,)``, boolean. When given, only pairs with at least one anchor
        atom are returned. A periodic search passes the primary-cell atoms
        of its ghost pool, so that pairs of two images are never
        materialised. ``None`` (default) returns every pair.
    padding : _Padding | None, optional
        When given, every list is returned already padded to its
        capacity, so that the pairs are copied into their final buffers
        only once. ``None`` (default) returns the pairs unpadded.

    Returns
    -------
    list[_Pairs]
        The pairs of each threshold, in the same order as ``thresholds``.

    Raises
    ------
    ValueError
        If ``pair_filter="native"`` is combined with an explicit
        ``distance_kernel``, or is not applicable (off CPU, positions not
        float32 or float64, or the native extension failed to load).
    """
    if pair_filter == "native" and distance_kernel is not None:
        raise ValueError(
            "pair_filter='native' cannot be combined with an explicit "
            "distance_kernel: the native path computes distances itself "
            "and has no way to honour a requested formula."
        )

    native_applicable = (
        reference_positions.device.type == "cpu"
        and reference_positions.dtype in _native.SUPPORTED_DTYPES
    )
    if (
        pair_filter != "python"
        and distance_kernel is None
        and native_applicable
    ):
        native_result = _native.atom_pairs_within_thresholds_native(
            tiles.index,
            tiles.valid,
            tile_a,
            tile_b,
            reference_positions,
            thresholds,
            anchor_atoms,
            padding,
            _CAPACITY_BUCKET,
        )
        if native_result is not None:
            return [_Pairs(*pairs) for pairs in native_result]
        if pair_filter == "native":
            raise ValueError(
                "pair_filter='native' requested but the native extension "
                "failed to load; check "
                "tad_mctc.neighbor._native.is_available()."
            )
    elif pair_filter == "native":
        raise ValueError(
            "pair_filter='native' requested but is not applicable: native "
            "support is CPU-only and needs float32 or float64 positions, "
            f"got {reference_positions.device.type} and "
            f"{reference_positions.dtype}."
        )

    tile_width = tiles.tile
    pairs_per_tile_pair = tile_width * tile_width
    chunk_size = max(1, max_block // pairs_per_tile_pair)

    slot_index = torch.arange(tile_width, device=reference_positions.device)
    strict_upper_triangle = slot_index.view(-1, 1) < slot_index.view(1, -1)

    # Which way to compute a chunk's exact squared distances is a solved,
    # separately-measured question -- see `._distance_kernels`'s module
    # docstring for the measurements behind each kernel's own
    # applicability check. Picked once here, outside the chunk loop.
    kernel = _distance_kernels.select_kernel(
        reference_positions.device, force=distance_kernel
    )

    # The distances are compared in the positions' dtype, against each
    # squared threshold rounded once to that dtype, exactly as the native
    # path compares them, so that both agree on a pair right at a
    # threshold.
    thresholds_squared = [
        torch.tensor(
            threshold * threshold,
            dtype=reference_positions.dtype,
            device=reference_positions.device,
        )
        for threshold in thresholds
    ]

    collected_i: list[list[Tensor]] = [[] for _ in thresholds]
    collected_j: list[list[Tensor]] = [[] for _ in thresholds]

    total_candidates = tile_a.shape[0]
    for start in range(0, total_candidates, chunk_size):
        chunk_a = tile_a[start : start + chunk_size]
        chunk_b = tile_b[start : start + chunk_size]

        atoms_a = tiles.index[chunk_a]  # (chunk, tile_width)
        atoms_b = tiles.index[chunk_b]  # (chunk, tile_width)

        positions_a = reference_positions[atoms_a]  # (chunk, tile_width, 3)
        positions_b = reference_positions[atoms_b]  # (chunk, tile_width, 3)

        distance_squared = kernel.compute(positions_a, positions_b)

        both_real = tiles.valid[chunk_a].unsqueeze(2) & tiles.valid[
            chunk_b
        ].unsqueeze(1)

        # Within one tile paired with itself, take the strict upper
        # triangle only: it excludes self-pairs and, since the two tiles
        # are the same atom set, avoids listing both (i, j) and (j, i).
        # Between two distinct tiles the full block is correct as is,
        # since the two atom sets are disjoint and each unordered tile
        # pair is enumerated exactly once by `tile_pairs`.
        same_tile = (chunk_a == chunk_b).view(-1, 1, 1)
        keep_shape = ~same_tile | strict_upper_triangle
        candidate = both_real & keep_shape
        if anchor_atoms is not None:
            has_anchor = anchor_atoms[atoms_a].unsqueeze(2) | anchor_atoms[
                atoms_b
            ].unsqueeze(1)
            candidate = candidate & has_anchor

        for k, threshold_squared in enumerate(thresholds_squared):
            keep = candidate & (distance_squared <= threshold_squared)
            block_sel, row_sel, col_sel = keep.nonzero(as_tuple=True)
            collected_i[k].append(atoms_a[block_sel, row_sel])
            collected_j[k].append(atoms_b[block_sel, col_sel])

    results = []
    for fragments_i, fragments_j in zip(collected_i, collected_j):
        results.append(
            _join_pairs(
                fragments_i, fragments_j, padding, reference_positions.device
            )
        )

    return results


def _ghost_pairs_to_atom_pairs(
    ghost_i: Tensor,
    ghost_j: Tensor,
    owner: Tensor,
    shift: Tensor,
    is_primary: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Turn pairs of ghost-pool indices into the ``(idx_i, idx_j, shift)``
    entries of a periodic :class:`.NeighborList`.

    Every pair has at least one primary side, because the search keeps
    only those (``anchor_atoms`` in :func:`_atom_pairs_within_thresholds`).
    That side becomes ``idx_i``, so the entry's shift reaches from the
    primary ghost to the other one.

    Parameters
    ----------
    ghost_i, ghost_j : Tensor
        ``(n_pair,)``, indices into the ghost pool, in no particular
        order.
    owner : Tensor
        ``(n_ghost,)``, the atom each ghost is an image of.
    shift : Tensor
        ``(n_ghost, 3)``, the integer shift of each ghost relative to its
        atom's caller coordinates
        (:func:`_ghost_shift_in_original_coordinates`).
    is_primary : Tensor
        ``(n_ghost,)``, ``True`` for a ghost with the zero image shift.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        ``idx_i`` and ``idx_j``, both ``(n_pair,)``, and ``shift``,
        ``(n_pair, 3)``, in the dtype of the ghosts' ``shift``.
    """
    # `index_select` rather than advanced indexing: its CPU gather is
    # about twice as fast on these pair-sized index tensors.
    i_is_primary = is_primary.index_select(0, ghost_i)
    anchor = torch.where(i_is_primary, ghost_i, ghost_j)
    other = torch.where(i_is_primary, ghost_j, ghost_i)
    idx_i = owner.index_select(0, anchor)
    idx_j = owner.index_select(0, other)
    pair_shift = shift.index_select(0, other) - shift.index_select(0, anchor)
    return idx_i, idx_j, pair_shift


def _pad_to_capacity(
    fragments_i: list[Tensor],
    fragments_j: list[Tensor],
    nat: int,
    cutoff: float,
    skin: float,
    capacity: int | None,
    build_positions: Tensor,
    dd: DD,
    shift_raw: Tensor | None = None,
    lattice: Tensor | None = None,
    periodic_axes: Tensor | None = None,
) -> NeighborList:
    """
    Pad raw, exact pairs, given as fragments (see :func:`_join_pairs`), out
    to a fixed capacity, with padded slots routed to the phantom atom
    ``nat``. See :func:`_neighbor_list` for the other parameters.
    """
    pairs = _join_pairs(
        fragments_i,
        fragments_j,
        _Padding(capacity, nat),
        build_positions.device,
    )
    return _neighbor_list(
        pairs,
        cutoff,
        skin,
        build_positions,
        dd,
        shift_raw=shift_raw,
        lattice=lattice,
        periodic_axes=periodic_axes,
    )


def _neighbor_list(
    pairs: _Pairs,
    cutoff: float,
    skin: float,
    build_positions: Tensor,
    dd: DD,
    shift_raw: Tensor | None = None,
    lattice: Tensor | None = None,
    periodic_axes: Tensor | None = None,
) -> NeighborList:
    """
    A :class:`NeighborList` from pairs already padded to their capacity.

    ``shift_raw`` is the periodic image shift per found pair, shape
    ``(n_found, 3)``, or already padded with zero shifts to ``(capacity,
    3)``; ``None`` (the molecular case) gives all-zero shifts.
    ``lattice``, when given, is stored only as a detached build-time
    snapshot for :meth:`NeighborList.stale`; consumers receive it as a
    call-time argument instead (``tad_mctc.ncoord.common``).
    """
    capacity = pairs.idx_i.shape[0]
    npair = min(pairs.n_found, capacity)

    # A dropped pair is a wrong energy, not a slow one: flag it rather than
    # silently discarding the ones that do not fit. The caller must check
    # `overflow` and rebuild with more capacity before trusting anything
    # computed from this list.
    overflow = pairs.n_found > capacity

    if shift_raw is not None and npair > 0:
        # A value outside `_SHIFT_DTYPE`'s range would be truncated
        # silently by the assignment into `shift` below, turning a wrong
        # image into a wrong distance with no error -- see `_SHIFT_DTYPE`
        # for what the shift counts.
        shift_min, shift_max = (
            torch.iinfo(_SHIFT_DTYPE).min,
            torch.iinfo(_SHIFT_DTYPE).max,
        )
        # Only the kept pairs are stored, so a discarded pair must not
        # raise here and hide `overflow`. Both bounds are compared, since
        # the range is asymmetric: `shift_min` fits, `-shift_min` does not.
        # One `tolist` reads both, a single device sync.
        lowest, highest = torch.aminmax(shift_raw[:npair])
        lowest, highest = torch.stack((lowest, highest)).tolist()
        if lowest < shift_min or highest > shift_max:
            bad = lowest if lowest < shift_min else highest
            raise ValueError(
                f"Periodic image shift {bad} lies outside what "
                f"`NeighborList.shift` ({_SHIFT_DTYPE}, range "
                f"[{shift_min}, {shift_max}]) can hold. The shift counts "
                "the cells between the two atoms of a pair as written in "
                "`positions`, plus the image rings `cutoff` reaches: "
                "either atoms are written that many cells apart (wrap "
                "`positions` into the cell first) or `cutoff` is very "
                "large relative to the cell (check `lattice`)."
            )

    mask = torch.empty(capacity, dtype=torch.bool, device=dd["device"])
    mask[:npair] = True
    mask[npair:] = False

    if shift_raw is None:
        shift = _molecular_shift(capacity, pairs.idx_i.device)
    elif shift_raw.shape[0] == capacity and shift_raw.dtype == _SHIFT_DTYPE:
        # Already padded with zero shifts (or exactly full), in its final
        # dtype.
        shift = shift_raw
    else:
        shift = torch.zeros(
            capacity, 3, dtype=_SHIFT_DTYPE, device=pairs.idx_i.device
        )
        shift[:npair] = shift_raw[:npair]

    return NeighborList(
        idx_i=pairs.idx_i,
        idx_j=pairs.idx_j,
        shift=shift,
        mask=mask,
        build_positions=build_positions,
        cutoff=cutoff,
        skin=skin,
        overflow=overflow,
        lattice=lattice,
        periodic_axes=periodic_axes,
        device=dd["device"],
        dtype=dd["dtype"],
    )


@torch.no_grad()
def _tiles_of_separated_systems(
    positions: Tensor, system: Tensor, search_cutoff: float, *, tile: int
) -> Tiles:
    """
    :class:`.Tiles` of the concatenated atoms of several independent
    systems, built so that no tile pair can hold atoms of two systems.

    The systems are shifted apart (see :func:`_separate_systems`) and the
    tiles are built on the shifted cloud; the exact distance filter that
    follows must still use the original ``positions``.

    Parameters
    ----------
    positions : Tensor
        Cartesian coordinates, shape ``(nat, 3)``, of every system
        concatenated together.
    system : Tensor
        ``(nat,)``, the system index owning each atom.
    search_cutoff : float
        The largest cutoff of the search, including skin.
    tile : int
        Maximum number of atoms per tile.
    """
    separated_positions, gap = _separate_systems(
        positions, system, search_cutoff
    )
    # `_separate_systems` only ever shifts along axis 0 (see its
    # docstring), so only that axis needs a bound, and the bound is
    # the minimum inter-system gap it guarantees: the coarsening loop
    # must never widen axis-0 bins past that gap, or a bin could
    # straddle it and mix atoms from two different systems. The other
    # two axes are unconstrained (`inf`), exactly like the
    # non-batched path.
    max_bin_width = torch.full(
        (3,),
        float("inf"),
        dtype=separated_positions.dtype,
        device=positions.device,
    )
    max_bin_width[0] = gap
    return Tiles(separated_positions, tile=tile, max_bin_width=max_bin_width)


def _tiles_holding_primary(tiles: Tiles, is_primary: Tensor) -> Tensor:
    """
    Which tiles of a ghost pool hold a primary-cell atom.

    A periodic list keeps only pairs with a primary side (see
    :func:`_ghost_pairs_to_atom_pairs`), and such a pair can only come from
    a tile pair with a tile of this kind. Most tile pairs of a ghost pool
    have none, so searching only those that have one (``anchors`` of
    :func:`.tile_pairs`) spares most of the work and changes no result.

    Parameters
    ----------
    tiles : Tiles
        Tiles built on the ghost pool.
    is_primary : Tensor
        ``(n_ghost,)``, ``True`` for a ghost that is the un-translated
        copy of its atom.

    Returns
    -------
    Tensor
        ``(ntile,)``, boolean.
    """
    return (is_primary[tiles.index] & tiles.valid).any(-1)


def _ghost_shift_in_original_coordinates(
    ghost_shift: Tensor, owner: Tensor, cell_shift: Tensor
) -> Tensor:
    """
    Back out of the fold into the central cell, once per ghost: a ghost
    sits at ``wrapped[owner] + ghost_shift @ lattice``, and
    ``wrapped == positions + cell_shift @ lattice``, so relative to its
    atom's caller coordinates it is shifted by
    ``ghost_shift + cell_shift[owner]``. The shift of a pair is the
    difference of its two ghosts' shifts (:func:`_ghost_pairs_to_atom_pairs`).

    The result is ``int16`` when every entry lies within half of
    ``int16``'s range, so that the difference of two cannot overflow.
    The narrow dtype makes the per-pair gathers and the range check in
    :func:`_neighbor_list` several times cheaper than in ``int64``. An
    atom written thousands of cells away keeps the input's dtype.

    Parameters
    ----------
    ghost_shift : Tensor
        ``(n_ghost, 3)``, the integer image shift of each ghost, between
        wrapped atoms.
    owner : Tensor
        ``(n_ghost,)``, the atom each ghost is an image of.
    cell_shift : Tensor
        ``(n_atoms, 3)``, the integer cell offset the fold applied to each
        atom (:func:`.wrap_to_central_cell`).

    Returns
    -------
    Tensor
        ``(n_ghost, 3)``, integer shifts.
    """
    shift = ghost_shift + cell_shift[owner]
    half_of_range = torch.iinfo(_SHIFT_DTYPE).max // 2
    if shift.numel() == 0 or int(shift.abs().max()) <= half_of_range:
        return shift.to(_SHIFT_DTYPE)
    return shift


def _split_cells_by_pair_budget(
    numbers: Tensor, lattices: Tensor, cutoff: float
) -> list[tuple[int, int]]:
    """
    Split a batch of cells into consecutive chunks of systems that are
    searched one after the other, each with an estimated pair count of at
    most :data:`_PAIRS_PER_CELL_SEARCH` (a chunk is never smaller than one
    system).

    A cell with ``n`` atoms and volume ``V`` has about
    ``n * (n / V) * (4 / 3) * pi * cutoff**3 / 2`` pairs: every atom sees
    the atoms in a cutoff sphere at the cell's density, and each pair is
    listed once. A slab or wire, whose lattice has a short placeholder
    vector, gets an over-estimate, which only makes its chunks smaller.

    Parameters
    ----------
    numbers : Tensor
        ``(n_systems, nat)``, ``0`` marking padding.
    lattices : Tensor
        ``(n_systems, 3, 3)``, lattice vectors as rows.
    cutoff : float
        Search radius, including skin.

    Returns
    -------
    list[tuple[int, int]]
        ``(first, stop)`` system index ranges covering the batch.
    """
    n_real = real_atoms(numbers).sum(-1).to(lattices.dtype)
    volume = torch.linalg.det(lattices).abs()
    sphere = 4.0 / 3.0 * math.pi * cutoff**3
    estimated_pairs = (n_real**2 * sphere / volume / 2).tolist()

    chunks = []
    first, chunk_pairs = 0, 0.0
    for system, pairs in enumerate(estimated_pairs):
        if system > first and chunk_pairs + pairs > _PAIRS_PER_CELL_SEARCH:
            chunks.append((first, system))
            first, chunk_pairs = system, 0.0
        chunk_pairs += pairs
    chunks.append((first, len(estimated_pairs)))
    return chunks


def _ghost_pool(
    is_real: Tensor,
    wrapped_positions: Tensor,
    lattices: Tensor,
    periodics: Tensor,
    cutoff: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    The ghost pool of one or more cells: every real atom at every forward
    image shift (see :func:`_is_forward`) within reach of ``cutoff``.

    A bond between atom ``i`` and image ``n`` of atom ``j`` is the same
    bond as the one between ``j`` and image ``-n`` of ``i``. Exactly one of
    ``n`` and ``-n`` is forward, so a search that pairs every primary atom
    with this pool finds each bond once, and needs no deduplication pass.
    A self-image bond (``i == j``) is then one entry that stands for both
    neighbours, ``+n`` and ``-n`` (see :class:`.NeighborList`).

    The shift table is shared, sized for the most demanding lattice
    (:func:`.build_periodic_shifts`), but every cell only gets the
    shifts within its own image rings. Without that, one small cell in a
    batch of large ones would give every cell its many rings. An open
    axis has no rings, so a cell with a short placeholder vector there
    gets no images along it.

    Parameters
    ----------
    is_real : Tensor
        ``(n_systems, nat)``, ``False`` marking padding atoms.
    wrapped_positions : Tensor
        ``(n_systems, nat, 3)``, folded into each system's central cell.
    lattices : Tensor
        ``(n_systems, 3, 3)``, lattice vectors as rows.
    periodics : Tensor
        ``(n_systems, 3)``, the periodic axes of each system.
    cutoff : float
        Sphere radius to cover.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor, Tensor]
        ``ghost_positions`` ``(n_ghost, 3)``; ``owner`` ``(n_ghost,)``,
        the flattened atom index ``b * nat + i`` each ghost is an image
        of; ``shift`` ``(n_ghost, 3)``, its integer lattice translation;
        and ``system`` ``(n_ghost,)``, the system it belongs to.
    """
    nat = is_real.shape[-1]

    rings = _image_rings(lattices, periodics, cutoff)  # (n_systems, 3)
    shifts = _integer_box(rings.amax(0))
    shifts = shifts[_is_forward(shifts)]
    within_rings = (shifts.abs()[None] <= rings[:, None, :]).all(-1)

    # Only the ghosts that exist are built, from their indices, in the
    # order of a row-major `(system, atom, shift)` grid. The grid itself
    # is never formed: the shared table can hold far more shifts than most
    # cells use, and `nat` counts padding, so it could far outgrow the
    # ghosts. Each real atom instead takes the run of its own system's
    # shifts from `within_rings`.
    atom_system, real_atom = is_real.nonzero(as_tuple=True)
    _, own_shift = within_rings.nonzero(as_tuple=True)
    shifts_per_system = within_rings.sum(-1)
    first_shift = torch.cumsum(shifts_per_system, 0) - shifts_per_system

    ghost_atom, position_in_run = _ragged_runs(shifts_per_system[atom_system])
    system = atom_system[ghost_atom]
    atom = real_atom[ghost_atom]
    shift_index = own_shift[first_shift[system] + position_in_run]

    translations = shifts.to(wrapped_positions.dtype) @ lattices
    return (
        wrapped_positions[system, atom] + translations[system, shift_index],
        system * nat + atom,
        shifts[shift_index],
        system,
    )


def _search_molecules(
    positions: Tensor,
    thresholds: tuple[float, ...],
    *,
    tile: int,
    distance_kernel: DistanceKernelName | None,
    stage: _StageHook,
    system: Tensor | None = None,
    padding: _Padding | None = None,
) -> list[_Pairs]:
    """
    The atom pairs within each threshold, for one molecule or several
    independent ones concatenated in ``positions``.

    Parameters
    ----------
    positions : Tensor
        ``(nat, 3)``, Cartesian coordinates.
    thresholds : tuple[float, ...]
        One cutoff (already including any skin) per requested list.
    tile, distance_kernel
        See :func:`build_neighborlists`.
    stage : _StageHook
        Wraps each step of the search, see :func:`_build_neighborlists`.
    system : Tensor | None, optional
        ``(nat,)``, the system owning each atom. ``None`` (default) treats
        ``positions`` as one system.
    padding : _Padding | None, optional
        See :func:`_atom_pairs_within_thresholds`.

    Returns
    -------
    list[_Pairs]
        The pairs of each threshold, indexing ``positions``.
    """
    search_cutoff = max(thresholds)
    with stage("tiles"):
        if system is None:
            tiles = Tiles(positions, tile=tile)
        else:
            tiles = _tiles_of_separated_systems(
                positions, system, search_cutoff, tile=tile
            )

    with stage("tile pairs"):
        tile_a, tile_b = tile_pairs(tiles, search_cutoff)

    with stage("pair filter"):
        return _atom_pairs_within_thresholds(
            tiles,
            tile_a,
            tile_b,
            positions,
            thresholds,
            distance_kernel=distance_kernel,
            padding=padding,
        )


def _search_cells(
    is_real: Tensor,
    positions: Tensor,
    lattices: Tensor,
    periodics: Tensor,
    thresholds: tuple[float, ...],
    *,
    tile: int,
    distance_kernel: DistanceKernelName | None,
    stage: _StageHook,
    padding: _Padding | None = None,
) -> list[tuple[_Pairs, Tensor]]:
    """
    The pairs within each threshold and their image shifts, for one cell
    or several independent ones, from one ghost pool and one tile search.

    Parameters
    ----------
    is_real : Tensor
        ``(n_systems, nat)``, ``False`` marking padding atoms.
    positions : Tensor
        ``(n_systems, nat, 3)``, Cartesian coordinates, not necessarily
        inside the cell.
    lattices : Tensor
        ``(n_systems, 3, 3)``, lattice vectors as rows.
    periodics : Tensor
        ``(n_systems, 3)``, the periodic axes of each system.
    thresholds : tuple[float, ...]
        One cutoff (already including any skin) per requested list.
    tile, distance_kernel
        See :func:`build_neighborlists`.
    stage : _StageHook
        Wraps each step of the search, see :func:`_build_neighborlists`.
    padding : _Padding | None, optional
        When given, the pairs and shifts of every list are returned
        already padded to its capacity, a padded slot holding
        ``padding.value`` twice and a zero shift. ``None`` (default)
        returns them unpadded.

    Returns
    -------
    list[tuple[_Pairs, Tensor]]
        Per threshold, the pairs, with atoms numbered ``b * nat + i`` like
        :class:`.NeighborList`, and the ``(n, 3)`` image shift of each.
    """
    n_systems, nat = is_real.shape
    search_cutoff = max(thresholds)

    # The ghost pool reaches a fixed number of image rings out from the
    # cell, sized from the lattice and the cutoff alone, so it only covers
    # pairs between atoms written within that many cells of each other.
    # Folding into the central cell first makes that unconditional, as
    # s-dftd3 does on every structure entering its API. The fold is added
    # back into each ghost's shift below, so the list still describes the
    # caller's own, unwrapped positions.
    with stage("wrap to cell"):
        wrapped_positions, cell_shift = wrap_to_central_cell(
            positions, lattices, periodics.unsqueeze(-2)
        )
        cell_shift = cell_shift.reshape(n_systems * nat, 3)

    with stage("ghost pool"):
        ghost_positions, owner, ghost_shift, system = _ghost_pool(
            is_real, wrapped_positions, lattices, periodics, search_cutoff
        )
        is_primary = (ghost_shift == 0).all(-1)

    with stage("tiles"):
        if n_systems == 1:
            tiles = Tiles(ghost_positions, tile=tile)
        else:
            tiles = _tiles_of_separated_systems(
                ghost_positions, system, search_cutoff, tile=tile
            )

    with stage("tile pairs"):
        tile_a, tile_b = tile_pairs(
            tiles,
            search_cutoff,
            anchors=_tiles_holding_primary(tiles, is_primary),
        )

    # The ghost pairs are padded with a phantom ghost one past the pool.
    # Mapped to the phantom atom with a zero shift below, its padded slots
    # come out as those of the final list, which is so written only once.
    n_ghost = ghost_positions.shape[0]
    ghost_padding = (
        None if padding is None else _Padding(padding.capacity, n_ghost)
    )
    with stage("pair filter"):
        per_threshold_ghost_pairs = _atom_pairs_within_thresholds(
            tiles,
            tile_a,
            tile_b,
            ghost_positions,
            thresholds,
            distance_kernel=distance_kernel,
            anchor_atoms=is_primary,
            padding=ghost_padding,
        )

    with stage("ghost shifts"):
        shift_from_original = _ghost_shift_in_original_coordinates(
            ghost_shift, owner, cell_shift
        )
        if padding is not None:
            owner = torch.cat([owner, owner.new_full((1,), padding.value)])
            shift_from_original = torch.cat(
                [shift_from_original, shift_from_original.new_zeros(1, 3)]
            )
            is_primary = torch.cat([is_primary, is_primary.new_ones(1)])

    with stage("ghost pairs to atoms"):
        per_threshold_pairs = []
        for ghost_pairs in per_threshold_ghost_pairs:
            idx_i, idx_j, shift = _ghost_pairs_to_atom_pairs(
                ghost_pairs.idx_i,
                ghost_pairs.idx_j,
                owner,
                shift_from_original,
                is_primary,
            )
            pairs = _Pairs(idx_i, idx_j, ghost_pairs.n_found)
            per_threshold_pairs.append((pairs, shift))
        return per_threshold_pairs


def _check_search_arguments(
    cutoffs: tuple[float, ...], skin: float, tile: int
) -> None:
    """Reject a negative cutoff or skin, and a tile of fewer than one
    atom."""
    if skin < 0:
        raise ValueError(f"`skin` must be non-negative, got {skin}.")
    for cutoff in cutoffs:
        if cutoff < 0:
            raise ValueError(f"`cutoff` must be non-negative, got {cutoff}.")
    if tile < 1:
        raise ValueError(f"`tile` must be at least 1, got {tile}.")


def _check_batched_search_radius(search_cutoff: float) -> None:
    """
    A batched search keeps systems apart purely geometrically, and the gap
    it can guarantee is proportional to the search radius (see
    :func:`_separate_systems`). At a zero radius there is no gap left, and
    two coincident atoms in *different* systems would come back as a
    pair, so the request is rejected.
    """
    if search_cutoff == 0.0:
        raise ValueError(
            "A batched neighbour search needs a positive search "
            "radius to keep systems apart, but `max(cutoffs) + skin` "
            "is zero. Pass a positive `cutoff` (or `skin`), or search "
            "each system separately."
        )


def _build_single_neighborlists(
    structure: Structure,
    cutoffs: tuple[float, ...],
    *,
    tile: int,
    skin: float,
    capacity: int | None,
    distance_kernel: DistanceKernelName | None,
    stage: _StageHook,
) -> tuple[NeighborList, ...]:
    """
    :func:`build_neighborlists` for a single-system ``structure``, see
    :func:`_build_neighborlists` for the arguments.
    """
    positions = structure.positions
    nat = positions.shape[0]
    thresholds = tuple(cutoff + skin for cutoff in cutoffs)
    dd: DD = {"device": positions.device, "dtype": positions.dtype}

    # Padding atoms (`numbers == 0`) must not get pairs, as in a batch.
    is_real = real_atoms(structure.numbers)

    if structure.lattice is None:
        if bool(is_real.all()):
            # The search pads each list itself, so its pairs are copied
            # into their final buffers only once.
            per_threshold_pairs = _search_molecules(
                positions,
                thresholds,
                tile=tile,
                distance_kernel=distance_kernel,
                stage=stage,
                padding=_Padding(capacity, nat),
            )
            with stage("finalize"):
                return tuple(
                    _neighbor_list(pairs, cutoff, skin, positions, dd)
                    for cutoff, pairs in zip(cutoffs, per_threshold_pairs)
                )

        # Search the real atoms only and renumber their pairs back into
        # `positions`. The phantom index `nat` of a padded slot has no
        # entry in `real_atom_index`, so the lists are padded only after
        # renumbering.
        real_atom_index = is_real.nonzero().squeeze(-1)
        per_threshold_pairs = _search_molecules(
            positions[is_real],
            thresholds,
            tile=tile,
            distance_kernel=distance_kernel,
            stage=stage,
        )
        with stage("pad to capacity"):
            return tuple(
                _pad_to_capacity(
                    [real_atom_index[pairs.idx_i]],
                    [real_atom_index[pairs.idx_j]],
                    nat,
                    cutoff,
                    skin,
                    capacity,
                    positions,
                    dd,
                )
                for cutoff, pairs in zip(cutoffs, per_threshold_pairs)
            )

    # `Structure` fills in a mask whenever it has a lattice.
    assert structure.periodic is not None
    # A singular cell would fail its inversion in `wrap_to_central_cell`
    # with an opaque linear-algebra error.
    _validate_lattice_periodic(structure.lattice, structure.periodic)

    # The cell search takes a batch of cells, here a batch of one.
    per_threshold_cell_pairs = _search_cells(
        is_real.unsqueeze(0),
        positions.unsqueeze(0),
        structure.lattice.unsqueeze(0),
        structure.periodic.unsqueeze(0),
        thresholds,
        tile=tile,
        distance_kernel=distance_kernel,
        stage=stage,
        padding=_Padding(capacity, nat),
    )
    with stage("finalize"):
        return tuple(
            _neighbor_list(
                pairs,
                cutoff,
                skin,
                positions,
                dd,
                shift_raw=shift,
                lattice=structure.lattice,
                periodic_axes=structure.periodic,
            )
            for cutoff, (pairs, shift) in zip(cutoffs, per_threshold_cell_pairs)
        )


def _build_batched_neighborlists(
    structure: Structure,
    cutoffs: tuple[float, ...],
    *,
    tile: int,
    skin: float,
    capacity: int | None,
    distance_kernel: DistanceKernelName | None,
    stage: _StageHook,
) -> tuple[NeighborList, ...]:
    """
    :func:`build_neighborlists` for a batched ``structure``: search its
    real atoms, number every pair in the padded, flattened layout
    ``b * nat + i``, and pad the result to one capacity. See
    :func:`_build_neighborlists` for the arguments.
    """
    nat = structure.numbers.shape[-1]
    numbers = structure.numbers.reshape(-1, nat)  # (B, nat)
    positions = structure.positions.reshape(-1, nat, 3)
    n_systems = numbers.shape[0]
    is_real = real_atoms(numbers)

    thresholds = tuple(cutoff + skin for cutoff in cutoffs)
    _check_batched_search_radius(max(thresholds))

    # The indices stay in fragments, which `_pad_to_capacity` joins with
    # the padding in a single copy.
    per_threshold_pairs: list[
        tuple[list[Tensor], list[Tensor], Tensor | None]
    ] = []
    if structure.lattice is None:
        # One search over the real atoms of every system, kept apart by
        # their system index, then renumbered into the padded layout.
        flat_index = torch.arange(n_systems * nat, device=positions.device)
        real_atom_index = flat_index.reshape(n_systems, nat)[is_real]
        system = torch.arange(n_systems, device=positions.device)
        system_of_atom = system.unsqueeze(-1).expand(n_systems, nat)

        for pairs in _search_molecules(
            positions[is_real],
            thresholds,
            tile=tile,
            distance_kernel=distance_kernel,
            stage=stage,
            system=system_of_atom[is_real],
        ):
            per_threshold_pairs.append(
                (
                    [real_atom_index[pairs.idx_i]],
                    [real_atom_index[pairs.idx_j]],
                    None,
                )
            )
    else:
        assert structure.periodic is not None
        batch_shape = structure.numbers.shape[:-1]
        lattices = structure.lattice.expand(*batch_shape, 3, 3)
        lattices = lattices.reshape(n_systems, 3, 3)
        periodics = structure.periodic.expand(*batch_shape, 3)
        periodics = periodics.reshape(n_systems, 3)
        # Before the split, whose pair estimate divides by the volume, and
        # the cell search, whose inversion fails on a singular cell.
        _validate_lattice_periodic(lattices, periodics)

        # Cells are searched in chunks, each with one shared ghost pool, to
        # bound the peak memory (see `_PAIRS_PER_CELL_SEARCH`).
        per_threshold_chunks: list[list[tuple[Tensor, Tensor, Tensor]]] = [
            [] for _ in thresholds
        ]
        for first, stop in _split_cells_by_pair_budget(
            numbers, lattices, max(thresholds)
        ):
            searched = _search_cells(
                is_real[first:stop],
                positions[first:stop],
                lattices[first:stop],
                periodics[first:stop],
                thresholds,
                tile=tile,
                distance_kernel=distance_kernel,
                stage=stage,
            )
            # The chunk numbers its atoms from zero.
            offset = first * nat
            for chunks, (pairs, shift) in zip(per_threshold_chunks, searched):
                chunks.append(
                    (pairs.idx_i + offset, pairs.idx_j + offset, shift)
                )

        for chunks in per_threshold_chunks:
            fragments_i, fragments_j, shifts = map(list, zip(*chunks))
            # Only the fragment lists hold the pairs, so that joining them
            # frees each one.
            chunks.clear()
            per_threshold_pairs.append(
                (fragments_i, fragments_j, torch.cat(shifts))
            )

    dd: DD = {"device": positions.device, "dtype": positions.dtype}
    with stage("pad to capacity"):
        return tuple(
            _pad_to_capacity(
                fragments_i,
                fragments_j,
                n_systems * nat,
                cutoff,
                skin,
                capacity,
                structure.positions,
                dd,
                shift_raw=shift,
                lattice=structure.lattice,
                periodic_axes=structure.periodic,
            )
            for cutoff, (fragments_i, fragments_j, shift) in zip(
                cutoffs, per_threshold_pairs
            )
        )


try:
    from torch._C._functorch import (  # pyright: ignore[reportMissingImports]
        peek_interpreter_stack,
        pop_dynamic_layer_stack,
        push_dynamic_layer_stack,
    )

    _CAN_POP_LAYERS = True
except ImportError:  # pragma: no cover
    _CAN_POP_LAYERS = False


@contextlib.contextmanager
def _pop_transform_layers() -> Generator[None]:
    """
    Run the body with every ``torch.func`` layer set aside.

    ``torch._functorch.pyfunctorch.temporarily_pop_interpreter_stack``
    pops one layer only, so under ``jacrev(jacrev(f))`` the tensors made in
    the body would still be wrapped by the outer one.
    """
    if not _CAN_POP_LAYERS:  # pragma: no cover
        raise RuntimeError(
            "Building a neighbour list inside `jacrev`/`grad` needs "
            "`torch._C._functorch.pop_dynamic_layer_stack`, which this "
            f"PyTorch ({torch.__version__}) does not provide. Build the "
            "list outside the transform."
        )

    popped = []
    try:
        while peek_interpreter_stack() is not None:
            popped.append(pop_dynamic_layer_stack())
        yield
    finally:
        for layer in reversed(popped):
            push_dynamic_layer_stack(layer)


@contextlib.contextmanager
def _outside_transforms(structure: Structure) -> Generator[Structure]:
    """
    Run a build with the ``jacrev``/``grad`` layers of ``torch.func`` set
    aside, on a structure whose tensors are unwrapped from them.

    A list is index data and carries no gradient. Under ``jacrev`` the
    tensors a build creates (tile indices, masks) would be wrapped like its
    inputs, and the native search cannot read a wrapped tensor, so the
    build runs as plain eager code on the values underneath. A ``vmap``
    layer cannot be set aside: the size of the list depends on the data,
    which a batched tensor does not have.

    Yields
    ------
    Structure
        ``structure`` without the grad-tracking wrappers of its tensors.

    Raises
    ------
    RuntimeError
        A tensor of ``structure`` is batched by ``torch.func.vmap``.
    """
    fields = ("numbers", "positions", "lattice", "periodic")
    tensors = {n: getattr(structure, n) for n in fields}

    if any(t is not None and is_vmapped(t) for t in tensors.values()):
        raise RuntimeError(
            "A neighbour list cannot be built inside `torch.func.vmap`: its "
            "size depends on the data. Build the list(s) outside `vmap` (at "
            "a common `capacity` for lists that are stacked) and pass them "
            "in."
        )

    unwrapped = {
        n: unwrap_gradtracking(t) for n, t in tensors.items() if t is not None
    }
    if all(unwrapped[n] is tensors[n] for n in unwrapped):
        yield structure
        return

    with _pop_transform_layers():
        # The list keeps `periodic` as its `periodic_axes`; a copy, made
        # with the layers aside so that it is not wrapped, so that no tensor
        # that sat under a wrapper outlives the transform inside the list.
        if unwrapped.get("periodic") is not None:
            unwrapped["periodic"] = unwrapped["periodic"].clone()
        yield structure.replace(**unwrapped)


@torch.compiler.disable
@torch.no_grad()
def _build_neighborlists(
    structure: Structure,
    cutoffs: tuple[float, ...],
    *,
    tile: int = 32,
    skin: float = 0.0,
    capacity: int | None = None,
    distance_kernel: DistanceKernelName | None = None,
    stage: _StageHook = _no_stage_hook,
) -> tuple[NeighborList, ...]:
    """
    :func:`build_neighborlists` with a ``stage`` hook around each step.

    ``stage(label)`` is entered around every step of the build, labelled
    ``"tiles"``, ``"tile pairs"``, ``"pair filter"`` and so on, so that a
    profiling tool (the ``tad_mctc --timing`` command line) can time the
    steps of the one real build instead of re-assembling it. The default
    does nothing.

    Parameters
    ----------
    structure, cutoffs, tile, skin, capacity, distance_kernel
        See :func:`build_neighborlists`.
    stage : _StageHook, optional
        Context manager factory entered around each step of the build.

    Returns
    -------
    tuple[NeighborList, ...]
        One list per entry of ``cutoffs``, in the same order.
    """
    _check_search_arguments(cutoffs, skin, tile)

    with _outside_transforms(structure) as plain:
        build = (
            _build_batched_neighborlists
            if plain.numbers.ndim > 1
            else _build_single_neighborlists
        )
        return build(
            plain,
            cutoffs,
            tile=tile,
            skin=skin,
            capacity=capacity,
            distance_kernel=distance_kernel,
            stage=stage,
        )


def build_neighborlists(
    structure: Structure,
    cutoffs: tuple[float, ...],
    *,
    tile: int = 32,
    skin: float = 0.0,
    capacity: int | None = None,
    distance_kernel: DistanceKernelName | None = None,
) -> tuple[NeighborList, ...]:
    """
    Build one :class:`.NeighborList` per cutoff in ``cutoffs``, sharing a
    single :class:`.Tiles` build and a single tile-pair search at the
    largest cutoff.

    This is data-dependent (`argsort`, `nonzero`, boolean-mask indexing,
    a Python-level padding decision) and runs under ``torch.no_grad()``:
    it belongs to list *construction*, never to the differentiated
    consumption path.

    D3 needs the coordination number at one cutoff and the two-body sum
    at a larger one; the pair count goes as cutoff cubed, so the smaller
    list holds only a fraction of the larger one's pairs. Carrying both
    cutoffs inside one list would size the smaller one at the larger
    cutoff and throw that saving away, so each returned list is
    compacted, and its capacity bucketed, to its own cutoff, even though
    the tile traversal that finds the candidates is shared.

    When ``structure`` has a lattice, the search pairs every atom of the
    cell with a ghost pool of its periodic images (:func:`._ghost_pool`)
    instead of searching ``positions`` directly, and each resulting pair
    carries the integer lattice ``shift`` that reaches it. Positions need
    not lie inside the cell: they are folded into it first
    (:func:`.wrap_to_central_cell`, as s-dftd3 does on every structure
    entering its API) and the fold is carried back out through each
    pair's ``shift``, so an unwrapped trajectory gives the same list as
    its wrapped equivalent. The list is built periodic along exactly
    ``structure.periodic``.

    A batched ``structure`` (``numbers`` of shape ``(..., nat)``) gives
    one list over its padded atoms, flattened: atom ``i`` of system ``b``
    is index ``b * nat + i`` (see :class:`.NeighborList`). A batch of
    molecules is searched at once, and so is a batch of cells, each with
    its own lattice and ``periodic`` mask.

    Parameters
    ----------
    structure : Structure
        One system, ``numbers`` of shape ``(nat,)``, or a batch,
        ``(..., nat)``, with ``0`` marking padding atoms in either. Its
        ``positions``, ``lattice`` and ``periodic`` are read.
    cutoffs : tuple[float, ...]
        Real-space cutoffs, one :class:`.NeighborList` returned per
        entry, in the same order.
    tile : int, optional
        Maximum atoms per tile, passed to :class:`.Tiles`.
    skin : float, optional
        Extra radius searched beyond each cutoff.
    capacity : int | None, optional
        Fixed capacity for every returned list. ``None`` (default) rounds
        each list's own pair count up to a multiple of 4096 instead.
    distance_kernel : DistanceKernelName | None, optional
        Force the exact-distance filter to use one named kernel
        (``"triton"``, ``"baddbmm"`` or ``"broadcast"``) instead of the
        automatic, device/dtype/tile-based choice -- for comparing
        kernels directly, e.g. in a benchmark. ``None`` (default) is the
        automatic choice; raises ``ValueError`` if the named kernel is
        not applicable on this device/dtype rather than silently using
        a different one. See :func:`._distance_kernels.select_kernel`.
    Returns
    -------
    tuple[NeighborList, ...]
        One list per entry of ``cutoffs``, in the same order.

    Raises
    ------
    ValueError
        A cutoff or ``skin`` is negative; ``tile < 1``; or a batched
        ``structure`` is searched with ``max(cutoffs) + skin == 0``.
    RuntimeError
        ``structure`` is batched by ``torch.func.vmap``. Inside ``jacrev``,
        ``jacfwd``, ``hessian``, ``jvp`` and ``grad`` the build works: it
        reads the values of the tensors, and the list holds only integer
        and boolean data, so it is a constant to the transform. (``jacfwd``
        batches only the tangents, not the primal positions the build
        reads.) The build is a graph break under ``torch.compile``, and
        PyTorch returns an all-zero gradient for ``torch.compile(jacrev(f))``
        when ``f`` has any graph break; use ``grad``, or build outside.
    """
    return _build_neighborlists(
        structure,
        cutoffs,
        tile=tile,
        skin=skin,
        capacity=capacity,
        distance_kernel=distance_kernel,
    )


def build_neighborlist(
    structure: Structure,
    cutoff: float,
    *,
    tile: int = 32,
    skin: float = 0.0,
    capacity: int | None = None,
    distance_kernel: DistanceKernelName | None = None,
) -> NeighborList:
    """
    Build a :class:`.NeighborList` for a single cutoff.

    One-cutoff form of :func:`build_neighborlists`; see there for the
    parameters and for why list construction runs under
    ``torch.no_grad()``.

    Parameters
    ----------
    structure : Structure
        One system or a batch, see :func:`build_neighborlists`. Its
        ``positions``, ``lattice`` and ``periodic`` are read.
    cutoff : float
        Real-space cutoff.
    tile : int, optional
        Maximum atoms per tile, passed to :class:`.Tiles`.
    skin : float, optional
        Extra radius searched beyond ``cutoff``.
    capacity : int | None, optional
        Fixed capacity. ``None`` (default) rounds the pair count up to a
        multiple of 4096 instead.
    distance_kernel : DistanceKernelName | None, optional
        Force the exact-distance filter to use one named kernel instead
        of the automatic choice; see :func:`build_neighborlists`.
    Returns
    -------
    NeighborList
        The padded neighbour list.
    """
    (neighbor_list,) = build_neighborlists(
        structure,
        (cutoff,),
        tile=tile,
        skin=skin,
        capacity=capacity,
        distance_kernel=distance_kernel,
    )
    return neighbor_list
