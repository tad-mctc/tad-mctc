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
Coordination number: Common helpers
===================================

A :class:`.CNModel` is a frozen value object holding everything one
coordination-number variant needs (the counting function and its
parameters, cutoff, radii and electronegativity tables, pair weight and
CN cap), so that ``cn_d3``, ``cn_d4``, ... (see ``ncoord/d3.py`` etc.) are
instances rather than separate hard-coded functions. A different variant
is obtained with :meth:`.CNModel.replace`, never a subclass.

One call, :meth:`.CNModel.__call__`, evaluates a molecule or a cell. Two
kernels sit behind it, chosen by how pairs are enumerated: the dense sum
over all pairs and their periodic images (:func:`_cn_dense`) and the sum
over a padded, pre-built neighbour list (:func:`_cn_sparse`). Each computes
its pair distances and sums the result; the per-pair count and its
masking are shared (:func:`_masked_pair_counts`).
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Literal, NamedTuple, Protocol

import torch
import torch.utils.checkpoint

from ..autograd import is_functorch_tensor
from ..batch import real_pairs
from ..data import en as eneg
from ..data import radii
from ..data.table import resolve_table
from ..io.structure import Structure
from ..neighbor import (
    gather_index,
    pair_distance_squared_from_columns,
    position_columns,
    split_lattice,
)
from ..neighbor.images import (
    PeriodicShifts,
    build_periodic_shifts,
    wrap_to_central_cell,
)
from ..neighbor.list import NeighborList
from ..tools import is_compiling
from ..tree import Node, child, context
from ..typing import CountingFunction, PairWeightFunction, TableFunction, Tensor
from . import defaults

__all__ = [
    "CNFunc",
    "CNModel",
    "NeighborListMode",
    "cut_coordination_number",
    "sum_over_neighborlist",
]

NeighborListMode = Literal["graph", "recompute"]
"""How :meth:`CNModel.__call__` evaluates a :class:`.NeighborList`, see its
``mode`` argument."""

# The sparse kernel evaluates its neighbour list in chunks of this many
# pairs (see `_chunk_size`). Unchunked, it materialises a dozen pair-sized
# temporaries. On CPU, a buffer beyond glibc's 32 MiB mmap threshold is
# fresh memory on every call, and its first-touch page faults cost more
# than the arithmetic. A chunk instead stays in cache and reuses the same
# memory.
#
# The devices need different sizes because a chunk costs them different
# things. On a GPU every chunk pays a dozen kernel launches, and must be
# large enough to fill the device: 131_072 pairs is about 2x slower there
# than the optimum near 524_288. On CPU a chunk should stay in cache:
# 1_048_576 is 1.2-2x slower than 131_072. No single size comes within
# 1.3x of both optima; 262_144, the best compromise, is up to 1.15x off on
# the GPU but up to 1.46x on eight CPU threads. On the GPU the curve is
# flat towards larger chunks (1_048_576 is 1.07-1.2x off, a single chunk
# 1.4x), so the GPU value errs on the large side, where a larger GPU, which
# needs more work per launch to fill it, finds its optimum.
#
# Measured on a Ryzen 7 5700X and an RTX 4070 with `cn_eeq` at 25 Bohr,
# forward and backward, on a sparse (glu_ala, 110 pairs/atom, 12M pairs)
# and a dense (benchRIB cutout, 434 pairs/atom, 22M pairs) system, sweeping
# 65_536 to 2_097_152 pairs. The figures above are float64. Density does
# not move either optimum, and on CPU neither does the thread count (1 or
# 8). float32 doubles both optima, as if they were fixed in bytes, but
# these values stay within 1.2x of every optimum measured, in either
# dtype. Re-measure on other hardware before changing them.
_CHUNK_SIZE_CPU = 131_072
_CHUNK_SIZE_GPU = 1_048_576


def _chunk_size(like: Tensor) -> int:
    """
    Pair entries per chunk of the pair loop on ``like``'s device, see
    ``_CHUNK_SIZE_CPU``. Read at call time, so tests can shrink the
    constants to exercise several chunks on a small system.
    """
    if like.device.type == "cpu":
        return _CHUNK_SIZE_CPU
    return _CHUNK_SIZE_GPU


def _species(numbers: Tensor) -> Tensor:
    """
    Map atomic numbers to per-element table indices, treating padding as
    atomic number 1.

    Padding atoms (``numbers == 0``) look up element 1, not 0: ``rcov[0]
    == 0`` makes ``r0**norm_exp`` non-differentiable, so table Jacobians
    would turn non-finite even though the padded pairs are masked out by
    ``real_pairs`` regardless of which species they resolve to here.
    """
    return torch.where(numbers == 0, torch.ones_like(numbers), numbers)


def _validate_mode(
    mode: NeighborListMode, pairs: PeriodicShifts | NeighborList | None
) -> None:
    """
    The checks :meth:`CNModel.__call__` adds on top of ``check_compatible``:
    ``mode`` is a known value, and anything but ``"graph"`` has a
    :class:`.NeighborList` to apply to. It reads plain Python values, so it
    is safe under ``vmap`` and ``torch.compile``.
    """
    if mode not in ("graph", "recompute"):
        raise ValueError(
            "`mode` must be either 'graph' or 'recompute', got "
            f"'{mode}'. 'recompute' checkpoints the pair loop and is not "
            "compatible with `vmap`, which raises `RuntimeError: You "
            "tried to vmap over _NoopSaveInputs`."
        )
    if mode == "recompute" and not isinstance(pairs, NeighborList):
        raise ValueError(
            "`mode='recompute'` chunks the pair loop of a neighbour list; "
            "pass a `NeighborList` as `pairs`."
        )


class CNFunc(Protocol):
    """
    Type annotation for a coordination-number function: the call signature
    every :class:`CNModel` preset (``cn_d3``, ``cn_d4``, ...) satisfies.
    """

    def __call__(
        self,
        structure: Structure,
        pairs: PeriodicShifts | NeighborList | None = None,
        *,
        mode: NeighborListMode = "graph",
    ) -> Tensor:
        """
        Calculate the coordination number of each atom in the system.
        """
        ...


class CNModel(Node):
    """
    One coordination-number variant, as a value.

    A variant (``cn_d3``, ``cn_d4``, ...) differs from another only in
    data: the counting function and its parameters, cutoff, radii and
    electronegativity tables, pair weight and CN cap. This class holds
    that data and is itself the callable that computes the coordination
    number, so ``cn_d3(structure)`` keeps working. A different variant is
    obtained with :meth:`replace` (``cn_d4.replace(cutoff=40.0)``), never
    a subclass.

    A frozen :class:`~tad_mctc.tree.Node`: a model is a value, compared and
    hashed by identity, because comparing the tensor fields would raise. A
    tensor in ``cn_max``, ``rcov`` or ``en`` is a pytree leaf, so it can be
    differentiated or batched with ``torch.func``; a table function or a
    number is static. ``count`` and ``pair_weight`` are static too, and
    hashable by identity: two models only have the same tree structure if
    they share the same function objects.

    Parameters
    ----------
    count : CountingFunction
        Pair counting function, ``(r, r0) -> Tensor``. Bind its
        parameters with :func:`functools.partial`, e.g.
        ``partial(erf_count, kcn=2.0, norm_exp=0.75)``.
    cutoff : float, optional
        Real-space cutoff. A hard, Python-level mask, so it is never
        differentiated.
    cn_max : Tensor | float | int | None, optional
        ``None`` (default) means no cap. Otherwise the smooth cap from
        :func:`cut_coordination_number`, applied after the sum. A scalar,
        like mctc-lib's ``cut``: a per-atom cap raises. A tensor is moved
        to the coordination number's device and dtype, so it can be
        differentiated.
    rcov, en : Tensor | TableFunction, optional
        Per-element tables, indexed by atomic number (entry 0 is the
        dummy). Either a :class:`~tad_mctc.typing.TableFunction` such as
        :func:`tad_mctc.data.radii.COV_D3` or a :class:`Tensor`, resolved
        on the input's device and dtype at call time. ``en`` is only read
        when ``pair_weight`` is set.
    pair_weight : PairWeightFunction | None, optional
        ``(en_i, en_j) -> Tensor``, elementwise and broadcastable: the
        weight on atom ``j``'s count as it is added to ``cn[i]``.
    """

    count: CountingFunction = context()
    cutoff: float = context(default=25.0)
    cn_max: Tensor | float | int | None = child(default=None)
    rcov: Tensor | TableFunction = child(default=radii.COV_D3)
    en: Tensor | TableFunction = child(default=eneg.PAULING)
    pair_weight: PairWeightFunction | None = context(default=None)

    def _validate(self) -> None:
        # Checked once when the model is built. `replace()` builds through
        # `__init__`, so every variant is checked too, also under `vmap`
        # where `cn_max` is a 0-d tensor per lane.
        if isinstance(self.cn_max, Tensor) and self.cn_max.ndim != 0:
            raise ValueError(
                "`cn_max` must be a scalar (0-d tensor); a per-atom cap is not "
                f"supported (got shape {tuple(self.cn_max.shape)})."
            )

    def __call__(
        self,
        structure: Structure,
        pairs: PeriodicShifts | NeighborList | None = None,
        *,
        mode: NeighborListMode = "graph",
    ) -> Tensor:
        """
        Compute the coordination number of one ``Structure`` or a batch of
        them (``structure.numbers.ndim > 1``).

        ``pairs`` says how the atom pairs are enumerated. A molecule
        (``structure.lattice is None``) and a cell are both handled; the
        structure decides which.

        - ``None`` (default): all pairs. A molecule needs no preparation.
          For a cell, periodic shifts are built from ``structure.lattice`` and
          ``structure.periodic`` at ``self.cutoff`` on every call, under
          ``torch.no_grad()``, so it follows the lattice as it changes
          (NPT, cell relaxation). Its shape is data-dependent, so for a
          cell this default does not work under ``torch.func.vmap``,
          ``jacrev`` over ``structure.lattice`` or ``torch.compile``.
        - :class:`~tad_mctc.neighbor.images.PeriodicShifts`: all pairs, over
          periodic shifts built ahead of time by
          :func:`~tad_mctc.neighbor.images.build_periodic_shifts`.
          Nothing data-dependent remains, so this works under ``vmap``,
          ``jacrev`` and ``torch.compile(fullgraph=True)``, also over
          ``structure.lattice``. A system that needs fewer image rings than
          a shared table has the extra shifts masked out.
        - :class:`~tad_mctc.neighbor.list.NeighborList`: only the listed
          pairs, memory ``O(n_pairs)`` instead of ``O(nat**2)``. Building
          the list is data-dependent (``nonzero``, Python-level padding), so
          build it once with
          :func:`~tad_mctc.neighbor.list.build_neighborlist` from the same
          ``structure`` and reuse it. Consuming it is fixed-shape tensor
          algebra, so it works under autograd of any order, ``vmap`` and
          ``torch.compile(fullgraph=True)``. Its pairs are masked at
          ``self.cutoff`` again, so a list built with extra skin (or for a
          larger cutoff shared with another model) never changes the
          result. A batched ``structure`` needs a list built from that
          same batch. ``structure.lattice`` supplies the image shifts of a
          periodic list, so differentiating with respect to it sees the
          argument, not a build-time snapshot.

        Parameters
        ----------
        structure : Structure
            The system(s) to evaluate. For a ``PeriodicShifts`` it must have
            a lattice; for a ``NeighborList``, the structure the list was
            built from.
        pairs : PeriodicShifts | NeighborList | None, optional
            How pairs are enumerated, see above.
        mode : NeighborListMode, optional
            How a ``NeighborList`` is evaluated. Both modes walk the list
            in fixed-size chunks. ``"graph"`` (default) keeps every
            chunk's intermediates for the backward pass, memory
            ``O(n_pairs)`` under autograd, and works with ``vmap``,
            ``fullgraph`` compilation and any derivative order. Under
            ``torch.compile`` it evaluates the list as one chunk, which
            the compiler fuses. ``"recompute"`` wraps each chunk in
            :func:`torch.utils.checkpoint.checkpoint`, so the backward
            pass also holds only one chunk at a time. It supports any
            derivative order, but **not** ``vmap``. Anything but
            ``"graph"`` needs a ``NeighborList`` as ``pairs``.

        Returns
        -------
        Tensor
            Coordination numbers, shape ``(..., nat)``.

        Raises
        ------
        ValueError
            ``mode`` is unknown, or is ``"recompute"`` without a
            ``NeighborList``; or ``pairs`` is not compatible with
            ``structure`` and this model's ``cutoff`` (see
            :meth:`~tad_mctc.neighbor.images.PeriodicShifts.check_compatible`
            and :meth:`~tad_mctc.neighbor.list.NeighborList.check_compatible`).
        """
        _validate_mode(mode, pairs)
        if pairs is not None:
            pairs.check_compatible(structure, self.cutoff)

        rcov = resolve_table(self.rcov, structure.positions)
        en = (
            resolve_table(self.en, structure.positions)
            if self.pair_weight is not None
            else None
        )

        if isinstance(pairs, NeighborList):
            cn = _cn_sparse(
                structure.numbers,
                structure.positions,
                rcov=rcov,
                en=en,
                count=self.count,
                pair_weight=self.pair_weight,
                cutoff=self.cutoff,
                nbl=pairs,
                lattice=structure.lattice,
                mode=mode,
            )
        else:
            images = _dense_images(structure, pairs, cutoff=self.cutoff)
            cn = _cn_dense(
                structure.numbers,
                images,
                rcov=rcov,
                en=en,
                count=self.count,
                pair_weight=self.pair_weight,
                cutoff=self.cutoff,
            )

        if self.cn_max is None:
            return cn

        return cut_coordination_number(cn, self.cn_max)


def _masked_pair_counts(
    distance_squared: Tensor,
    rcov_pair_sum: Tensor,
    valid: Tensor,
    *,
    count: CountingFunction,
    cutoff: float,
) -> Tensor:
    """
    The pair kernel every evaluation path shares: the unweighted count of
    each pair, zero wherever the pair is not ``valid`` or lies beyond
    ``cutoff``.

    All arguments are elementwise and broadcast against each other, so the
    dense paths pass ``(..., nat, nat[, n_shift])`` grids and the sparse
    path passes flat ``(n_pairs,)`` lists.

    Parameters
    ----------
    distance_squared : Tensor
        Squared pair distances.
    rcov_pair_sum : Tensor
        Sum of the two atoms' covalent radii.
    valid : Tensor
        ``True`` for a pair that exists: not padding and not an atom with
        itself at the zero shift.
    count : CountingFunction
        Pair counting function, ``(r, r0) -> Tensor``.
    cutoff : float
        Real-space cutoff.

    Returns
    -------
    Tensor
        Masked counts, broadcast shape of the arguments.
    """
    within_cutoff = valid & (distance_squared <= cutoff**2)

    # Mask before the square root, not after: an atom with itself (or a
    # padding slot) has `distance_squared == 0`, where `sqrt` has an
    # infinite derivative that would turn the masked gradient into `nan`.
    # One expression, so the filled copy is freed right after the square
    # root instead of staying alive through `count`.
    #
    # The fill values are plain Python scalars, which `torch.where`
    # broadcasts without allocating a full replacement tensor.
    distance = torch.where(within_cutoff, distance_squared, 1.0).sqrt()

    counts = count(distance, rcov_pair_sum)
    return torch.where(within_cutoff, counts, 0.0)


class _Images(NamedTuple):
    """
    The geometry the dense kernel sums over: every atom paired with every
    periodic image of every atom. A molecule is the special case of one
    zero translation.
    """

    positions: Tensor
    """Cartesian coordinates, ``(..., nat, 3)``. Folded into the central
    cell for a cell (see :func:`_periodic_images`)."""

    translations: Tensor | None
    """Cartesian translation of each image, ``(..., n_shift, 3)``, or
    ``None`` for a molecule, whose single image is the zero translation."""

    valid: Tensor
    """``True`` for the ``(i, j, image)`` entries that are real pairs,
    ``(..., nat, nat, n_shift)`` (``n_shift == 1`` for a molecule)."""


def _molecular_images(numbers: Tensor, positions: Tensor) -> _Images:
    """
    The images of a molecule: the atoms themselves, at a single zero
    translation. Only padding and an atom with itself are not real pairs.
    """
    valid = real_pairs(numbers, mask_diagonal=True).unsqueeze(-1)
    return _Images(positions, None, valid)


def _periodic_images(
    numbers: Tensor,
    positions: Tensor,
    lattice: Tensor,
    shifts: Tensor,
    periodic: Tensor,
) -> _Images:
    """
    The images of a cell, one per entry of ``shifts``.

    Differentiable in ``positions`` and ``lattice``: the fold into the
    central cell only adds integer cell offsets.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers, shape ``(..., nat)``. ``0`` marks batch padding.
    positions : Tensor
        Cartesian coordinates, shape ``(..., nat, 3)``.
    lattice : Tensor
        Lattice vectors as rows, shape ``(..., 3, 3)``, in Bohr.
    shifts : Tensor
        Integer lattice translations, shape ``(n_shift, 3)``, shared by
        every system in ``(...)``, never batched itself.
    periodic : Tensor
        Boolean mask, shape ``(3,)`` or ``(..., 3)``, which axes are
        periodic **per system**. Always each system's own mask, never a
        shared table's (possibly batch-wide) one: folding or translating
        along an axis that is not periodic for a system changes that
        system's physics.

    Returns
    -------
    _Images
        The folded positions, the image translations and the valid pairs.
    """
    # `shifts` only covers a cutoff sphere anchored at the primary cell
    # (see `build_periodic_shifts`), so it is only valid once every atom
    # lies inside that cell, the invariant mctc-lib's own
    # `wrap_to_central_cell` establishes. The wrapped positions are the
    # caller's `positions` plus whole-cell offsets, so they stay
    # differentiable in `positions` and `lattice`. `periodic.unsqueeze(-2)`
    # gives the mask an explicit atom axis, so a batched `(..., 3)` mask
    # cannot align with the atom axis when a batch size equals `nat`.
    folded_positions, _ = wrap_to_central_cell(
        positions, lattice, periodic.unsqueeze(-2)
    )  # (..., nat, 3)

    # (..., n_shift, 3): `shifts` broadcasts over the batch of `lattice`.
    translations = shifts.to(positions.dtype) @ lattice

    valid = _periodic_valid_pairs(numbers, shifts, periodic)
    return _Images(folded_positions, translations, valid)


def _dense_images(
    structure: Structure, shifts: PeriodicShifts | None, *, cutoff: float
) -> _Images:
    """
    The images for :func:`_cn_dense`: a molecule's own atoms, or a cell's
    images over ``shifts``, built here if not given.
    """
    if structure.lattice is None:
        return _molecular_images(structure.numbers, structure.positions)

    # `Structure` fills in a mask whenever it has a lattice.
    assert structure.periodic is not None
    if shifts is None:
        shifts = build_periodic_shifts(
            structure.lattice, structure.periodic, cutoff
        )
    return _periodic_images(
        structure.numbers,
        structure.positions,
        structure.lattice,
        shifts.shifts,
        structure.periodic,
    )


def _cn_dense(
    numbers: Tensor,
    images: _Images,
    *,
    rcov: Tensor,
    en: Tensor | None,
    count: CountingFunction,
    pair_weight: PairWeightFunction | None,
    cutoff: float,
) -> Tensor:
    """
    Dense coordination number: every real pair evaluated at every image,
    masked at ``cutoff``. The same kernel serves molecules and cells.

    Batched over a leading ``(...)`` dimension: every intermediate inserts
    its new axes counted from the end of the shape, so no branch on the
    number of batch dimensions is needed. Consuming ``images`` is
    fixed-shape tensor algebra, so this stays differentiable to any order
    and ``vmap``-friendly as long as one shift table covers every batch
    element.

    An atom pairing with its own periodic image (``i == j`` at a non-zero
    shift) is a real pair; only the true self-pair is excluded.

    Distances come from direct position differences, not the quadratic
    expansion ``|x|^2 + |y|^2 - 2 x.y``: the expansion cancels large terms
    and loses about an order of magnitude of ``float32`` accuracy, while
    the difference tensor costs ``O(nat**2 * n_shift)`` memory either way
    once there are images. The neighbour-list path is the low-memory one.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers, shape ``(..., nat)``. ``0`` marks batch padding.
    images : _Images
        Geometry to sum over, see :func:`_molecular_images` and
        :func:`_periodic_images`.
    rcov, en : Tensor | None
        Resolved per-element tables; ``en`` is ``None`` unless
        ``pair_weight`` is set.
    count : CountingFunction
        Pair counting function.
    pair_weight : PairWeightFunction | None
        See :class:`CNModel`.
    cutoff : float
        Real-space cutoff.

    Returns
    -------
    Tensor
        Coordination numbers, shape ``(..., nat)``.
    """
    species = _species(numbers)
    rcov_species = rcov[species]  # (..., nat)
    rcov_pair_sum = rcov_species[..., :, None] + rcov_species[..., None, :]
    rcov_pair_sum = rcov_pair_sum.unsqueeze(-1)  # (..., nat, nat, 1)

    # Entry `(i, j, image)` points from atom `i` to that image of atom `j`.
    positions = images.positions
    pair_vector = positions[..., None, :, :] - positions[..., :, None, :]
    if images.translations is None:
        # The single image is the atom itself: nothing to add, and a view
        # spares a second `(..., nat, nat, 3)` tensor.
        difference = pair_vector.unsqueeze(-2)
    else:
        translations = images.translations[..., None, None, :, :]
        difference = pair_vector[..., None, :] + translations
    distance_squared = (difference * difference).sum(-1)
    del pair_vector, difference

    counts = _masked_pair_counts(
        distance_squared,
        rcov_pair_sum,
        images.valid,
        count=count,
        cutoff=cutoff,
    )
    # Not needed any more; free it before the pair weight allocates its
    # own `(..., nat, nat, n_shift)` tensors.
    del distance_squared

    if pair_weight is not None:
        assert en is not None
        en_species = en[species]  # (..., nat)
        # Entry `(i, j, image)` is what atom `i` receives from that image of
        # atom `j`.
        en_i = en_species[..., :, None, None]
        en_j = en_species[..., None, :, None]
        counts = pair_weight(en_i, en_j) * counts

    return counts.sum(dim=(-2, -1))


def _periodic_valid_pairs(
    numbers: Tensor, shifts: Tensor, periodic: Tensor
) -> Tensor:
    """
    Which ``(i, j, shift)`` entries of the pair grid :func:`_cn_dense` sums
    over are real pairs: both atoms real, not an atom with itself at the zero
    shift, and no translation along an axis the system is not periodic
    along.

    An atom with its own image at a non-zero shift is a real pair. The
    self-pair test can use the raw ``shifts`` because for ``i == j`` the
    wrap's cell offsets cancel.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers, shape ``(..., nat)``. ``0`` marks batch padding.
    shifts : Tensor
        Integer lattice translations, shape ``(n_shift, 3)``.
    periodic : Tensor
        Periodic axes of each system, shape ``(3,)`` or ``(..., 3)``.

    Returns
    -------
    Tensor
        Boolean mask, shape ``(..., nat, nat, n_shift)``.
    """
    both_real = real_pairs(numbers, mask_diagonal=False)  # (..., nat, nat)
    both_real = both_real.unsqueeze(-1)  # (..., nat, nat, 1)

    is_same_atom = torch.eye(
        numbers.shape[-1], dtype=torch.bool, device=numbers.device
    )
    is_zero_shift = (shifts == 0).all(-1)  # (n_shift,)
    # (nat, nat, n_shift), shared by every system of a batch
    is_self_at_zero_shift = is_same_atom.unsqueeze(-1) & is_zero_shift

    # A batch shares one shift table, which may translate along an axis
    # that is periodic for another system but not for this one. Such an
    # image does not exist, and the cutoff check alone does not remove
    # it: a non-periodic axis may carry a short placeholder lattice
    # vector, so its images can land well inside the cutoff.
    moves_along_open_axis = (shifts != 0) & ~periodic.unsqueeze(-2)
    is_real_image = ~moves_along_open_axis.any(-1)  # (..., n_shift)
    is_real_image = is_real_image[..., None, None, :]  # (..., 1, 1, n_shift)

    return both_real & ~is_self_at_zero_shift & is_real_image


class _PaddedAtoms(NamedTuple):
    """
    Per-atom data of a (flattened) batch for the sparse path, with one
    phantom atom appended at index ``total_atoms``, where padded list
    slots point.

    The phantom's position is a constant with no gradient path back to
    the real atoms, and its tables resolve element 1 (see
    :func:`_species`), so a padded slot never contaminates a value or a
    derivative, even before its contribution is masked to zero.
    """

    position_columns: tuple[Tensor, Tensor, Tensor]
    """Cartesian coordinates, split into the ``x``, ``y`` and ``z`` columns
    (``(total_atoms + 1,)`` each) that the pair distances gather from."""

    rcov: Tensor
    """Covalent radius of each atom, ``(total_atoms + 1,)``."""

    en: Tensor | None
    """Electronegativity of each atom, ``(total_atoms + 1,)``, or ``None``
    for an unweighted model."""

    shared_lattice: Tensor | None
    """One ``(3, 3)`` cell used by every pair, or ``None``."""

    system_lattices: Tensor | None
    """One cell per system, ``(n_systems + 1, 3, 3)`` with a phantom cell
    for the phantom atom's system, or ``None``. At most one of the two
    lattice fields is set; neither is for a molecule."""

    atoms_per_system: int
    """Atoms per system in the padded batch, so that atom ``k`` belongs
    to system ``k // atoms_per_system``."""


def _pad_atoms(
    numbers: Tensor,
    positions: Tensor,
    *,
    rcov: Tensor,
    en: Tensor | None,
    lattice: Tensor | None,
    atoms_per_system: int,
) -> _PaddedAtoms:
    """
    Look up each atom's table entries and append the phantom atom (and,
    for per-system cells, a phantom cell).

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers of the flattened batch, shape ``(total_atoms,)``.
    positions : Tensor
        Cartesian coordinates, shape ``(total_atoms, 3)``.
    rcov, en : Tensor | None
        Resolved per-element tables; ``en`` is ``None`` for an unweighted
        model.
    lattice : Tensor | None
        ``None``, one cell (``(3, 3)`` or ``(1, 3, 3)``, shared by every
        system), or ``(..., 3, 3)`` per system.
    atoms_per_system : int
        Atoms per system in the padded batch.

    Returns
    -------
    _PaddedAtoms
        The padded per-atom data.
    """
    species = torch.cat([numbers, numbers.new_ones(1)])
    padded_positions = torch.cat([positions, positions.new_zeros(1, 3)])

    shared_lattice, system_lattices = split_lattice(lattice)
    if system_lattices is not None:
        system_lattices = torch.cat(
            [system_lattices, system_lattices.new_zeros(1, 3, 3)]
        )

    return _PaddedAtoms(
        position_columns=position_columns(padded_positions),
        rcov=rcov[species],
        en=None if en is None else en[species],
        shared_lattice=shared_lattice,
        system_lattices=system_lattices,
        atoms_per_system=atoms_per_system,
    )


def _sparse_pair_contributions(
    idx_i: Tensor,
    idx_j: Tensor,
    mask: Tensor,
    shift: Tensor,
    atoms: _PaddedAtoms,
    *,
    count: CountingFunction,
    pair_weight: PairWeightFunction | None,
    cutoff: float,
) -> tuple[Tensor, Tensor]:
    """
    The amounts each pair of a neighbour list (or a chunk of one) adds to
    atom ``idx_i`` and to atom ``idx_j``.

    One list entry stands for the pair in both directions. Without a pair
    weight both atoms receive the same count, so the same tensor is
    returned twice; with one, each orientation is weighted separately.
    """
    distance_squared = pair_distance_squared_from_columns(
        idx_i,
        idx_j,
        shift,
        atoms.position_columns,
        shared_lattice=atoms.shared_lattice,
        system_lattices=atoms.system_lattices,
        atoms_per_system=atoms.atoms_per_system,
    )
    rcov = atoms.rcov
    rcov_pair_sum = rcov.index_select(0, idx_i) + rcov.index_select(0, idx_j)

    counts = _masked_pair_counts(
        distance_squared, rcov_pair_sum, mask, count=count, cutoff=cutoff
    )

    if pair_weight is None:
        return counts, counts

    assert atoms.en is not None
    en_i = atoms.en.index_select(0, idx_i)
    en_j = atoms.en.index_select(0, idx_j)
    # A masked count is exactly zero, so it stays zero for any finite
    # weight. The weight is taken as seen from the receiving atom, so an
    # antisymmetric one gives the two atoms different amounts.
    to_i = pair_weight(en_i, en_j) * counts
    to_j = pair_weight(en_j, en_i) * counts
    return to_i, to_j


def sum_over_neighborlist(
    nbl: NeighborList,
    pair_contributions: Callable[
        [Tensor, Tensor, Tensor, Tensor], tuple[Tensor, Tensor]
    ],
    positions: Tensor,
    *,
    mode: NeighborListMode = "graph",
) -> Tensor:
    """
    Sum a per-pair quantity over a :class:`.NeighborList` into one value
    per atom, in chunks of the list's pair slots that bound the memory.

    This is the walk behind the sparse coordination number, for any other
    per-atom pair sum (a dispersion coordination number, a pairwise
    energy, ...): the caller supplies only the physics of a chunk of
    pairs.

    Consumption is pure, fixed-shape tensor algebra (given a
    ``pair_contributions`` that is): no ``nonzero``, no ``.item()``, no
    data-dependent control flow, so it survives autograd to any order,
    ``vmap`` and ``torch.compile(fullgraph=True)`` (mode ``"graph"``).

    Parameters
    ----------
    nbl : NeighborList
        The pre-built neighbour list to sum over. A batch is summed as one
        flat system, atom ``i`` of system ``b`` being ``b * nat + i``.
    pair_contributions : Callable
        ``pair_contributions(idx_i, idx_j, mask, shift) -> (to_i, to_j)``,
        called on one chunk of the list's slots: index, mask and shift
        tensors of shape ``(n,)``, ``(n,)``, ``(n,)`` and ``(n, 3)``. It
        returns the ``(n,)`` amounts that the pair adds to atom ``idx_i``
        and to atom ``idx_j`` (the same tensor twice for a symmetric
        quantity). Slots where ``mask`` is ``False`` are padding, which
        point at the phantom atom ``positions.shape[0]``: the function
        must give them zero, and may look the phantom atom up. The indices
        come as :func:`.gather_index` returns them: the list's ``int32``
        on PyTorch 2.8 and later, ``int64`` before. Per-atom data gathered
        from them should be one column per quantity, not rows of a table
        (see :func:`.pair_distance_squared_from_columns`).
    positions : Tensor
        Cartesian coordinates of the flattened batch, ``(total_atoms, 3)``.
        Sets the device, the dtype and the chunk size, and ties the result
        to the graph of the positions.
    mode : NeighborListMode, optional
        ``"graph"`` keeps every chunk's intermediates for the backward
        pass; ``"recompute"`` checkpoints each chunk and recomputes it
        there instead, so that memory stays at one chunk (not compatible
        with ``torch.compile``, see :meth:`CNModel.__call__`). Defaults to
        ``"graph"``.

    Returns
    -------
    Tensor
        The per-atom sums, shape ``(total_atoms,)``.
    """
    total_atoms = positions.shape[0]

    # `capacity` is a plain Python int (a tensor's shape is always static),
    # so the chunk loop has a fixed trip count under tracing.
    capacity = nbl.idx_i.shape[0]
    chunk = _chunk_size(positions)
    if mode == "graph" and is_compiling():
        # The compiler fuses the pair kernel itself; a chunk loop would only
        # be unrolled into a longer compile and a slower result.
        chunk = max(capacity, 1)

    # One extra slot for the phantom atom, dropped at the end.
    counts = positions.new_zeros(total_atoms + 1)

    # At least one chunk, empty for a list without pairs, so that the result
    # is still part of the graph of `positions` (with a zero gradient).
    for start in range(0, max(capacity, 1), chunk):
        stop = min(start + chunk, capacity)
        # The list stores `int32`. From PyTorch 2.8 the slices are used as
        # they are: views, so autograd keeps them for the backward pass at no
        # cost. Before 2.8 the backward pass needs `int64` indices, and
        # `gather_index` widens this slice (never the whole list); every
        # node that uses it then keeps the copy until the backward pass.
        chunk_i = gather_index(nbl.idx_i[start:stop])
        chunk_j = gather_index(nbl.idx_j[start:stop])
        chunk_mask = nbl.mask[start:stop]
        chunk_shift = nbl.shift[start:stop]

        if mode == "recompute":
            # Recomputed in the backward pass instead of stored, so memory
            # stays at one chunk.
            to_i, to_j = (
                torch.utils.checkpoint.checkpoint(  # pyright: ignore[reportGeneralTypeIssues]
                    pair_contributions,
                    chunk_i,
                    chunk_j,
                    chunk_mask,
                    chunk_shift,
                    use_reentrant=False,
                )
            )
        else:
            to_i, to_j = pair_contributions(
                chunk_i, chunk_j, chunk_mask, chunk_shift
            )

        # Out of place, `index_add` copies all of `counts` per call, which
        # for a large system costs more than the pair kernel itself. In
        # place is not possible on a tensor inside `torch.func`.
        operands = (counts, chunk_i, chunk_j, to_i, to_j)
        if any(is_functorch_tensor(t) for t in operands):
            counts = counts.index_add(0, chunk_i, to_i)
            counts = counts.index_add(0, chunk_j, to_j)
        else:
            counts.index_add_(0, chunk_i, to_i)
            counts.index_add_(0, chunk_j, to_j)

    return counts[:total_atoms]


def _cn_sparse(
    numbers: Tensor,
    positions: Tensor,
    *,
    rcov: Tensor,
    en: Tensor | None,
    count: CountingFunction,
    pair_weight: PairWeightFunction | None,
    cutoff: float,
    nbl: NeighborList,
    lattice: Tensor | None,
    mode: NeighborListMode,
) -> Tensor:
    """
    Coordination number from a pre-built :class:`.NeighborList` instead of
    the dense all-pairs sum.

    Consumption is pure, fixed-shape tensor algebra: no ``nonzero``, no
    ``.item()``, no data-dependent control flow, so it survives autograd
    to any order, ``vmap`` and ``torch.compile(fullgraph=True)`` (mode
    ``"graph"``; see :meth:`CNModel.__call__`'s ``mode`` for why
    ``"recompute"`` does not).

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers, shape ``(nat,)`` or ``(..., nat)`` for a batch,
        ``0`` marking padding.
    positions : Tensor
        Cartesian coordinates, shape ``(..., nat, 3)``.
    rcov, en : Tensor | None
        Resolved per-element tables; ``en`` is ``None`` unless
        ``pair_weight`` is set.
    count : CountingFunction
        Pair counting function.
    pair_weight : PairWeightFunction | None
        See :class:`CNModel`.
    cutoff : float
        Real-space cutoff. Masked here even though ``nbl`` already
        truncated the search, because a list built with skin (or shared
        across variants at a larger cutoff) may hold pairs beyond this
        model's own cutoff.
    nbl : NeighborList
        The pre-built neighbour list to sum over.
    lattice : Tensor | None
        Lattice vectors as rows, shape ``(3, 3)`` or per system
        ``(..., 3, 3)``, forming the periodic image shift.
    mode : NeighborListMode
        ``"graph"`` or ``"recompute"``, see :meth:`CNModel.__call__`.

    Returns
    -------
    Tensor
        Coordination numbers, shape ``(..., nat)``.
    """
    # A batch is summed as one flat system: the list indexes atom `i` of
    # system `b` as `b * atoms_per_system + i` (see `NeighborList`).
    numbers_shape = numbers.shape
    atoms_per_system = numbers.shape[-1]
    numbers = numbers.reshape(-1)
    positions = positions.reshape(-1, 3)

    atoms = _pad_atoms(
        numbers,
        positions,
        rcov=rcov,
        en=en,
        lattice=lattice,
        atoms_per_system=atoms_per_system,
    )
    pair_contributions = partial(
        _sparse_pair_contributions,
        atoms=atoms,
        count=count,
        pair_weight=pair_weight,
        cutoff=cutoff,
    )

    counts = sum_over_neighborlist(
        nbl, pair_contributions, positions, mode=mode
    )
    return counts.reshape(numbers_shape)


def cut_coordination_number(
    cn: Tensor, cn_max: Tensor | float | int = defaults.CUTOFF_EEQ_MAX
) -> Tensor:
    """
    Apply the smooth logarithmic cutoff used throughout mctc projects.

    Parameters
    ----------
    cn : Tensor
        Coordination numbers.
    cn_max : Tensor | float | int, optional
        Maximum coordination number. Large values disable the cutoff.

    Returns
    -------
    Tensor
        Cut coordination numbers.
    """
    if isinstance(cn_max, Tensor):
        cn_max_tensor = cn_max.to(device=cn.device, dtype=cn.dtype)
    else:
        # The `> 50` shortcut only applies to a plain Python number: it
        # disables the cap exactly. Branching on a *tensor's* value here
        # instead would read tensor data at trace time and break
        # `torch.compile(fullgraph=True)`.
        if cn_max > 50:
            return cn
        cn_max_tensor = torch.tensor(cn_max, device=cn.device, dtype=cn.dtype)

    zero = cn.new_zeros(())
    return torch.logaddexp(zero, cn_max_tensor) - torch.logaddexp(
        zero, cn_max_tensor - cn
    )
