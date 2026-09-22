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
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

import torch

from .. import storch
from ..autograd import is_functorch_tensor
from ..batch import real_pairs
from ..data import en as eneg
from ..data import radii
from ..io.structure import Structure
from ..neighbor.images import (
    PeriodicShifts,
    build_periodic_shifts,
    build_shared_periodic_shifts,
    wrap_to_central_cell,
)
from ..tools import is_compiling
from ..typing import (
    DD,
    CountingFunction,
    PairWeightFunction,
    TableFunction,
    Tensor,
)
from . import defaults

__all__ = ["CNModel", "cut_coordination_number"]

# Cache of resolved `TableFunction` tensors, keyed by the table's own
# identity (not any `CNModel` instance) plus the device/dtype it was
# resolved for. Function-level, not instance state: a `CNModel` is a
# frozen dataclass and `dataclasses.replace()` copies are meant to share
# this rather than each holding (and possibly going stale on) their own
# cached tensor.
#
# Keyed on the table object itself (relying on the default,
# identity-based `__hash__`/`__eq__` every plain function and callable
# object has unless it opts out) rather than on `id(table)`: using a bare
# `id()` as a dict key does not keep the underlying object alive, so once
# it is garbage-collected CPython is free to reuse that same id for an
# unrelated object, which would then collide with a stale cache entry.
# Using the table object as (part of) the key instead makes the cache
# dict hold a strong reference to it, so that failure mode cannot occur.
# Module-level `TableFunction`s such as `radii.COV_D3` already live for
# the process lifetime regardless, so this only matters for callables
# constructed more dynamically.
#
# Cardinality in practice is tiny (a handful of named tables times a
# handful of device/dtype combinations actually used), so a plain dict
# that is never evicted is simpler than `functools.lru_cache` and avoids
# any risk of its wrapping interacting badly with `torch.compile`.
_TABLE_CACHE: dict[tuple[TableFunction, torch.device, torch.dtype], Tensor] = {}


def _resolve_table(table: Tensor | TableFunction, like: Tensor) -> Tensor:
    """
    Resolve a per-element table on the device and dtype of ``like``.

    A plain :class:`Tensor` is just moved with :meth:`Tensor.to` (a no-op
    if it is already on the right device/dtype). A :class:`TableFunction`
    such as :func:`tad_mctc.data.radii.COV_D3` is memoized at function
    level, keyed by the table's identity and ``(device, dtype)``: the
    first call for a given combination builds and caches the tensor
    (including any H2D copy), every later call for that same combination
    reuses it instead of rebuilding it from scratch.

    Parameters
    ----------
    table : Tensor | TableFunction
        Either an already-built table, or a callable such as
        :func:`tad_mctc.data.radii.COV_D3` that builds one for a given
        device and dtype.
    like : Tensor
        Tensor whose device and dtype the table is resolved to.

    Returns
    -------
    Tensor
        The table, one entry per atomic number. Callers must treat it as
        read-only: it is shared, via the cache, with every other call
        site that resolves the same table on the same device/dtype.
    """
    if isinstance(table, Tensor):
        return table.to(device=like.device, dtype=like.dtype)

    key = (table, like.device, like.dtype)
    cached = _TABLE_CACHE.get(key)
    if cached is None:
        cached = table(device=like.device, dtype=like.dtype)
        _TABLE_CACHE[key] = cached
    return cached


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


def _validate_call(cn_max: Tensor | float | int | None) -> None:
    """
    Python-level validation for :meth:`CNModel.__call__` and
    :meth:`CNModel.with_precomputed_shifts`, run before any tensor work.

    Only ``cn_max`` is checked here: a shape mismatch between ``structure.
    numbers`` and ``structure.positions`` cannot reach this function,
    since ``Structure.__post_init__`` (``structure_check``) already
    rejects that when the ``Structure`` is built --
    including through :meth:`Structure.replace`, which reruns
    ``__post_init__`` -- so every ``Structure`` `CNModel` receives already
    has consistent shapes.
    """
    if isinstance(cn_max, Tensor) and cn_max.ndim != 0:
        raise ValueError(
            "`cn_max` must be a scalar (0-d tensor); a per-atom cap is not "
            f"supported (got shape {tuple(cn_max.shape)})."
        )


@dataclass(frozen=True, eq=False)
class CNModel:
    """
    One coordination-number variant, as a value.

    A variant (``cn_d3``, ``cn_d4``, ...) differs from another only in
    data: the counting function and its parameters, cutoff, radii and
    electronegativity tables, pair weight and CN cap. This class holds
    that data and is itself the callable that computes the coordination
    number, so ``cn_d3(structure)`` keeps working. A different variant is
    obtained with :meth:`replace` (``cn_d4.replace(cutoff=40.0)``), never
    a subclass.

    Frozen because a model is a value, and ``eq=False`` because a
    generated ``__eq__`` would compare the tensor fields and raise.

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

    count: CountingFunction
    cutoff: float = 25.0
    cn_max: Tensor | float | int | None = None
    rcov: Tensor | TableFunction = radii.COV_D3
    en: Tensor | TableFunction = eneg.PAULING
    pair_weight: PairWeightFunction | None = None

    def __call__(self, structure: Structure) -> Tensor:
        """
        Compute the dense (all-pairs) coordination number for one
        ``Structure`` or a batch of them (``structure.numbers.ndim > 1``).

        ``structure.lattice is None`` is the plain molecular path (no
        periodic images); otherwise this is the periodic path's
        auto-build tier: a shift table is built fresh from
        ``structure.lattice``/``structure.periodic`` at ``self.cutoff``
        every call, under ``torch.no_grad()`` -- tracking whatever
        ``structure.lattice`` currently is (e.g. across an NPT/cell-
        relaxation trajectory) at the cost of a data-dependent shape
        every call, which is not ``vmap``/``jacrev``-over-``lattice``
        friendly. :meth:`with_precomputed_shifts` is the friendly
        alternative for that case -- see its own docstring for why it
        has to be a separate method rather than a keyword here (in
        short: rebuilding the shift table *inside* a
        ``torch.func.vmap``-over-``lattice`` trace hits real
        ``RuntimeError: vmap: ... data-dependent control flow`` failures,
        so reusing a precomputed table needs its own entry point either
        way; this method stays the single default one for everything
        else).

        Parameters
        ----------
        structure : Structure
            The system(s) to evaluate.

        Returns
        -------
        Tensor
            Coordination numbers, shape ``(..., nat)``.

        Raises
        ------
        ValueError
            On a non-scalar tensor ``cn_max`` (see :func:`_validate_call`).
        """
        _validate_call(self.cn_max)
        return self._dispatch(structure, shifts=None)

    def with_precomputed_shifts(
        self,
        structure: Structure,
        *,
        shifts: PeriodicShifts,
    ) -> Tensor:
        """
        Compute the dense periodic coordination number from a shift table
        built ahead of time, instead of rebuilding one from
        ``structure.lattice`` every call.

        This is :meth:`__call__`'s only periodic-specific counterpart:
        the shape ``torch.func.vmap``/``jacrev`` over a batch of
        ``lattice`` values needs, since rebuilding the shift table
        *inside* such a trace hits real, already-observed failures
        (``RuntimeError: vmap: ... data-dependent control flow``,
        because different lattices in a batch can legitimately need
        different ring counts -- a per-lane-varying shape ``vmap``
        cannot represent). Pass a ``shifts`` table that already covers
        every lattice the trace will see (see
        :func:`tad_mctc.neighbor.images.build_shared_periodic_shifts`
        for one sized to a whole batch at once) and this method performs
        no data-dependent rebuild at all.

        ``structure.numbers.ndim > 1`` (a real leading batch dimension,
        distinct from ``vmap``-over-several-single-system-``Structure``s
        above) routes to the batched dense-periodic path: atom counts
        pad exactly like the molecular path (:func:`_cn_dense_mol`, via
        ``real_pairs`` on ``numbers == 0``), and a system needing fewer
        image rings than the batch's shared table gets its extra shift
        entries masked out, not rejected.

        Both the wrap that folds positions into the primary cell and the
        choice of which shifts are real images use each system's *own*
        ``structure.periodic``, never a shift table's (possibly
        batch-reduced) one -- see :func:`_cn_dense_per`'s ``periodic``
        parameter docs for why that distinction matters.

        Parameters
        ----------
        structure : Structure
            The system(s) to evaluate. ``structure.lattice`` must be set.
        shifts : PeriodicShifts
            The precomputed shift table. Its ``.periodic`` only records
            what the table was built for; ``structure.periodic`` decides
            which axes are periodic (see above).

        Returns
        -------
        Tensor
            Coordination numbers, shape ``(..., nat)``.

        Raises
        ------
        ValueError
            On a non-scalar tensor ``cn_max`` (see :func:`_validate_call`);
            ``structure.lattice is None``; or ``shifts.cutoff`` smaller
            than ``self.cutoff``, or ``shifts.periodic`` missing a periodic
            axis of ``structure`` (either under-built table would otherwise
            silently under-count instead of raising).
        """
        _validate_call(self.cn_max)

        if structure.lattice is None:
            raise ValueError(
                "`structure.lattice` is `None`; a `Structure` with no "
                "lattice has nothing periodic to evaluate."
            )

        return self._dispatch(structure, shifts=shifts)

    def replace(self, **changes: Any) -> CNModel:
        """
        Copy this model with some fields swapped out, e.g. a different
        ``cutoff`` or ``cn_max``.

        Thin wrapper around :func:`dataclasses.replace` so call sites do
        not need their own import of it. This is the documented way to
        get a different variant (see the class docstring); a subclass is
        not.

        Parameters
        ----------
        **changes : Any
            Field name/value pairs to override, e.g. ``cutoff=40.0``.

        Returns
        -------
        CNModel
            A new instance with the given fields replaced.
        """
        return dataclasses.replace(self, **changes)

    def _dispatch(
        self, structure: Structure, *, shifts: PeriodicShifts | None
    ) -> Tensor:
        """
        Shared core behind both public entry points, :meth:`__call__` and
        :meth:`with_precomputed_shifts`: resolves ``rcov``/``en`` once,
        routes to the plain molecular path or :meth:`_periodic`, then
        applies the ``cn_max`` cap. ``shifts`` only matters once routed
        to :meth:`_periodic` -- reached from :meth:`__call__` with
        ``shifts=None`` (auto-build) or from
        :meth:`with_precomputed_shifts` with an explicit table; the
        molecular path (``structure.lattice is None``) is only reachable
        from :meth:`__call__`, since :meth:`with_precomputed_shifts`
        rejects that case before ever calling here.
        """
        rcov = _resolve_table(self.rcov, structure.positions)
        en = (
            _resolve_table(self.en, structure.positions)
            if self.pair_weight is not None
            else None
        )

        if structure.lattice is None:
            cn = _cn_dense_mol(
                structure.numbers,
                structure.positions,
                rcov=rcov,
                en=en,
                count=self.count,
                pair_weight=self.pair_weight,
                cutoff=self.cutoff,
            )
        else:
            cn = self._periodic(
                structure,
                lattice=structure.lattice,
                shifts=shifts,
                rcov=rcov,
                en=en,
            )

        if self.cn_max is None:
            return cn

        return cut_coordination_number(cn, self.cn_max)

    def _periodic(
        self,
        structure: Structure,
        *,
        lattice: Tensor,
        shifts: PeriodicShifts | None,
        rcov: Tensor,
        en: Tensor | None,
    ) -> Tensor:
        """
        Periodic-evaluation core: shift-table resolution/validation and
        the wrap-mask/batch-mask split, shared between :meth:`__call__`'s
        auto-build tier (``shifts=None``) and :meth:`with_precomputed_
        shifts`'s explicit tier. Not itself part of the public API -- see
        both callers' docstrings, via :meth:`_dispatch`, for what each
        tier is for.

        ``lattice``, ``rcov`` and ``en`` arrive already resolved/narrowed
        from :meth:`_dispatch` (``lattice`` is ``structure.lattice``,
        already known non-``None`` there), so this method itself never
        touches ``structure.lattice``, ``self.rcov`` or ``self.en``.
        """
        batched = structure.numbers.ndim > 1
        # `Structure` fills in a mask whenever it has a lattice.
        periodic = structure.periodic
        assert periodic is not None

        if shifts is None:
            with torch.no_grad():
                shifts = (
                    build_shared_periodic_shifts(
                        lattice, periodic, cutoff=self.cutoff
                    )
                    if batched
                    else build_periodic_shifts(
                        lattice, periodic, cutoff=self.cutoff
                    )
                )
        elif shifts.cutoff < self.cutoff:
            raise ValueError(
                f"`shifts.cutoff` ({shifts.cutoff}) is smaller than this "
                f"model's own `cutoff` ({self.cutoff}); an under-built "
                "table would silently under-count real periodic-image "
                "pairs. Build it with `build_periodic_shifts(lattice, "
                "periodic, cutoff=model.cutoff)` at at least this "
                "model's own `cutoff`."
            )
        elif _misses_periodic_axis(shifts.periodic, periodic):
            raise ValueError(
                f"`shifts.periodic` ({shifts.periodic.tolist()}) does not "
                "cover every periodic axis of the structure "
                f"({periodic.tolist()}); the images along the missing axes "
                "would be silently dropped. Build the table with the "
                "structure's own `periodic` mask."
            )

        # The wrap and the shift masking follow each system's own mask,
        # never the table's: a shared table's `.periodic` may be the union
        # over a batch (or over the lanes of a `vmap`), which is right for
        # sizing the table but wrong for any one system -- see
        # `_cn_dense_per`'s `periodic` parameter docs.
        return _cn_dense_per(
            structure.numbers,
            structure.positions,
            rcov=rcov,
            en=en,
            count=self.count,
            pair_weight=self.pair_weight,
            cutoff=self.cutoff,
            lattice=lattice,
            shifts=shifts.shifts,
            periodic=periodic,
        )


def _misses_periodic_axis(table_periodic: Tensor, periodic: Tensor) -> bool:
    """
    Whether ``periodic`` (``(3,)`` or ``(..., 3)``) has a periodic axis the
    shift table was not built for. Always ``False`` under ``vmap`` or
    ``torch.compile``, which do not allow data-dependent control flow.
    """
    if is_compiling() or is_functorch_tensor(periodic):
        return False
    return bool((periodic & ~table_periodic).any())


def _cn_dense_mol(
    numbers: Tensor,
    positions: Tensor,
    *,
    rcov: Tensor,
    en: Tensor | None,
    count: CountingFunction,
    pair_weight: PairWeightFunction | None,
    cutoff: float,
) -> Tensor:
    """
    Dense, all-pairs coordination number, batched over a leading ``(...)``
    dimension.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers, shape ``(..., nat)``. ``0`` marks batch padding.
    positions : Tensor
        Cartesian coordinates, shape ``(..., nat, 3)``.
    rcov, en : Tensor | None
        Resolved per-element tables; ``en`` is ``None`` unless
        ``pair_weight`` is set.
    count : CountingFunction
        Pair counting function.
    pair_weight : PairWeightFunction | None
        See :meth:`CNModel.__call__`.
    cutoff : float
        Real-space cutoff.

    Returns
    -------
    Tensor
        Coordination numbers, shape ``(..., nat)``.
    """
    dd: DD = {"device": positions.device, "dtype": positions.dtype}

    pair_mask = real_pairs(numbers, mask_diagonal=True)
    species = _species(numbers)
    rcov_species = rcov[species]
    rcov_pair_sum = rcov_species.unsqueeze(-1) + rcov_species.unsqueeze(-2)

    eps = torch.tensor(torch.finfo(positions.dtype).eps, **dd)
    distance = torch.where(
        pair_mask, storch.cdist(positions, positions, p=2), eps
    )

    counts = count(distance, rcov_pair_sum)

    if pair_weight is not None:
        assert en is not None
        en_species = en[species]
        en_i = en_species.unsqueeze(-1)  # varies with the row index i
        en_j = en_species.unsqueeze(-2)  # varies with the column index j
        counts = pair_weight(en_i, en_j) * counts

    within_cutoff = pair_mask & (distance <= cutoff)
    # `0.0` as a plain Python scalar, not `zeros_like`: `torch.where`
    # already broadcasts a scalar `other`, so materialising a full
    # replacement array here is only an extra allocate-and-fill pass over
    # memory for a constant every element already shares. Same reasoning
    # applies everywhere else in this module `torch.where` is given a
    # plain-scalar fill value.
    counts = torch.where(within_cutoff, counts, 0.0)
    return counts.sum(-1)


def _cn_dense_per(
    numbers: Tensor,
    positions: Tensor,
    *,
    rcov: Tensor,
    en: Tensor | None,
    count: CountingFunction,
    pair_weight: PairWeightFunction | None,
    cutoff: float,
    lattice: Tensor,
    shifts: Tensor,
    periodic: Tensor,
) -> Tensor:
    """
    Dense periodic coordination number: every real pair evaluated at
    every translation in ``shifts``, masked at ``cutoff``. Batched over a
    leading ``(...)`` dimension exactly like :func:`_cn_dense_mol` batches
    the plain molecular path -- ``...`` empty is the
    single-system case, a real leading batch dimension is
    :meth:`CNModel.__call__`/:meth:`CNModel.with_precomputed_shifts`'s
    batched path, and no other code path is needed for either: every
    intermediate below is built with
    ``Ellipsis``-based broadcasting (``tensor[..., :, None, :]``-style
    indexing, or plain trailing ``unsqueeze(-1)``/``unsqueeze(-2)``),
    which inserts new axes counted from the *end* of the shape and so
    needs no branch on how many leading batch dimensions ``numbers``/
    ``positions`` actually have.

    ``shifts`` is built ahead of this call by
    :func:`tad_mctc.neighbor.images.build_periodic_shifts` (single
    system) or :func:`~tad_mctc.neighbor.images.
    build_shared_periodic_shifts` (a batch, one table shared across it,
    sized to the batch's most demanding lattice -- decision 5 of this
    feature's spec), under ``torch.no_grad()`` and outside any traced/
    transformed region, so that consumption here is pure, fixed-shape
    tensor algebra in ``positions`` and ``lattice`` (``shifts`` itself is
    a no-grad integer buffer): it stays differentiable to any order,
    jacrev-friendly, and ``vmap``-friendly over ``lattice`` or
    ``positions`` as long as the same ``shifts`` table covers every batch
    element. A batched system needing fewer rings than a shared table
    just has its extra shift entries masked out -- by its own
    ``periodic`` mask along an axis that is not periodic for it, and by
    the ordinary ``distance_squared <= cutoff**2`` check otherwise --
    same as atom-count padding (``numbers == 0``, via ``real_pairs``, exactly like the
    molecular path).

    An atom pairing with its own periodic image (``i == j`` at a
    non-zero shift) is a real pair; only the true self-pair (``i == j``
    at the zero shift) is excluded.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers, shape ``(..., nat)``. ``0`` marks batch padding.
    positions : Tensor
        Cartesian coordinates, shape ``(..., nat, 3)``.
    rcov, en : Tensor | None
        Resolved per-element tables; ``en`` is ``None`` unless
        ``pair_weight`` is set.
    count : CountingFunction
        Pair counting function.
    pair_weight : PairWeightFunction | None
        See :meth:`CNModel._periodic`.
    cutoff : float
        Real-space cutoff.
    lattice : Tensor
        Lattice vectors as rows, shape ``(..., 3, 3)``, in Bohr.
    shifts : Tensor
        Integer lattice-translation shifts, shape ``(n_shift, 3)`` --
        shared by every system in ``(...)``, never batched itself.
    periodic : Tensor
        Boolean mask, shape ``(3,)`` or ``(..., 3)``, which axes are
        periodic **per system**. Used to fold ``positions`` into the
        primary cell before summing (see :func:`tad_mctc.neighbor.
        images.wrap_to_central_cell`) and to drop every shift that
        translates along an axis that is not periodic for that system.
        The actual differentiable pair math always uses the caller's
        original, un-wrapped ``positions``, so neither use can affect
        gradients with respect to them. Deliberately not a shift table's
        own (possibly batch-reduced) mask: wrapping along, or
        translating along, an axis that is not actually periodic *for a
        given system* silently changes that system's physics (see
        ``test_dense_periodic_periodic_mask_leaves_slab_axis_
        unwrapped`` and ``test_batch_matches_single_for_mixed_
        periodicity``), so every caller passes each system's own mask
        here.

    Returns
    -------
    Tensor
        Coordination numbers, shape ``(..., nat)``.
    """
    species = _species(numbers)
    rcov_species = rcov[species]  # (..., nat)
    rcov_pair_sum = rcov_species.unsqueeze(-1) + rcov_species.unsqueeze(-2)
    rcov_pair_sum = rcov_pair_sum.unsqueeze(-1)  # (..., nat, nat, 1)

    # `shifts` only covers a cutoff sphere anchored at the primary cell
    # (see `build_periodic_shifts`), so it is only valid once every atom
    # lies inside that cell -- exactly the invariant mctc-lib's own
    # `wrap_to_central_cell` establishes for its callers. `wrap_to_
    # central_cell` runs under `no_grad()` and returns the integer whole-
    # cell offset it applied to each atom (`cell_shift`); folding that
    # offset into each pair's shift, rather than using the wrapped
    # positions directly in the distance formula below, is what keeps
    # this differentiable in the caller's original `positions` (and in
    # `lattice`) despite the wrap. `periodic.unsqueeze(-2)` gives it an
    # explicit atom axis: without it, a batched `(..., 3)` mask would
    # right-align against `cell_shift`'s `(..., nat, 3)` on the *nat*
    # axis instead of broadcasting over it, wrongly conflating the two
    # whenever a batch dimension happens to equal `nat`.
    _, cell_shift = wrap_to_central_cell(
        positions, lattice, periodic.unsqueeze(-2)
    )  # (..., nat, 3)

    # `i == j` cancels `cell_shift[..., j, :] - cell_shift[..., i, :]`
    # exactly, so the zero-shift self-pair exclusion below (on the raw
    # `shifts`, before this correction) is unaffected by the wrap.
    # `shifts` (`(n_shift, 3)`, no batch dimension of its own) broadcasts
    # directly against the `(..., nat, 1, 1, 3)`/`(..., 1, nat, 1, 3)`
    # terms below without any reshaping: ordinary broadcasting right-
    # aligns trailing dimensions and implicitly left-pads the shorter
    # shape with size-1 axes, which is exactly "shared across every
    # batch dimension and both atom axes."
    #
    # (..., nat, nat, n_shift, 3)
    effective_shift = (
        shifts
        + cell_shift[..., None, :, None, :]  # varies with j
        - cell_shift[..., :, None, None, :]  # varies with i
    )

    # `lattice[..., None, None, :, :]` gives it two extra singleton
    # "matrix batch" axes (for the `i`/`j` atom axes) before the trailing
    # `(3, 3)`, so plain `@` broadcasts its own `(...)` batch dimensions
    # against `effective_shift`'s `(..., nat, nat)` ones correctly. A
    # bare `effective_shift @ lattice` would instead right-align
    # `lattice`'s `(...)` batch shape against `effective_shift`'s
    # `(..., nat, nat)` one, silently computing the wrong contraction
    # whenever a batch dimension happened to equal `nat`.
    translation = effective_shift.to(dtype=positions.dtype) @ (
        lattice[..., None, None, :, :]
    )  # (..., nat, nat, n_shift, 3)

    position_i = positions[..., :, None, None, :]  # (..., nat, 1, 1, 3)
    position_j = positions[..., None, :, None, :]  # (..., 1, nat, 1, 3)
    difference = position_j + translation - position_i
    distance_squared = (difference * difference).sum(
        -1
    )  # (..., nat, nat, n_shift)

    pair_mask = real_pairs(numbers, mask_diagonal=False)  # (..., nat, nat)
    is_self_pair = torch.eye(
        numbers.shape[-1], dtype=torch.bool, device=numbers.device
    )
    is_zero_shift = (shifts == 0).all(-1)  # (n_shift,)
    # Neither operand here has a `(...)` batch dimension of its own, so
    # `exclude` (`(nat, nat, n_shift)`) is shared, unbatched, and simply
    # broadcasts against the batched `within_cutoff` below.
    exclude = is_self_pair.unsqueeze(-1) & is_zero_shift

    # A batch shares one shift table, which may translate along an axis
    # that is periodic for another system but not for this one. Such an
    # image does not exist, and the cutoff check alone does not remove
    # it: a non-periodic axis may carry a short placeholder lattice
    # vector, so its images can land well inside the cutoff.
    moves_along_open_axis = (shifts != 0) & ~periodic.unsqueeze(-2)
    is_real_image = ~moves_along_open_axis.any(-1)  # (..., n_shift)
    is_real_image = is_real_image[..., None, None, :]  # (..., 1, 1, n_shift)

    within_cutoff = (
        pair_mask.unsqueeze(-1)
        & ~exclude
        & is_real_image
        & (distance_squared <= cutoff**2)
    )

    # Mask before the square root, not after: an unmasked excluded entry
    # is atom `i` minus itself at the zero shift, i.e. `distance_squared
    # == 0`, and `sqrt` at exactly zero has an infinite derivative. `1.0`
    # is a plain Python scalar for the same reason as `_cn_dense_mol`'s
    # `torch.where(..., 0.0)` above.
    distance_squared = torch.where(within_cutoff, distance_squared, 1.0)
    distance = distance_squared.sqrt()

    counts = count(distance, rcov_pair_sum)

    if pair_weight is not None:
        assert en is not None
        en_species = en[species]  # (..., nat)
        en_i = en_species[..., :, None, None]  # varies with i
        en_j = en_species[..., None, :, None]  # varies with j
        counts = pair_weight(en_i, en_j) * counts

    # `0.0` as a plain Python scalar, same reasoning as `_cn_dense_mol`'s
    # `torch.where(..., 0.0)` above.
    counts = torch.where(within_cutoff, counts, 0.0)
    return counts.sum(dim=(-2, -1))


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
