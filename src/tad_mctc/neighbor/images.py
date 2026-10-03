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
Neighbour search: periodic images
===================================

Turns a lattice and a cutoff into the integer image shifts within reach
of the cutoff (:func:`build_periodic_shifts`). The dense periodic
coordination number sums over them directly. The neighbour search of
:mod:`.list` replicates the atoms of the cell at these shifts into a
*ghost pool*, plain Cartesian data that its tile search handles
unchanged -- the whole point of the ghost-pool route over the
modular-wrapping alternative used by LASP-D3 and NVIDIA's
``nvalchemiops`` is that nothing downstream has to learn about
periodicity at all.

Two, independently-sourced formulas answer the same question -- how many
image rings are needed along each lattice axis to cover a cutoff sphere
-- and both are kept, rather than one replacing the other:

:func:`count_image_rings_cp2k` is a direct port of CP2K's
``nnp_compute_pbc_copies`` (``cp2k/src/nnp_cell_list.F``): a *half*
interplanar spacing plus a ``floor``.

:func:`count_image_rings_mctclib` is a port of mctc-lib's own
``get_translations`` (``src/mctc/cutoff.f90``): the *full* interplanar
spacing plus a ``ceil``. Empirically tighter (fewer excess rings) than
the CP2K version while remaining just as safe -- see the module's own
test suite for the brute-force comparison -- so :func:`build_periodic_shifts`
uses this one; the CP2K version
stays available on its own for a caller that wants it specifically.

Both share the same per-axis projection -- the lattice vector's component
along the unit normal of the plane spanned by the other two -- which is
what stays correct for a triclinic cell where the three lattice vectors
are not orthogonal.

Both functions here are data-dependent (a lattice determines how many
images exist at all) and are meant to run under ``torch.no_grad()`` as
part of neighbour-list *construction*, exactly like the tile search. The
ghost positions they produce are used only to decide *which* pairs exist;
the differentiable translation term is re-formed from ``positions`` and
the integer ``shift`` at *consumption* time (see ``ncoord/common.py``),
so a derivative with respect to ``lattice`` never has to pass through this
module.

Example
-------
>>> import torch
>>> from tad_mctc.neighbor.images import count_image_rings_mctclib
>>>
>>> lattice = 12.0 * torch.eye(3, dtype=torch.double)
>>> periodic = torch.tensor([True, True, True])
>>> count_image_rings_mctclib(lattice, periodic, cutoff=14.0)
tensor([2, 2, 2])
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from ..autograd import is_functorch_tensor
from ..typing import Tensor
from ._tiles import _integer_box

if TYPE_CHECKING:
    from ..io.structure import Structure

__all__ = [
    "count_image_rings_cp2k",
    "count_image_rings_mctclib",
    "build_periodic_shifts",
    "wrap_to_central_cell",
    "PeriodicShifts",
]


# Guards the `floor` below against a ratio that should land exactly on an
# integer (e.g. `cutoff` a whole multiple of the interplanar spacing) but
# lands one ULP under it instead, which would silently drop the outermost
# ring. Small enough to never promote a genuinely non-integer ratio.
_FLOOR_EPS = 1e-10


# Matches `p_pbc_eps` in mctc-lib's and s-dftd3's own `shift_back_abc`:
# a fractional coordinate that should be exactly 0 or exactly 1 but lands
# an ULP off must not be folded a whole cell in the wrong direction.
_WRAP_EPS = 1e-14


@dataclass(frozen=True, eq=False)
class PeriodicShifts:
    """
    A shift table bundled with the periodic-axis mask and cutoff that
    produced it, so the three can never be supplied out of step with each
    other -- see :func:`build_periodic_shifts`, which constructs one.

    Frozen because this is a value, and ``eq=False`` (matching
    :class:`tad_mctc.ncoord.common.CNModel`'s own precedent) because a
    generated ``__eq__`` would compare ``shifts``/``periodic_axes``
    directly, which raises (a tensor's ``==`` returns another tensor, not
    a ``bool``).

    Parameters
    ----------
    shifts : Tensor
        Integer lattice-translation shifts, shape ``(n_shift, 3)``,
        ``torch.long``.
    periodic_axes : Tensor
        Boolean mask, shape ``(3,)``, which axes the table translates
        along.
    cutoff : float
        Real-space cutoff this table was built to cover, in Bohr. A plain
        Python float, never a traced tensor: a consumer compares it
        against a model's own ``cutoff`` (also a plain float) without
        reading any tensor's value, which is what keeps that comparison
        safe under ``torch.func.vmap`` over a batch of lattices.

    Raises
    ------
    RuntimeError
        ``shifts`` is not an ``(n_shift, 3)`` ``torch.long`` tensor, or
        ``periodic_axes`` is not a ``(3,)`` ``torch.bool`` tensor.
    """

    shifts: Tensor
    periodic_axes: Tensor
    cutoff: float

    def __post_init__(self) -> None:
        if self.shifts.ndim != 2 or self.shifts.shape[-1] != 3:
            raise RuntimeError(
                "`shifts` must be an `(n_shift, 3)` tensor of integer "
                f"lattice translations, but shape is "
                f"{tuple(self.shifts.shape)}."
            )
        if self.shifts.dtype != torch.long:
            raise RuntimeError(
                f"`shifts` must be a `torch.long` tensor, but dtype is "
                f"'{self.shifts.dtype}'."
            )
        if self.periodic_axes.shape != (3,):
            raise RuntimeError(
                "`periodic_axes` must be a `(3,)` tensor, but shape is "
                f"{tuple(self.periodic_axes.shape)}."
            )
        if self.periodic_axes.dtype != torch.bool:
            raise RuntimeError(
                "`periodic_axes` must be a `torch.bool` tensor, but dtype "
                f"is '{self.periodic_axes.dtype}'."
            )

    def check_compatible(self, structure: Structure, cutoff: float) -> None:
        """
        Raise unless this table is compatible with ``structure`` and
        ``cutoff``, i.e. using it cannot silently drop a periodic image.

        Covering more axes than ``structure`` is periodic along is fine:
        a consumer drops the shifts along each system's open axes itself
        (see :func:`build_periodic_shifts`). So is covering more
        rings than ``structure.lattice`` needs at ``cutoff``, but not
        fewer: a table built for a larger cell (before an NPT step
        compressed it, or for another cell of a batch) misses the outer
        images of a smaller one.

        The axis and ring comparisons read the values of
        ``structure.periodic`` and ``structure.lattice``, so they are
        skipped under ``torch.compile``, ``vmap`` and ``jacrev``. Call
        this once eagerly before transforming an evaluation.

        Parameters
        ----------
        structure : Structure
            The system(s) the table is about to be used with.
        cutoff : float
            Real-space cutoff of the consumer, e.g. a model's ``cutoff``.

        Raises
        ------
        ValueError
            ``structure.lattice`` is ``None``; ``self.cutoff`` is smaller
            than ``cutoff``; the table misses a periodic axis of
            ``structure``; or it holds fewer image rings along an axis
            than ``structure.lattice`` needs at ``cutoff``.
        """
        if structure.lattice is None:
            raise ValueError(
                "`structure.lattice` is `None`; a `Structure` with no "
                "lattice has nothing periodic to evaluate."
            )

        if self.cutoff < cutoff:
            raise ValueError(
                f"`shifts.cutoff` ({self.cutoff}) is smaller than the "
                f"consumer's cutoff ({cutoff}); the table would silently "
                "drop periodic images. Build it with "
                "`build_periodic_shifts(lattice, periodic, cutoff=...)` "
                "at at least that cutoff."
            )

        # `Structure` fills in a mask whenever it has a lattice.
        periodic = structure.periodic
        assert periodic is not None
        if is_functorch_tensor(periodic):
            return

        table_axes = self.periodic_axes.to(periodic.device)
        if (periodic & ~table_axes).any():
            raise ValueError(
                f"`shifts.periodic_axes` ({table_axes.tolist()}) does not "
                "cover every periodic axis of the structure "
                f"({periodic.tolist()}); the images along the missing axes "
                "would be silently dropped. Build the table with the "
                "structure's own `periodic` mask."
            )

        lattice = structure.lattice
        if is_functorch_tensor(lattice):
            return

        # The table spans `-rings..rings` along each axis (see
        # `build_periodic_shifts`). Every cell of a batch must fit in it.
        needed = count_image_rings_mctclib(lattice, periodic, cutoff)
        needed = needed.reshape(-1, 3).amax(0)
        table_rings = self.shifts.abs().amax(0).to(needed.device)
        if (table_rings < needed).any():
            raise ValueError(
                f"`shifts` reaches {table_rings.tolist()} image rings along "
                f"each axis, but the structure's lattice needs "
                f"{needed.tolist()} at a cutoff of {cutoff}; the outer "
                "images would be silently dropped. Rebuild the table from "
                "this lattice."
            )

    def replace(self, **changes: Any) -> PeriodicShifts:
        """
        Copy these periodic shifts with some fields swapped out, e.g. a
        different ``periodic_axes`` mask.

        Thin wrapper around :func:`dataclasses.replace` so call sites do
        not need their own import of it. Re-runs the shape/dtype checks
        in :meth:`__post_init__` on the new instance, same as the
        constructor.

        Parameters
        ----------
        **changes : Any
            Field name/value pairs to override, e.g.
            ``periodic_axes=mask``.

        Returns
        -------
        PeriodicShifts
            A new, re-validated instance with the given fields replaced.
        """
        return dataclasses.replace(self, **changes)


def wrap_to_central_cell(
    positions: Tensor, lattice: Tensor, periodic: Tensor
) -> tuple[Tensor, Tensor]:
    """
    Fold every atom into the central cell along each periodic axis.

    Port of ``wrap_to_central_cell`` (``mctc-lib/src/mctc/cutoff.f90``,
    and the identical routine in ``s-dftd3/src/dftd3/utils.f90``), which
    s-dftd3 applies to every structure entering its API before any energy
    evaluation. The neighbour search (:mod:`.list`) needs it for the same
    reason: its ghost pool reaches a fixed number of image rings out from the
    cell, sized from ``lattice`` and the cutoff alone, so a pair between
    atoms written more than that many cells apart would simply not be in
    it -- the ordinary case for an unwrapped MD trajectory.

    Unlike the Fortran original, which folds all three components as soon
    as *any* axis is periodic, this wraps each axis independently: a slab
    must keep its non-periodic axis untouched.

    The integer cell offset applied is returned alongside the coordinates
    rather than discarded, because ``positions`` here is the caller's
    tensor, not a structure this function owns: a consumer reconstructs a
    pair vector from the caller's *original* coordinates plus the stored
    integer shift (see :mod:`tad_mctc.ncoord.common`), so whatever the
    wrap moved has to be added back into that shift.

    Parameters
    ----------
    positions : Tensor
        Cartesian coordinates, shape ``(nat, 3)``.
    lattice : Tensor
        Lattice vectors as rows, shape ``(3, 3)``, in Bohr.
    periodic : Tensor
        Boolean mask, shape ``(3,)``, which axes are periodic.

    Returns
    -------
    tuple[Tensor, Tensor]
        ``wrapped``, shape ``(nat, 3)``: coordinates whose fractional
        components lie in ``[0, 1)`` along every periodic axis.

        ``cell_shift``, shape ``(nat, 3)``, ``torch.long``: the integer
        cell offset applied, so that ``wrapped == positions +
        cell_shift.to(positions.dtype) @ lattice`` exactly.

    Example
    -------
    >>> import torch
    >>> from tad_mctc.neighbor.images import wrap_to_central_cell
    >>>
    >>> lattice = 10.0 * torch.eye(3, dtype=torch.double)
    >>> periodic = torch.tensor([True, True, True])
    >>> positions = torch.tensor([[21.0, -5.0, 0.0]], dtype=torch.double)
    >>> wrapped, cell_shift = wrap_to_central_cell(
    ...     positions, lattice, periodic
    ... )
    >>> wrapped
    tensor([[1., 5., 0.]], dtype=torch.float64)
    >>> cell_shift
    tensor([[-2,  1,  0]])
    """
    # The integer shift is piecewise constant, so none of this needs a
    # graph.
    with torch.no_grad():
        fractional = positions @ torch.linalg.inv(lattice)

        # `-floor(frac + eps)` reproduces the Fortran `shift_back_abc`
        # branch-for-branch, including its epsilon guard: a coordinate a
        # hair under an integer folds with that integer rather than a cell
        # below it.
        cell_shift = -torch.floor(fractional + _WRAP_EPS).long()
        cell_shift = torch.where(
            periodic, cell_shift, torch.zeros_like(cell_shift)
        )

    # Outside `no_grad`, so `wrapped` is differentiable like the plain
    # `positions + shift @ lattice` it is: the identity with respect to
    # `positions`, `cell_shift` with respect to `lattice`. Reverse mode
    # (`jacrev`) then agrees with forward mode (`jacfwd`), which ignores
    # `no_grad` altogether.
    wrapped = positions + cell_shift.to(positions.dtype) @ lattice
    return wrapped, cell_shift


def _validate_lattice_periodic(lattice: Tensor, periodic: Tensor) -> None:
    """Shape/non-zero-volume checks shared by both ring-count formulas.
    A left-handed cell (negative determinant) is fine: the interplanar
    spacing takes the absolute value, as in mctc-lib's `get_translations`."""
    if lattice.shape[-2:] != (3, 3):
        raise RuntimeError(
            "Lattice vectors must be a '(..., 3, 3)' tensor, but shape is "
            f"'{tuple(lattice.shape)}'."
        )
    if periodic.shape != (3,) and periodic.shape != lattice.shape[:-1]:
        raise RuntimeError(
            "Periodicity mask must be a '(3,)' or '(..., 3)' tensor "
            f"broadcasting against the lattice batch, but shape is "
            f"'{tuple(periodic.shape)}'."
        )

    volume = torch.linalg.det(lattice)
    if bool((volume == 0.0).any()):
        raise RuntimeError(
            "Lattice vectors must span a cell of non-zero volume, but the "
            f"determinant(s) are '{volume}'. Give each non-periodic axis a "
            "placeholder vector instead of a zero row."
        )


def _axis_spacing(lattice: Tensor) -> Tensor:
    """
    Full interplanar spacing along each lattice axis: axis ``a``'s
    component along the unit normal of the plane spanned by ``b`` and
    ``c``, and cyclically. ``b`` and ``c`` project to zero onto that
    normal by construction, so this is exactly the spacing between the
    cell's own face and its periodic image, whether or not the three
    lattice vectors are orthogonal.

    Shared by both :func:`count_image_rings_cp2k` (which halves it) and
    :func:`count_image_rings_mctclib` (which uses it as-is).
    """
    a, b, c = lattice[..., 0, :], lattice[..., 1, :], lattice[..., 2, :]

    normal_bc = _unit_normal(b, c)
    normal_ac = _unit_normal(a, c)
    normal_ab = _unit_normal(a, b)

    return torch.stack(
        [
            (a * normal_bc).sum(-1).abs(),
            (b * normal_ac).sum(-1).abs(),
            (c * normal_ab).sum(-1).abs(),
        ],
        dim=-1,
    )


def _unit_normal(u: Tensor, v: Tensor) -> Tensor:
    """Unit vector along ``u x v``, batched over a leading ``(...)``
    dimension. Never degenerate for a valid cell."""
    normal = torch.linalg.cross(u, v, dim=-1)
    return normal / normal.norm(dim=-1, keepdim=True)


@torch.no_grad()
def count_image_rings_cp2k(
    lattice: Tensor, periodic: Tensor, cutoff: float
) -> Tensor:
    """
    Number of periodic image rings needed along each lattice axis to cover
    a sphere of radius ``cutoff`` around any point of the primary cell.

    Ported from CP2K's ``nnp_compute_pbc_copies``: half the interplanar
    spacing (:func:`_axis_spacing`), floored. See
    :func:`count_image_rings_mctclib` for an independently-sourced
    formula answering the same question, empirically tighter at the same
    safety margin -- :func:`build_periodic_shifts` uses that one; this one
    stays
    available on its own.

    A non-periodic axis always gets zero rings, regardless of ``cutoff``.
    A periodic axis can still come back zero here when ``cutoff`` does
    not reach past the cell's own boundary in that direction --
    :func:`build_periodic_shifts` raises that to one ring, because an atom
    sitting right at a periodic boundary can have a real neighbour
    arbitrarily close on the other side of it, however small ``cutoff``
    is.

    Accepts a batch of lattices, ``(..., 3, 3)`` (the unbatched ``(3, 3)``
    case still works, as ``...`` is then empty): each lattice in the batch
    gets its own ring count, independent of every other one -- this
    function does not combine them into a single shared count (see
    :func:`build_periodic_shifts` for that).

    Parameters
    ----------
    lattice : Tensor
        Lattice vectors as rows, shape ``(..., 3, 3)``, in Bohr.
    periodic : Tensor
        Boolean mask, shape ``(..., 3)`` or ``(3,)`` (broadcasting against
        the batch), which axes are periodic.
    cutoff : float
        Sphere radius to cover, in Bohr.

    Returns
    -------
    Tensor
        Number of image rings per axis, shape ``(..., 3)``, ``torch.long``.

    Example
    -------
    >>> import torch
    >>> from tad_mctc.neighbor.images import count_image_rings_cp2k
    >>>
    >>> lattice = 12.0 * torch.eye(3, dtype=torch.double)
    >>> periodic = torch.tensor([True, True, True])
    >>> count_image_rings_cp2k(lattice, periodic, cutoff=5.0)
    tensor([0, 0, 0])
    """
    _validate_lattice_periodic(lattice, periodic)

    half_spacing = 0.5 * _axis_spacing(lattice)
    rings = torch.floor(cutoff / half_spacing + _FLOOR_EPS).long()
    return torch.where(periodic, rings, torch.zeros_like(rings))


@torch.no_grad()
def count_image_rings_mctclib(
    lattice: Tensor, periodic: Tensor, cutoff: float
) -> Tensor:
    """
    Number of periodic image rings needed along each lattice axis to cover
    a sphere of radius ``cutoff`` around any point of the primary cell.

    Ported from mctc-lib's ``get_translations`` (``src/mctc/cutoff.f90``):
    the full interplanar spacing (:func:`_axis_spacing`), ``ceil``ed --
    as opposed to :func:`count_image_rings_cp2k`'s half-spacing-plus-
    ``floor``. Empirically tighter (fewer excess rings, so a smaller
    :func:`build_periodic_shifts` table) at the same safety margin: never
    observed to under-cover across randomized cubic, orthorhombic and
    triclinic lattices, including cutoffs set to exact spacing multiples.
    This is the formula :func:`build_periodic_shifts` uses.

    A non-periodic axis always gets zero rings, regardless of ``cutoff``.
    Unlike the CP2K version, a periodic axis only comes back zero here
    when ``cutoff`` is exactly zero -- ``ceil`` of any positive ratio is
    already at least one.

    Accepts a batch of lattices, ``(..., 3, 3)`` (the unbatched ``(3, 3)``
    case still works, as ``...`` is then empty): each lattice in the batch
    gets its own ring count, independent of every other one -- this
    function does not combine them into a single shared count (see
    :func:`build_periodic_shifts` for that).

    Parameters
    ----------
    lattice : Tensor
        Lattice vectors as rows, shape ``(..., 3, 3)``, in Bohr.
    periodic : Tensor
        Boolean mask, shape ``(..., 3)`` or ``(3,)`` (broadcasting against
        the batch), which axes are periodic.
    cutoff : float
        Sphere radius to cover, in Bohr.

    Returns
    -------
    Tensor
        Number of image rings per axis, shape ``(..., 3)``, ``torch.long``.

    Example
    -------
    >>> import torch
    >>> from tad_mctc.neighbor.images import count_image_rings_mctclib
    >>>
    >>> lattice = 12.0 * torch.eye(3, dtype=torch.double)
    >>> periodic = torch.tensor([True, True, True])
    >>> count_image_rings_mctclib(lattice, periodic, cutoff=5.0)
    tensor([1, 1, 1])
    """
    _validate_lattice_periodic(lattice, periodic)

    spacing = _axis_spacing(lattice)
    rings = torch.ceil(cutoff / spacing).long()
    return torch.where(periodic, rings, torch.zeros_like(rings))


def _image_rings(lattice: Tensor, periodic: Tensor, cutoff: float) -> Tensor:
    """
    The image rings of each lattice along each axis: those of
    :func:`count_image_rings_mctclib`, raised to at least one ring along a
    periodic axis. An atom sitting right at a periodic boundary can have a
    real neighbour arbitrarily close on the other side of it, however
    small ``cutoff`` is.

    Parameters
    ----------
    lattice : Tensor
        Lattice vectors as rows, shape ``(..., 3, 3)``, in Bohr.
    periodic : Tensor
        Boolean mask, shape ``(..., 3)`` or ``(3,)``, which axes are
        periodic.
    cutoff : float
        Sphere radius to cover, in Bohr.

    Returns
    -------
    Tensor
        Ring counts, shape ``(..., 3)``.
    """
    rings = count_image_rings_mctclib(lattice, periodic, cutoff)
    return torch.maximum(rings, periodic.long())


@torch.no_grad()
def build_periodic_shifts(
    lattice: Tensor, periodic: Tensor, cutoff: float
) -> PeriodicShifts:
    """
    Integer lattice-translation shifts covering every periodic image
    within reach of ``cutoff``, for one cell or a batch of cells.

    Every combination of per-axis rings from :func:`_image_rings`,
    including the zero shift for the primary cell itself. For a batch,
    one table covers every cell: it reaches the per-axis **maximum** ring
    count of the batch. A system that needs fewer rings just has its
    extra shift entries masked out downstream by the ordinary
    ``distance_squared <= cutoff**2`` check -- over-covering a periodic
    axis is free, under-covering is what silently drops real neighbours.
    This lets one dense-periodic batch evaluation (see
    :mod:`tad_mctc.ncoord.common`) use a single, fixed-shape ``shifts``
    tensor for every system in the batch, no ``torch.func.vmap`` required.

    The table is bundled with the ``periodic`` mask and ``cutoff`` that
    produced it (:class:`PeriodicShifts`) so the three can never be
    supplied out of step with each other. For a batch, its
    ``periodic_axes`` is the batch's per-axis ``.any()`` -- the union of
    what any system needs. A consumer must apply each system's own
    ``periodic`` mask, not this reduced one, both to fold positions into
    the central cell and to drop shifts along an axis that is not periodic
    for that system. The cutoff check does not remove those shifts: a
    non-periodic axis may carry a short placeholder lattice vector, which
    puts its images well inside the cutoff (see
    ``test_batch_matches_single_for_mixed_periodicity`` in
    ``test/test_ncoord/test_dense_periodic.py``).

    Parameters
    ----------
    lattice : Tensor
        Lattice vectors as rows, shape ``(3, 3)`` or ``(..., 3, 3)``, in
        Bohr.
    periodic : Tensor
        Boolean mask, shape ``(3,)`` or ``(..., 3)``, which axes are
        periodic.
    cutoff : float
        Sphere radius to cover, in Bohr, for every lattice in the batch.

    Returns
    -------
    PeriodicShifts
        ``shifts``, shape ``(n_shift, 3)``, ``torch.long``, bundled with
        the (batch-reduced) ``periodic`` mask and this call's ``cutoff``.

    Example
    -------
    >>> import torch
    >>> from tad_mctc.neighbor.images import build_periodic_shifts
    >>>
    >>> lattice = 12.0 * torch.eye(3, dtype=torch.double)
    >>> periodic = torch.tensor([True, True, True])
    >>> bundle = build_periodic_shifts(lattice, periodic, cutoff=1.0)
    >>> bundle.shifts.shape
    torch.Size([27, 3])
    >>> batch = torch.stack([lattice, 2.0 * lattice])
    >>> build_periodic_shifts(batch, periodic, cutoff=20.0).shifts.shape
    torch.Size([125, 3])
    """
    rings = _image_rings(lattice, periodic, cutoff)
    max_rings = rings.reshape(-1, 3).amax(0)
    periodic_axes = periodic.reshape(-1, 3).any(0)
    return PeriodicShifts(
        shifts=_integer_box(max_rings),
        periodic_axes=periodic_axes,
        cutoff=cutoff,
    )
