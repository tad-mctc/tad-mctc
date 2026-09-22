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
Test the periodic ghost pool: `count_image_rings_cp2k`,
`count_image_rings_mctclib`, and `build_ghost_pool`.

The unit tests check each ring-count formula against an independent,
literal port of its own source (CP2K's while-loop, mctc-lib's Fortran
subroutine), plus a brute-force comparison between the two, and the ghost
pool's bookkeeping (shape, owner, shift) directly.
"""

from __future__ import annotations

import math

import pytest
import torch

from tad_mctc.neighbor.images import (
    PeriodicShifts,
    build_ghost_pool,
    build_periodic_shifts,
    build_shared_periodic_shifts,
    count_image_rings_cp2k,
    count_image_rings_mctclib,
    wrap_to_central_cell,
)
from tad_mctc.typing import DD as DDType
from tad_mctc.typing import Tensor

from ..conftest import DEVICE

DD: DDType = {"device": DEVICE, "dtype": torch.double}


##############################################################################
# count_image_rings_cp2k / count_image_rings_mctclib
##############################################################################


def _cross(u: list[float], v: list[float]) -> list[float]:
    return [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ]


def _unit(u: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in u))
    return [x / n for x in u]


def _dot(u: list[float], v: list[float]) -> float:
    return sum(x * y for x, y in zip(u, v))


def _reference_pbc_copies(
    lattice: Tensor, periodic: Tensor, cutoff: float
) -> list[int]:
    """
    Literal, independent port of CP2K's while-loop
    ``nnp_compute_pbc_copies`` (``cp2k/src/nnp_cell_list.F``), used as
    ground truth for :func:`count_image_rings_cp2k`'s closed-form
    (``floor``-based) version of the same formula.
    """
    a, b, c = lattice[0].tolist(), lattice[1].tolist(), lattice[2].tolist()

    bxc, axc, axb = _cross(b, c), _cross(a, c), _cross(a, b)
    proja = 0.5 * abs(_dot(a, _unit(bxc)))
    projb = 0.5 * abs(_dot(b, _unit(axc)))
    projc = 0.5 * abs(_dot(c, _unit(axb)))

    copies = [0, 0, 0]
    for axis, proj in enumerate((proja, projb, projc)):
        while (copies[axis] + 1) * proj <= cutoff:
            copies[axis] += 1

    periodic_list = periodic.tolist()
    return [n if p else 0 for n, p in zip(copies, periodic_list)]


def _reference_get_translations(
    lattice: Tensor, periodic: Tensor, cutoff: float
) -> list[int]:
    """
    Literal, independent port of mctc-lib's ``get_translations``
    (``src/mctc/cutoff.f90``), used as ground truth for
    :func:`count_image_rings_mctclib`'s (full-spacing, ``ceil``-based)
    version of the same formula.
    """
    a, b, c = lattice[0].tolist(), lattice[1].tolist(), lattice[2].tolist()

    bxc, axc, axb = _cross(b, c), _cross(a, c), _cross(a, b)
    proja = abs(_dot(a, _unit(bxc)))
    projb = abs(_dot(b, _unit(axc)))
    projc = abs(_dot(c, _unit(axb)))

    reps = [math.ceil(cutoff / proj) for proj in (proja, projb, projc)]

    periodic_list = periodic.tolist()
    return [n if p else 0 for n, p in zip(reps, periodic_list)]


_CUBIC = 12.0 * torch.eye(3, dtype=torch.double)
_SMALL_CUBIC = 6.0 * torch.eye(3, dtype=torch.double)
_TRICLINIC = torch.tensor(
    [[6.0, 0.0, 0.0], [1.5, 5.5, 0.0], [0.8, 1.2, 5.0]], dtype=torch.double
)
_SLAB = torch.tensor(
    [[6.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 40.0]], dtype=torch.double
)


@pytest.mark.parametrize(
    "lattice, periodic, cutoff",
    [
        (_CUBIC, [True, True, True], 14.0),
        (_CUBIC, [True, True, True], 5.0),
        (_SMALL_CUBIC, [True, True, True], 10.0),
        (_TRICLINIC, [True, True, True], 9.0),
        (_SLAB, [True, True, False], 8.0),
    ],
)
def test_count_image_rings_cp2k_matches_reference_loop(
    lattice: Tensor, periodic: list[bool], cutoff: float
) -> None:
    """The vectorised, `floor`-based ring count agrees exactly with a
    literal port of CP2K's own while-loop version of the same formula."""
    periodic_t = torch.tensor(periodic, device=DEVICE)
    rings = count_image_rings_cp2k(lattice.to(DEVICE), periodic_t, cutoff)
    expected = _reference_pbc_copies(lattice, periodic_t, cutoff)
    assert rings.tolist() == expected


@pytest.mark.parametrize(
    "lattice, periodic, cutoff",
    [
        (_CUBIC, [True, True, True], 14.0),
        (_CUBIC, [True, True, True], 5.0),
        (_SMALL_CUBIC, [True, True, True], 10.0),
        (_TRICLINIC, [True, True, True], 9.0),
        (_SLAB, [True, True, False], 8.0),
    ],
)
def test_count_image_rings_mctclib_matches_reference_loop(
    lattice: Tensor, periodic: list[bool], cutoff: float
) -> None:
    """The vectorised, `ceil`-based ring count agrees exactly with a
    literal port of mctc-lib's own `get_translations` formula."""
    periodic_t = torch.tensor(periodic, device=DEVICE)
    rings = count_image_rings_mctclib(lattice.to(DEVICE), periodic_t, cutoff)
    expected = _reference_get_translations(lattice, periodic_t, cutoff)
    assert rings.tolist() == expected


@pytest.mark.parametrize(
    "lattice, periodic, cutoff",
    [
        (_CUBIC, [True, True, True], 6.0),
        (_CUBIC, [True, True, True], 12.0),
        (_CUBIC, [True, True, True], 18.0),
        (_CUBIC, [True, True, True], 25.0),
        (_SMALL_CUBIC, [True, True, True], 10.0),
        (_TRICLINIC, [True, True, True], 9.0),
        (_SLAB, [True, True, False], 8.0),
    ],
)
def test_count_image_rings_mctclib_never_looser_than_cp2k(
    lattice: Tensor, periodic: list[bool], cutoff: float
) -> None:
    """mctc-lib's formula is the tighter (or equal) one everywhere it's
    been checked -- never more rings than CP2K's, which is the whole
    reason `build_periodic_shifts` prefers it."""
    periodic_t = torch.tensor(periodic, device=DEVICE)
    cp2k = count_image_rings_cp2k(lattice.to(DEVICE), periodic_t, cutoff)
    mctclib = count_image_rings_mctclib(lattice.to(DEVICE), periodic_t, cutoff)
    assert bool((mctclib <= cp2k).all())


def test_count_image_rings_zero_on_nonperiodic_axis() -> None:
    """A non-periodic axis gets zero rings, however large the cutoff, for
    either formula."""
    lattice = _SLAB.to(DEVICE)
    periodic = torch.tensor([True, True, False], device=DEVICE)

    for count_image_rings in (
        count_image_rings_cp2k,
        count_image_rings_mctclib,
    ):
        rings = count_image_rings(lattice, periodic, cutoff=50.0)
        assert rings[2].item() == 0
        assert rings[0].item() > 0
        assert rings[1].item() > 0


def test_count_image_rings_accepts_batched_lattice() -> None:
    """A batch of lattices, `(..., 3, 3)`, gives each one its own ring
    count, matching calling the unbatched function once per lattice, for
    either formula."""
    periodic = torch.tensor([True, True, True], device=DEVICE)
    batch = torch.stack(
        [_CUBIC.to(DEVICE), _SMALL_CUBIC.to(DEVICE), _TRICLINIC.to(DEVICE)]
    )

    for count_image_rings in (
        count_image_rings_cp2k,
        count_image_rings_mctclib,
    ):
        batched_rings = count_image_rings(batch, periodic, cutoff=9.0)
        expected = torch.stack(
            [count_image_rings(lat, periodic, cutoff=9.0) for lat in batch]
        )
        assert torch.equal(batched_rings, expected)


def test_count_image_rings_rejects_degenerate_cell() -> None:
    """A cell with zero volume cannot define an interplanar spacing, so it
    is rejected rather than looping forever or dividing by zero, for
    either formula."""
    lattice = torch.zeros(3, 3, device=DEVICE, dtype=torch.double)
    periodic = torch.tensor([True, True, True], device=DEVICE)

    for count_image_rings in (
        count_image_rings_cp2k,
        count_image_rings_mctclib,
    ):
        with pytest.raises(RuntimeError):
            count_image_rings(lattice, periodic, cutoff=10.0)


def test_count_image_rings_rejects_malformed_lattice_shape() -> None:
    """A lattice that is not `(..., 3, 3)` cannot define interplanar
    spacings, so it is rejected before any determinant is taken."""
    lattice = torch.zeros(3, 2, device=DEVICE, dtype=torch.double)
    periodic = torch.tensor([True, True, True], device=DEVICE)

    for count_image_rings in (
        count_image_rings_cp2k,
        count_image_rings_mctclib,
    ):
        with pytest.raises(RuntimeError):
            count_image_rings(lattice, periodic, cutoff=10.0)


def test_count_image_rings_rejects_malformed_periodic_shape() -> None:
    """`periodic` must broadcast against the lattice batch, either as a
    bare `(3,)` mask or one matching the batch shape exactly."""
    lattice = _TRICLINIC.to(DEVICE)
    periodic = torch.tensor([True, True], device=DEVICE)

    for count_image_rings in (
        count_image_rings_cp2k,
        count_image_rings_mctclib,
    ):
        with pytest.raises(RuntimeError):
            count_image_rings(lattice, periodic, cutoff=10.0)


def test_count_image_rings_left_handed_cell() -> None:
    """Swapping two lattice vectors flips the cell's handedness but not
    its geometry, so the ring counts swap along with the vectors."""
    lattice = _TRICLINIC.to(DEVICE)
    left_handed = lattice[[1, 0, 2]]
    assert torch.linalg.det(left_handed) < 0
    periodic = torch.tensor([True, True, True], device=DEVICE)

    for count_image_rings in (
        count_image_rings_cp2k,
        count_image_rings_mctclib,
    ):
        rings = count_image_rings(lattice, periodic, cutoff=9.0)
        swapped = count_image_rings(left_handed, periodic, cutoff=9.0)
        assert torch.equal(swapped, rings[[1, 0, 2]])


##############################################################################
# build_ghost_pool
##############################################################################


def test_build_ghost_pool_consistency() -> None:
    """Every ghost position equals its owner's position translated by its
    own integer shift through the lattice, exactly."""
    torch.manual_seed(0)
    positions = torch.rand(6, 3, **DD) * 6.0
    lattice = _SMALL_CUBIC.to(DEVICE)
    periodic = torch.tensor([True, True, True], device=DEVICE)

    ghosts, owner, shift = build_ghost_pool(
        positions, lattice, periodic, cutoff=10.0
    )
    expected = positions[owner] + shift.to(positions.dtype) @ lattice
    assert torch.allclose(ghosts, expected, atol=1e-12)


def test_build_ghost_pool_zero_shift_is_primary_copy() -> None:
    """Exactly one ghost per atom carries the zero shift, and it
    reproduces that atom's own position."""
    torch.manual_seed(1)
    nat = 5
    positions = torch.rand(nat, 3, **DD) * 6.0
    lattice = _SMALL_CUBIC.to(DEVICE)
    periodic = torch.tensor([True, True, True], device=DEVICE)

    ghosts, owner, shift = build_ghost_pool(
        positions, lattice, periodic, cutoff=10.0
    )
    is_primary = (shift == 0).all(-1)

    assert int(is_primary.sum()) == nat
    order = owner[is_primary].argsort()
    assert torch.equal(
        owner[is_primary][order], torch.arange(nat, device=DEVICE)
    )
    assert torch.allclose(ghosts[is_primary][order], positions, atol=1e-12)


def test_build_ghost_pool_forces_minimum_one_ring() -> None:
    """Even when the projection formula alone would call for zero rings,
    a periodic axis still gets one: an atom right at a cell boundary can
    have a real neighbour arbitrarily close on the other side of it,
    however small ``cutoff`` is.

    mctc-lib's ``ceil``-based formula (used by :func:`build_ghost_pool` via
    :func:`build_periodic_shifts`) already returns at least one ring for
    any ``cutoff > 0`` on its own -- unlike CP2K's ``floor``-based one, see
    :func:`count_image_rings_cp2k`'s docstring -- so the only case where
    the formula itself calls for zero rings is ``cutoff == 0.0`` exactly.
    """
    positions = torch.tensor([[3.0, 3.0, 3.0]], **DD)
    lattice = _CUBIC.to(DEVICE)
    periodic = torch.tensor([True, True, True], device=DEVICE)

    rings = count_image_rings_mctclib(lattice, periodic, cutoff=0.0)
    assert rings.tolist() == [0, 0, 0]

    _, owner, shift = build_ghost_pool(positions, lattice, periodic, cutoff=0.0)
    assert shift.shape[0] == 27  # 3 ** 3: one forced ring on every axis
    assert bool((owner == 0).all())


##############################################################################
# build_periodic_shifts
##############################################################################


def test_build_periodic_shifts_matches_ghost_pool_shift() -> None:
    """`build_periodic_shifts` is the same shift table `build_ghost_pool`
    derives internally -- the dense periodic coordination-number path
    (`tad_mctc.ncoord.common`) needs that table on its own, without a
    ghost pool of Cartesian positions attached to it."""
    positions = torch.rand(6, 3, **DD)
    lattice = _SMALL_CUBIC.to(DEVICE)
    periodic = torch.tensor([True, True, True], device=DEVICE)

    _, _, pool_shift = build_ghost_pool(
        positions, lattice, periodic, cutoff=10.0
    )
    expected = torch.unique(pool_shift, dim=0)

    bundle = build_periodic_shifts(lattice, periodic, cutoff=10.0)

    assert bundle.shifts.dtype == torch.long
    assert torch.equal(torch.unique(bundle.shifts, dim=0), expected)


def test_build_periodic_shifts_includes_zero_shift() -> None:
    """The primary cell itself (the zero translation) is always one of
    the returned shifts, however small `cutoff` is."""
    lattice = _CUBIC.to(DEVICE)
    periodic = torch.tensor([True, True, True], device=DEVICE)

    bundle = build_periodic_shifts(lattice, periodic, cutoff=1.0)

    zero = torch.zeros(3, dtype=torch.long, device=DEVICE)
    assert bool((bundle.shifts == zero).all(-1).any())


##############################################################################
# build_shared_periodic_shifts
##############################################################################


def test_build_shared_periodic_shifts_matches_elementwise_max() -> None:
    """For a batch of lattices with genuinely different sizes (not just
    uniform scaling), the shared table's ring count in each axis equals
    the max of calling `count_image_rings_mctclib` on each lattice
    individually and taking the elementwise max by hand."""
    periodic = torch.tensor([True, True, True], device=DEVICE)
    lattices = [
        _CUBIC.to(DEVICE),
        _SMALL_CUBIC.to(DEVICE),
        _TRICLINIC.to(DEVICE),
    ]
    batch = torch.stack(lattices)
    cutoff = 9.0

    bundle = build_shared_periodic_shifts(batch, periodic, cutoff)

    per_lattice_rings = torch.stack(
        [count_image_rings_mctclib(lat, periodic, cutoff) for lat in lattices]
    )
    expected_max_rings = per_lattice_rings.amax(dim=0)
    expected_max_rings = torch.maximum(expected_max_rings, periodic.long())

    actual_max_rings = bundle.shifts.amax(dim=0)
    assert torch.equal(actual_max_rings, expected_max_rings)


def test_build_shared_periodic_shifts_covers_the_more_demanding_lattice() -> (
    None
):
    """A batch where one lattice needs strictly more rings than another:
    the shared table still contains every shift the more-demanding
    lattice needs."""
    periodic = torch.tensor([True, True, True], device=DEVICE)
    small = _SMALL_CUBIC.to(DEVICE)  # needs more rings at a fixed cutoff
    large = _CUBIC.to(DEVICE)
    batch = torch.stack([small, large])
    cutoff = 9.0

    shared = build_shared_periodic_shifts(batch, periodic, cutoff)
    demanding = build_periodic_shifts(small, periodic, cutoff)

    shared_set = {tuple(row.tolist()) for row in shared.shifts}
    demanding_set = {tuple(row.tolist()) for row in demanding.shifts}
    assert demanding_set.issubset(shared_set)
    # The small (more demanding) lattice needs strictly more rings than
    # the large one at this cutoff, so the shared table is strictly
    # larger than what the large lattice alone would need.
    assert (
        shared.shifts.shape[0]
        > build_periodic_shifts(large, periodic, cutoff).shifts.shape[0]
    )


##############################################################################
# PeriodicShifts
##############################################################################


def test_periodic_shifts_rejects_malformed_shifts_shape() -> None:
    periodic = torch.tensor([True, True, True], device=DEVICE)
    with pytest.raises(RuntimeError):
        PeriodicShifts(
            shifts=torch.zeros(3, dtype=torch.long, device=DEVICE),
            periodic=periodic,
            cutoff=10.0,
        )


def test_periodic_shifts_rejects_non_long_shifts_dtype() -> None:
    periodic = torch.tensor([True, True, True], device=DEVICE)
    with pytest.raises(RuntimeError):
        PeriodicShifts(
            shifts=torch.zeros(1, 3, dtype=torch.double, device=DEVICE),
            periodic=periodic,
            cutoff=10.0,
        )


def test_periodic_shifts_rejects_malformed_periodic_shape() -> None:
    with pytest.raises(RuntimeError):
        PeriodicShifts(
            shifts=torch.zeros(1, 3, dtype=torch.long, device=DEVICE),
            periodic=torch.tensor([True, True], device=DEVICE),
            cutoff=10.0,
        )


def test_periodic_shifts_rejects_non_bool_periodic_dtype() -> None:
    with pytest.raises(RuntimeError):
        PeriodicShifts(
            shifts=torch.zeros(1, 3, dtype=torch.long, device=DEVICE),
            periodic=torch.tensor([1, 1, 1], device=DEVICE),
            cutoff=10.0,
        )


def test_periodic_shifts_replace_swaps_fields_and_revalidates() -> None:
    """`.replace()` is a thin wrapper around `dataclasses.replace`: it
    both swaps the given fields and re-runs `__post_init__` on the
    result, same as the constructor."""
    bundle = PeriodicShifts(
        shifts=torch.zeros(1, 3, dtype=torch.long, device=DEVICE),
        periodic=torch.tensor([True, True, True], device=DEVICE),
        cutoff=10.0,
    )

    new_periodic = torch.tensor([True, False, True], device=DEVICE)
    replaced = bundle.replace(periodic=new_periodic)

    assert replaced is not bundle
    assert torch.equal(replaced.periodic, new_periodic)
    assert torch.equal(replaced.shifts, bundle.shifts)
    assert replaced.cutoff == bundle.cutoff

    with pytest.raises(RuntimeError):
        bundle.replace(periodic=torch.tensor([True, True], device=DEVICE))


def test_build_periodic_shifts_returns_matching_bundle() -> None:
    """`build_periodic_shifts`'s returned `.shifts`/`.periodic`/`.cutoff`
    match today's inputs/outputs exactly, just wrapped in a bundle."""
    lattice = _SMALL_CUBIC.to(DEVICE)
    periodic = torch.tensor([True, True, True], device=DEVICE)
    cutoff = 10.0

    bundle = build_periodic_shifts(lattice, periodic, cutoff)

    assert isinstance(bundle, PeriodicShifts)
    assert torch.equal(bundle.periodic, periodic)
    assert bundle.cutoff == cutoff

    _, _, pool_shift = build_ghost_pool(
        torch.rand(6, 3, **DD), lattice, periodic, cutoff
    )
    assert torch.equal(
        torch.unique(bundle.shifts, dim=0), torch.unique(pool_shift, dim=0)
    )


##############################################################################
# wrap_to_central_cell
##############################################################################


def test_wrap_to_central_cell_reports_the_offset_it_applied() -> None:
    """The returned integer offset is the whole contract: a caller
    reconstructs pair vectors from its *own* coordinates plus a stored
    shift, so `wrapped == positions + cell_shift @ lattice` has to hold
    exactly, not approximately."""
    lattice = _TRICLINIC.to(DEVICE)
    periodic = torch.tensor([True, True, True], device=DEVICE)

    torch.manual_seed(3)
    fractional = torch.rand(12, 3, **DD) + torch.randint(-4, 5, (12, 3), **DD)
    positions = fractional @ lattice

    wrapped, cell_shift = wrap_to_central_cell(positions, lattice, periodic)

    assert cell_shift.dtype == torch.long
    assert torch.allclose(
        wrapped, positions + cell_shift.to(positions.dtype) @ lattice
    )

    wrapped_fractional = wrapped @ torch.linalg.inv(lattice)
    assert bool((wrapped_fractional >= -1e-12).all())
    assert bool((wrapped_fractional < 1.0).all())


def test_wrap_to_central_cell_leaves_a_non_periodic_axis_alone() -> None:
    """A slab is periodic in two axes only, and folding the third would
    move an atom to a physically different place. This is a deliberate
    departure from the Fortran original, which folds all three components
    as soon as any axis is periodic."""
    lattice = _CUBIC.to(DEVICE)
    periodic = torch.tensor([True, True, False], device=DEVICE)

    # `_CUBIC` has a 12 Bohr edge.
    positions = torch.tensor([[13.0, -3.0, 21.0]], **DD)
    wrapped, cell_shift = wrap_to_central_cell(positions, lattice, periodic)

    assert cell_shift.tolist() == [[-1, 1, 0]]
    assert wrapped.tolist() == [[1.0, 9.0, 21.0]]


def test_wrap_to_central_cell_guards_the_cell_boundary() -> None:
    """An atom whose fractional coordinate should be exactly 0 or exactly
    1 but lands a rounding error off it must not be thrown a whole cell
    the wrong way -- the epsilon guard carried over from the Fortran
    `shift_back_abc`.

    Rounding *down* to just under the far face folds to the near one
    (both are the same point); rounding *below* the near face does not
    fold at all, which is what keeps the two cases from disagreeing.
    """
    edge = 10.0
    lattice = edge * torch.eye(3, **DD)
    periodic = torch.tensor([True, True, True], device=DEVICE)

    positions = torch.tensor(
        [
            [edge, 0.0, 0.0],  # fractional exactly 1
            [edge - 1e-15, 0.0, 0.0],  # a rounding error under it
            [-1e-15, 0.0, 0.0],  # a rounding error under 0
        ],
        **DD,
    )
    _, cell_shift = wrap_to_central_cell(positions, lattice, periodic)

    assert cell_shift[:, 0].tolist() == [-1, -1, 0]


@pytest.mark.cuda
def test_build_ghost_pool_stays_on_input_device() -> None:
    """`build_ghost_pool` must place every tensor it builds on
    ``positions``'/``lattice``'s own device, never relying on the ambient
    default device. A per-axis shift range built without ``device=``
    would make `shift.to(dtype=positions.dtype) @ lattice` mix a CPU
    ``shift`` with a CUDA ``lattice`` whenever the ambient default device
    stays CPU, which this test never touches."""
    positions = torch.tensor(
        [[1.0, 1.0, 1.0]], dtype=torch.double, device="cuda"
    )
    lattice = 5.0 * torch.eye(3, dtype=torch.double, device="cuda")
    periodic = torch.tensor([True, True, True], device="cuda")

    ghost_positions, owner, shift = build_ghost_pool(
        positions, lattice, periodic, cutoff=8.0
    )

    assert ghost_positions.device.type == "cuda"
    assert owner.device.type == "cuda"
    assert shift.device.type == "cuda"
