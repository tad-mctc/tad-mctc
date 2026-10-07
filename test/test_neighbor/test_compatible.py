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
Test `check_compatible` on `PeriodicShifts` and `NeighborList`: which
structure and cutoff each may be used with.

Periodic shifts may cover more axes than a structure is periodic along; a
neighbour list must match them exactly, since its pair shifts already
encode the axes it was built for.
"""

from __future__ import annotations

import pytest
import torch

from tad_mctc.io.structure import Structure, pack_structures
from tad_mctc.neighbor.images import (
    PeriodicShifts,
    build_periodic_shifts,
)
from tad_mctc.neighbor.list import NeighborList, build_neighborlist
from tad_mctc.typing import DD

from ..conftest import DEVICE
from ..utils import load_structure

CUTOFF = 8.0

ALL_AXES = [True, True, True]
SLAB_AXES = [True, True, False]


def _cell(dd: DD, periodic: list[bool]) -> Structure:
    """A cubic cell, periodic along the axes `periodic` marks."""
    structure = load_structure("other", "periodic_cubic", dd)
    mask = torch.tensor(periodic, device=dd["device"])
    return structure.replace(periodic=mask)


def _shifts_for(
    structure: Structure, periodic: list[bool], cutoff: float
) -> PeriodicShifts:
    assert structure.lattice is not None
    mask = torch.tensor(periodic, device=structure.positions.device)
    return build_periodic_shifts(structure.lattice, mask, cutoff)


def _list_for(
    structure: Structure, periodic: list[bool], cutoff: float
) -> NeighborList:
    mask = torch.tensor(periodic, device=structure.positions.device)
    return build_neighborlist(structure.replace(periodic=mask), cutoff)


########################################################################
# PeriodicShifts


def test_shifts_with_matching_axes_pass() -> None:
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    slab = _cell(dd, SLAB_AXES)

    shifts = _shifts_for(slab, SLAB_AXES, CUTOFF)
    shifts.check_compatible(slab, CUTOFF)


def test_shifts_covering_extra_axes_pass() -> None:
    """A consumer drops the shifts along a structure's open axes itself,
    so a table built for all three axes serves a slab."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    slab = _cell(dd, SLAB_AXES)

    shifts = _shifts_for(slab, ALL_AXES, CUTOFF)
    shifts.check_compatible(slab, CUTOFF)


def test_shifts_missing_an_axis_raise() -> None:
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    bulk = _cell(dd, ALL_AXES)

    shifts = _shifts_for(bulk, SLAB_AXES, CUTOFF)
    with pytest.raises(ValueError, match="periodic_axes"):
        shifts.check_compatible(bulk, CUTOFF)


def test_shifts_with_short_cutoff_raise() -> None:
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    bulk = _cell(dd, ALL_AXES)

    shifts = _shifts_for(bulk, ALL_AXES, CUTOFF / 2)
    with pytest.raises(ValueError, match="cutoff"):
        shifts.check_compatible(bulk, CUTOFF)


def test_shifts_for_structure_without_lattice_raise() -> None:
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    bulk = _cell(dd, ALL_AXES)
    molecule = bulk.replace(lattice=None, periodic=None)

    shifts = _shifts_for(bulk, ALL_AXES, CUTOFF)
    with pytest.raises(ValueError, match="lattice"):
        shifts.check_compatible(molecule, CUTOFF)


def test_shifts_built_for_a_larger_cell_raise() -> None:
    """A cell compressed after its table was built (an NPT step, say)
    needs more image rings than the table holds."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    bulk = _cell(dd, ALL_AXES)
    assert bulk.lattice is not None
    compressed = bulk.replace(
        positions=0.5 * bulk.positions, lattice=0.5 * bulk.lattice
    )

    shifts = _shifts_for(bulk, ALL_AXES, CUTOFF)
    with pytest.raises(ValueError, match="image rings"):
        shifts.check_compatible(compressed, CUTOFF)


def test_shifts_built_for_a_smaller_cell_pass() -> None:
    """Extra rings are only masked out by the cutoff downstream."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    bulk = _cell(dd, ALL_AXES)
    assert bulk.lattice is not None
    compressed = bulk.replace(
        positions=0.5 * bulk.positions, lattice=0.5 * bulk.lattice
    )

    shifts = _shifts_for(compressed, ALL_AXES, CUTOFF)
    shifts.check_compatible(bulk, CUTOFF)


def test_shifts_of_one_cell_for_a_batch_raise() -> None:
    """A table built for the batch's largest cell misses the outer images
    of its smaller one."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    bulk = _cell(dd, ALL_AXES)
    assert bulk.lattice is not None
    compressed = bulk.replace(
        positions=0.5 * bulk.positions, lattice=0.5 * bulk.lattice
    )
    batch = pack_structures([bulk, compressed])

    shifts = _shifts_for(bulk, ALL_AXES, CUTOFF)
    with pytest.raises(ValueError, match="image rings"):
        shifts.check_compatible(batch, CUTOFF)


def test_shared_shifts_for_a_batch_pass() -> None:
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    bulk = _cell(dd, ALL_AXES)
    assert bulk.lattice is not None
    compressed = bulk.replace(
        positions=0.5 * bulk.positions, lattice=0.5 * bulk.lattice
    )
    batch = pack_structures([bulk, compressed])
    assert batch.lattice is not None and batch.periodic is not None

    shifts = build_periodic_shifts(batch.lattice, batch.periodic, CUTOFF)
    shifts.check_compatible(batch, CUTOFF)


########################################################################
# NeighborList


def test_list_with_matching_axes_pass() -> None:
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    slab = _cell(dd, SLAB_AXES)

    nbl = _list_for(slab, SLAB_AXES, CUTOFF)
    nbl.check_compatible(slab, CUTOFF)


def test_list_covering_extra_axes_raise() -> None:
    """Unlike periodic shifts, a list built for all three axes adds images
    along a slab's open axis, so it must not serve a slab."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    slab = _cell(dd, SLAB_AXES)

    nbl = _list_for(slab, ALL_AXES, CUTOFF)
    with pytest.raises(ValueError, match="periodic_axes"):
        nbl.check_compatible(slab, CUTOFF)


def test_list_missing_an_axis_raise() -> None:
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    bulk = _cell(dd, ALL_AXES)

    nbl = _list_for(bulk, SLAB_AXES, CUTOFF)
    with pytest.raises(ValueError, match="periodic_axes"):
        nbl.check_compatible(bulk, CUTOFF)


def test_list_with_short_cutoff_raise() -> None:
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    bulk = _cell(dd, ALL_AXES)

    nbl = _list_for(bulk, ALL_AXES, CUTOFF / 2)
    with pytest.raises(ValueError, match="cutoff"):
        nbl.check_compatible(bulk, CUTOFF)


def test_periodic_list_for_structure_without_lattice_raise() -> None:
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    bulk = _cell(dd, ALL_AXES)
    molecule = bulk.replace(lattice=None, periodic=None)

    nbl = _list_for(bulk, ALL_AXES, CUTOFF)
    with pytest.raises(ValueError, match="lattice"):
        nbl.check_compatible(molecule, CUTOFF)


def test_molecular_list_for_periodic_structure_raise() -> None:
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    bulk = _cell(dd, ALL_AXES)

    molecule = bulk.replace(lattice=None, periodic=None)
    nbl = build_neighborlist(molecule, CUTOFF)
    with pytest.raises(ValueError, match="molecular list"):
        nbl.check_compatible(bulk, CUTOFF)


def test_overflowed_list_raise() -> None:
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    bulk = _cell(dd, ALL_AXES)

    nbl = build_neighborlist(
        bulk,
        CUTOFF,
        capacity=1,
    )
    assert nbl.overflow is True

    with pytest.raises(ValueError, match="overflow"):
        nbl.check_compatible(bulk, CUTOFF)


def test_list_for_other_atoms_raise() -> None:
    """A list indexes the atoms it was built over, so a structure with a
    different atom layout (here: the same cell, batched) must not use it."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    bulk = _cell(dd, ALL_AXES)
    batch = pack_structures([bulk, bulk])

    nbl = _list_for(bulk, ALL_AXES, CUTOFF)
    with pytest.raises(ValueError, match="shape"):
        nbl.check_compatible(batch, CUTOFF)


@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize(("built", "current"), [(6, 0), (0, 6)])
def test_list_rejects_a_changed_real_atom_mask(
    periodic: bool, built: int, current: int
) -> None:
    """A list cannot cross a real-atom/padding transition."""
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=torch.double,
    )
    lattice = 12.0 * torch.eye(3, dtype=torch.double) if periodic else None
    structure = Structure(
        numbers=torch.tensor([1, built, 8]),
        positions=positions,
        lattice=lattice,
    )
    changed = structure.replace(numbers=torch.tensor([1, current, 8]))

    nbl = build_neighborlist(structure, CUTOFF, skin=1.0)

    with pytest.raises(ValueError, match="real-atom mask"):
        nbl.check_compatible(changed, CUTOFF)
    assert nbl.stale(changed)


def test_batched_list_rejects_a_changed_real_atom_mask() -> None:
    """A padding transition in one batch member makes the whole list stale."""
    positions = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        ],
        dtype=torch.double,
    )
    structure = Structure(
        numbers=torch.tensor([[1, 6, 8], [1, 0, 8]]), positions=positions
    )
    changed = structure.replace(numbers=torch.tensor([[1, 0, 8], [1, 0, 8]]))

    nbl = build_neighborlist(structure, CUTOFF, skin=1.0)

    with pytest.raises(ValueError, match="real-atom mask"):
        nbl.check_compatible(changed, CUTOFF)
    assert nbl.stale(changed)


def test_list_accepts_a_different_nonzero_element() -> None:
    """Changing an element does not change pair topology."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    structure = _cell(dd, ALL_AXES)
    changed = structure.replace(numbers=structure.numbers.roll(1, dims=-1))
    assert bool((changed.numbers != 0).all())

    nbl = build_neighborlist(structure, CUTOFF, skin=1.0)

    nbl.check_compatible(changed, CUTOFF)
    assert not nbl.stale(changed)
