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
Periodic cells beyond a plain right-handed bulk cell.

A batch shares one shift table built for the union of its periodic axes.
A system that is not periodic along one of those axes must still get the
same coordination number as it does on its own. A left-handed cell is the
same crystal as its right-handed counterpart, so it must give the same
coordination number.
"""

from __future__ import annotations

import pytest
import torch

from tad_mctc.autograd import vmap
from tad_mctc.io.structure import Structure, pack_structures
from tad_mctc.ncoord import cn_d3
from tad_mctc.neighbor.images import build_shared_periodic_shifts
from tad_mctc.typing import DD, Tensor

from ..conftest import DEVICE

# A short lattice vector along the non-periodic axes, as the Turbomole
# `$cell` reader writes for a slab or wire. Images along such an axis
# would land inside the cutoff if they were not masked out.
PLACEHOLDER = 1.0


def _carbon_pair(
    dd: DD, lattice: torch.Tensor, periodic: list[bool]
) -> Structure:
    numbers = torch.tensor([6, 6], device=DEVICE)
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.4, 1.4, 0.0]], **dd)
    return Structure(
        numbers=numbers,
        positions=positions,
        lattice=lattice,
        periodic=torch.tensor(periodic, device=DEVICE),
    )


PERIODIC_CASES = [
    [True, True, False],  # slab
    [True, False, False],  # wire
    [False, False, False],  # molecule in a box
]


def _bulk_and_lower_dim(dd: DD, periodic: list[bool]) -> Structure:
    """A bulk cell packed with a system that is periodic only along
    `periodic`, with placeholder lattice vectors on its open axes."""
    bulk = _carbon_pair(dd, 4.7 * torch.eye(3, **dd), [True, True, True])

    lattice = 4.7 * torch.eye(3, **dd)
    for axis, is_periodic in enumerate(periodic):
        if not is_periodic:
            lattice[axis, axis] = PLACEHOLDER
    lower_dim = _carbon_pair(dd, lattice, periodic)

    return pack_structures([bulk, lower_dim])


def _single_system_cns(batch: Structure) -> Tensor:
    assert batch.lattice is not None and batch.periodic is not None
    return torch.stack(
        [
            cn_d3(
                Structure(
                    numbers=batch.numbers[i],
                    positions=batch.positions[i],
                    lattice=batch.lattice[i],
                    periodic=batch.periodic[i],
                )
            )
            for i in range(batch.numbers.shape[0])
        ]
    )


@pytest.mark.parametrize("periodic", PERIODIC_CASES)
def test_batch_matches_single_for_mixed_periodicity(
    periodic: list[bool],
) -> None:
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    batch = _bulk_and_lower_dim(dd, periodic)

    batched = cn_d3(batch)

    expected = _single_system_cns(batch)
    assert torch.allclose(batched, expected, atol=1e-12, rtol=0)


@pytest.mark.parametrize("periodic", PERIODIC_CASES)
def test_vmap_matches_single_for_mixed_periodicity(
    periodic: list[bool],
) -> None:
    """Under `vmap` every lane is a single system, but the shift table is
    still shared and built for the union of the lanes' periodic axes."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    batch = _bulk_and_lower_dim(dd, periodic)
    assert batch.lattice is not None and batch.periodic is not None

    shifts = build_shared_periodic_shifts(
        batch.lattice, batch.periodic, cutoff=cn_d3.cutoff
    )

    def cn_one(
        numbers: Tensor, positions: Tensor, lattice: Tensor, mask: Tensor
    ) -> Tensor:
        structure = Structure(
            numbers=numbers, positions=positions, lattice=lattice, periodic=mask
        )
        return cn_d3.with_precomputed_shifts(structure, shifts=shifts)

    vmapped = vmap(cn_one)(
        batch.numbers, batch.positions, batch.lattice, batch.periodic
    )

    expected = _single_system_cns(batch)
    assert torch.allclose(vmapped, expected, atol=1e-12, rtol=0)


@pytest.mark.parametrize(
    "periodic",
    [
        [True, True, True],  # bulk
        [True, True, False],  # slab, as a left-handed `$lattice` block
    ],
)
def test_left_handed_cell_matches_right_handed(periodic: list[bool]) -> None:
    """Swapping the first two lattice vectors flips the handedness of the
    cell but describes the same crystal."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    lattice = torch.tensor(
        [[4.7, 0.0, 0.0], [0.6, 5.1, 0.0], [0.0, 0.0, 5.5]], **dd
    )
    if not periodic[2]:
        lattice[2, 2] = PLACEHOLDER

    right_handed = _carbon_pair(dd, lattice, periodic)
    left_handed = right_handed.replace(lattice=lattice[[1, 0, 2]])
    assert left_handed.lattice is not None
    assert torch.linalg.det(left_handed.lattice) < 0

    expected = cn_d3(right_handed)
    assert torch.allclose(cn_d3(left_handed), expected, atol=1e-12, rtol=0)
