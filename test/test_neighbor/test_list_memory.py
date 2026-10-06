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
"""Test `estimate_neighborlist_memory`."""

from __future__ import annotations

import pytest
import torch

from tad_mctc.neighbor.list import (
    build_neighborlist,
    estimate_neighborlist_memory,
)

from ..utils import hydrogens


def test_matches_a_molecular_lists_own_tensor_bytes() -> None:
    """The estimate must track `idx_i`/`idx_j`/`mask`'s actual combined
    size, so a dtype change to any of those slots is caught here rather
    than only staying consistent with itself. A molecular list's shift is
    one zero row, whatever the capacity."""
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [10.0, 10.0, 10.0],
        ]
    )
    nbl = build_neighborlist(hydrogens(positions), cutoff=2.0, tile=2)
    actual = sum(
        t.element_size() * t.nelement()
        for t in (nbl.idx_i, nbl.idx_j, nbl.mask)
    )
    capacity = int(nbl.idx_i.shape[0])

    assert nbl.shift.untyped_storage().nbytes() == 6
    assert estimate_neighborlist_memory(capacity, periodic=False) == actual


def test_matches_a_periodic_lists_own_tensor_bytes() -> None:
    """A periodic list stores a shift per slot, which the estimate must
    include."""
    lattice = 6.0 * torch.eye(3)
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    nbl = build_neighborlist(
        hydrogens(positions, lattice=lattice), cutoff=4.0, tile=2
    )
    actual = sum(
        t.element_size() * t.nelement()
        for t in (nbl.idx_i, nbl.idx_j, nbl.shift, nbl.mask)
    )
    capacity = int(nbl.idx_i.shape[0])

    assert estimate_neighborlist_memory(capacity, periodic=True) == actual


def test_matches_measured_glu_ala_capacity() -> None:
    """Regression pin at the capacity this function was first checked
    against (`examples/scaling/glu_ala_b_512_to_65536/4096.xyz`): 15 bytes
    per periodic slot, 9 per molecular slot (int32 atom indices)."""
    capacity = 11_735_040
    assert estimate_neighborlist_memory(capacity, periodic=True) == 176_025_600
    assert estimate_neighborlist_memory(capacity, periodic=False) == 105_615_360


def test_zero_capacity_is_zero() -> None:
    """An empty list has zero size."""
    assert estimate_neighborlist_memory(0, periodic=False) == 0
    assert estimate_neighborlist_memory(0, periodic=True) == 0


def test_negative_capacity_raises() -> None:
    """A negative capacity is not a valid list size."""
    with pytest.raises(ValueError):
        estimate_neighborlist_memory(-1, periodic=False)
