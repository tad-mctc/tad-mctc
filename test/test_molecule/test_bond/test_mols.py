# This file is part of tad-multicharge.
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
Test bond order functionality for molecules.
"""

from __future__ import annotations

import pytest
import torch

from tad_mctc.batch import pack
from tad_mctc.molecule import bond
from tad_mctc.typing import DD

from ...conftest import DEVICE
from .samples import samples


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["PbH4-BiH3", "C6H5I-CH3SH"])
def test_single(dtype: torch.dtype, name: str):
    dd: DD = {"device": DEVICE, "dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)
    cn = sample["cn"].to(**dd)
    ref = sample["bo"].to(**dd)

    bond_order = bond.guess_bond_order(numbers, positions, cn)
    assert bond_order.dtype == dtype

    mask = bond_order[bond_order > 0.3]
    assert pytest.approx(ref.cpu(), abs=1e-3) == mask.cpu()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_ghost(dtype: torch.dtype):
    dd: DD = {"device": DEVICE, "dtype": dtype}

    sample = samples["PbH4-BiH3"]
    numbers = sample["numbers"].to(DEVICE).clone()
    numbers[[0, 1, 2, 3, 4]] = 0
    positions = sample["positions"].to(**dd)
    cn = sample["cn"].to(**dd)
    ref = torch.tensor([0.5760, 0.5760, 0.5760, 0.5760, 0.5760, 0.5760], **dd)

    bond_order = bond.guess_bond_order(numbers, positions, cn)
    assert bond_order.dtype == dtype

    mask = bond_order[bond_order > 0.3]
    assert pytest.approx(ref.cpu(), abs=1e-3) == mask.cpu()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_batch(dtype: torch.dtype):
    dd: DD = {"device": DEVICE, "dtype": dtype}

    sample1, sample2 = (
        samples["PbH4-BiH3"],
        samples["C6H5I-CH3SH"],
    )
    numbers = pack(
        (
            sample1["numbers"].to(DEVICE),
            sample2["numbers"].to(DEVICE),
        )
    )
    positions = pack(
        (
            sample1["positions"].to(**dd),
            sample2["positions"].to(**dd),
        )
    )
    cn = pack(
        (
            sample1["cn"].to(**dd),
            sample2["cn"].to(**dd),
        )
    )
    ref = torch.cat(
        (
            sample1["bo"].to(**dd),
            sample2["bo"].to(**dd),
        )
    )

    bond_order = bond.guess_bond_order(numbers, positions, cn)
    assert bond_order.dtype == dtype

    mask = bond_order[bond_order > 0.3]
    assert pytest.approx(ref.cpu(), abs=1e-3) == mask.cpu()
