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
Test the molecule representation.
"""

from __future__ import annotations

import pytest
import torch

from tad_mctc.data.molecules import mols as samples
from tad_mctc.molecule.container import Mol
from tad_mctc.typing import DD

sample_list = ["H2", "LiH", "H2O", "SiH4", "MB16_43_01", "vancoh2"]

device = None


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_dist(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"device": device, "dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"].to(device)
    positions = sample["positions"].to(**dd)

    mol = Mol(numbers, positions)
    mol.clear_cache()
    dist = mol.distances()
    mol.clear_cache()

    assert dist.shape[-1] == numbers.shape[-1]
    assert dist.shape[-2] == numbers.shape[-1]


@pytest.mark.parametrize("name", sample_list)
def test_name(name: str) -> None:
    sample = samples[name]
    numbers = sample["numbers"].to(device)
    positions = sample["positions"].to(device)

    mol = Mol(numbers, positions, name=name)
    assert mol.name == name

    mol.name = "wrong"
    assert mol.name != name


def test_cache() -> None:
    sample = samples["SiH4"]
    numbers = sample["numbers"]
    positions = sample["positions"]

    mol = Mol(numbers, positions)
    assert hasattr(mol.distances, "clear")

    del mol.distances.__dict__["clear"]
    assert not hasattr(mol.distances, "clear")

    # clear cache should still execute without error
    mol.clear_cache()


@pytest.mark.parametrize("name", sample_list)
def test_formula(name: str) -> None:
    sample = samples[name]
    numbers = sample["numbers"].to(device)
    positions = sample["positions"].to(device)

    ref = {
        "H2": "H2",
        "LiH": "HLi",
        "H2O": "H2O",
        "SiH4": "H4Si",
        "MB16_43_01": "H6B2N2O2FNaAlCl",
        "vancoh2": "H77C66N9O24",
    }

    mol = Mol(numbers, positions)
    assert ref[name] == mol.sum_formula()
