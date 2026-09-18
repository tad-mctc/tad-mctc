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
Test the free molecular property functions that replaced ``Mol``.
"""

from __future__ import annotations

import pytest
import torch

from tad_mctc import storch
from tad_mctc.data.structures import structures
from tad_mctc.data.structures.mstore import get_structure
from tad_mctc.io.structure import Structure
from tad_mctc.properties.general import sum_formula
from tad_mctc.typing import DD

# Historical name -> (mstore collection, record), or None for a bespoke
# `tad_mctc.data.structures` entry -- see `test_ncoord/samples.py` for the
# same pattern and its rationale.
_SAMPLE_SOURCES: dict[str, tuple[str, str] | None] = {
    "H2": ("mb16_43", "H2"),
    "LiH": ("mb16_43", "LiH"),
    "H2O": ("heavy28", "h2o"),
    "SiH4": ("mb16_43", "SiH4"),
    "MB16_43_01": ("mb16_43", "01"),
    "vancoh2": None,
}


def _structure(name: str) -> Structure:
    source = _SAMPLE_SOURCES[name]
    if source is None:
        return structures[name]
    collection, record = source
    return get_structure(collection, record)


sample_list = list(_SAMPLE_SOURCES)

device = None


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_dist(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"device": device, "dtype": dtype}

    sample = _structure(name)
    numbers = sample.numbers.to(device)
    positions = sample.positions.to(**dd)

    dist = storch.cdist(positions)

    assert dist.shape[-1] == numbers.shape[-1]
    assert dist.shape[-2] == numbers.shape[-1]


@pytest.mark.parametrize("name", sample_list)
def test_formula(name: str) -> None:
    sample = _structure(name)
    numbers = sample.numbers.to(device)

    ref = {
        "H2": "H2",
        "LiH": "HLi",
        "H2O": "H2O",
        "SiH4": "H4Si",
        "MB16_43_01": "H6B2N2O2FNaAlCl",
        "vancoh2": "H77C66N9O24",
    }

    assert ref[name] == sum_formula(numbers)
