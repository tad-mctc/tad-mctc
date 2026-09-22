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
Test the calculation of the geometric properties of a molecule.
"""

from __future__ import annotations

import pytest
import torch

from tad_mctc.batch import pack
from tad_mctc.data.structures import get_structure
from tad_mctc.properties import geometry
from tad_mctc.typing import DD

from ..conftest import DEVICE

# Historical name -> (mstore collection, record), or None for a bespoke
# `tad_mctc.data.structures` entry (`CO2` has no molecular mstore
# namesake -- `x23/CO2` is a crystal) -- see `test_ncoord/samples.py` for
# the same pattern and its rationale.
_SAMPLE_SOURCES: dict[str, tuple[str, str] | None] = {
    "LiH": ("mb16_43", "LiH"),
    "CO2": None,
    "H2O": ("heavy28", "h2o"),
    "SiH4": ("mb16_43", "SiH4"),
}
samples = {
    name: (
        get_structure("other", name)
        if source is None
        else get_structure(*source)
    )
    for name, source in _SAMPLE_SOURCES.items()
}


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["LiH", "CO2"])
def test_linear(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    numbers = samples[name].numbers.to(DEVICE)
    positions = samples[name].positions.to(**dd)

    mask = geometry.is_linear(numbers, positions)
    assert (mask == torch.tensor([True])).all()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", ["LiH", "CO2"])
@pytest.mark.parametrize("name2", ["LiH", "CO2"])
def test_linear_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    numbers = pack(
        [
            samples[name1].numbers.to(DEVICE),
            samples[name2].numbers.to(DEVICE),
        ]
    )
    positions = pack(
        [
            samples[name1].positions.to(**dd),
            samples[name2].positions.to(**dd),
        ]
    )

    mask = geometry.is_linear(numbers, positions)
    assert (mask == torch.tensor([True, True])).all()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["H2O", "SiH4"])
def test_nonlinear(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    numbers = samples[name].numbers.to(DEVICE)
    positions = samples[name].positions.to(**dd)

    mask = geometry.is_linear(numbers, positions)
    assert (mask == torch.tensor([False])).all()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", ["H2O", "SiH4"])
@pytest.mark.parametrize("name2", ["H2O", "SiH4"])
def test_nonlinear_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    numbers = pack(
        [
            samples[name1].numbers.to(DEVICE),
            samples[name2].numbers.to(DEVICE),
        ]
    )
    positions = pack(
        [
            samples[name1].positions.to(**dd),
            samples[name2].positions.to(**dd),
        ]
    )

    mask = geometry.is_linear(numbers, positions)
    assert (mask == torch.tensor([False, False])).all()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", ["H2O", "SiH4"])
@pytest.mark.parametrize("name2", ["LiH", "CO2"])
def test_mixed_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    numbers = pack(
        [
            samples[name1].numbers.to(DEVICE),
            samples[name2].numbers.to(DEVICE),
        ]
    )
    positions = pack(
        [
            samples[name1].positions.to(**dd),
            samples[name2].positions.to(**dd),
        ]
    )

    mask = geometry.is_linear(numbers, positions)
    assert (mask == torch.tensor([False, True])).all()
