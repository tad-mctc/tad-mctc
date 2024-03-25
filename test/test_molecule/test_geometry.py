# This file is part of tad-mctc.
#
# SPDX-Identifier: LGPL-3.0
# Copyright (C) 2023 Marvin Friede
#
# tad-mctc is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# tad_mctc is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with tad-mctc. If not, see <https://www.gnu.org/licenses/>.
"""
Test the calculation of the geometric properties of a molecule.
"""

from __future__ import annotations

import pytest
import torch

from tad_mctc.batch import pack
from tad_mctc.data.molecules import mols as samples
from tad_mctc.molecule import geometry
from tad_mctc.typing import DD

from ..conftest import DEVICE


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["LiH", "CO2"])
def test_linear(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    numbers = samples[name]["numbers"].to(DEVICE)
    positions = samples[name]["positions"].to(**dd)

    mask = geometry.is_linear(numbers, positions)
    assert (mask == torch.tensor([True])).all()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", ["LiH", "CO2"])
@pytest.mark.parametrize("name2", ["LiH", "CO2"])
def test_linear_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    numbers = pack(
        [
            samples[name1]["numbers"].to(DEVICE),
            samples[name2]["numbers"].to(DEVICE),
        ]
    )
    positions = pack(
        [
            samples[name1]["positions"].to(**dd),
            samples[name2]["positions"].to(**dd),
        ]
    )

    mask = geometry.is_linear(numbers, positions)
    assert (mask == torch.tensor([True, True])).all()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["H2O", "SiH4"])
def test_nonlinear(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    numbers = samples[name]["numbers"].to(DEVICE)
    positions = samples[name]["positions"].to(**dd)

    mask = geometry.is_linear(numbers, positions)
    assert (mask == torch.tensor([False])).all()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", ["H2O", "SiH4"])
@pytest.mark.parametrize("name2", ["H2O", "SiH4"])
def test_nonlinear_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    numbers = pack(
        [
            samples[name1]["numbers"].to(DEVICE),
            samples[name2]["numbers"].to(DEVICE),
        ]
    )
    positions = pack(
        [
            samples[name1]["positions"].to(**dd),
            samples[name2]["positions"].to(**dd),
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
            samples[name1]["numbers"].to(DEVICE),
            samples[name2]["numbers"].to(DEVICE),
        ]
    )
    positions = pack(
        [
            samples[name1]["positions"].to(**dd),
            samples[name2]["positions"].to(**dd),
        ]
    )

    mask = geometry.is_linear(numbers, positions)
    assert (mask == torch.tensor([False, True])).all()
