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
Test the calculation of the mass center.
"""

from __future__ import annotations

import pytest
import torch

from tad_mctc.molecule import property
from tad_mctc.typing import DD

from ...conftest import DEVICE


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_single(dtype: torch.dtype) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    masses = torch.tensor([1.0, 2.0], **dd)
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], **dd)

    ref = torch.tensor([2.0 / 3, 0.0, 0.0], **dd)
    assert pytest.approx(ref) == property.mass_center(masses, positions)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_batch(dtype: torch.dtype) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    masses = torch.tensor([[1.0, 2.0], [2.0, 1.0]], **dd)
    positions = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        ],
        **dd,
    )

    ref = torch.tensor([[2.0 / 3, 0.0, 0.0], [0.0, 1.0 / 3, 0.0]], **dd)
    assert pytest.approx(ref) == property.mass_center(masses, positions)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_zero(dtype: torch.dtype) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    masses = torch.tensor([0.0, 2.0], **dd)
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], **dd)

    ref = torch.tensor([1.0, 0.0, 0.0], **dd)
    assert pytest.approx(ref) == property.mass_center(masses, positions)
