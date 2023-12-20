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
Test elemental safeops.
"""
from __future__ import annotations

import pytest
import torch

from tad_mctc import storch
from tad_mctc.typing import DD

from ..conftest import DEVICE


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_sqrt_fail(dtype: torch.dtype) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    with pytest.raises(TypeError):
        storch.sqrt(torch.tensor([1, 2, 3], **dd), "0")  # type: ignore

    with pytest.raises(ValueError):
        storch.sqrt(torch.tensor([-1, 2, 3], **dd), -2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_sqrt(dtype: torch.dtype) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    x = torch.tensor([-1, 2, 3], **dd)

    out = storch.sqrt(x)
    assert (torch.isnan(out) == False).all()

    out = storch.sqrt(x, eps=0.1)
    assert (torch.isnan(out) == False).all()

    out = storch.sqrt(x, eps=0)
    assert (torch.isnan(out) == False).all()

    out = storch.sqrt(x, eps=torch.tensor(torch.finfo(dtype).eps))
    assert (torch.isnan(out) == False).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_divide_fail(dtype: torch.dtype) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}
    x = torch.tensor([1, 2, 3], **dd)
    y = torch.tensor([1, 2, 3], **dd)

    with pytest.raises(TypeError):
        storch.divide(x, y, eps="0")  # type: ignore


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_divide(dtype: torch.dtype) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    x = torch.tensor([-1, 2, 3], **dd)
    y = torch.tensor([0, 0, 3], **dd)

    out = storch.divide(x, y)
    assert (torch.isnan(out) == False).all()

    out = storch.divide(x, y, eps=0.1)
    assert (torch.isnan(out) == False).all()

    out = storch.divide(x, y, eps=0)
    assert (torch.isnan(out) == False).all()

    out = storch.divide(x, y, eps=torch.tensor(torch.finfo(dtype).eps))
    assert (torch.isnan(out) == False).all()
