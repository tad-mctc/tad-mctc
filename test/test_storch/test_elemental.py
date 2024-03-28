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
        storch.sqrt(torch.tensor([1, 2, 3], **dd), eps=str(0))  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        storch.sqrt(torch.tensor([-1, 2, 3], **dd), eps=-2)


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


###############################################################################


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_divide_fail(dtype: torch.dtype) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}
    x = torch.tensor([1, 2, 3], **dd)
    y = torch.tensor([1, 2, 3], **dd)

    with pytest.raises(TypeError):
        storch.divide(x, y, eps="0")  # type: ignore[arg-type]


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


###############################################################################


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_reciprocal_fail(dtype: torch.dtype) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}
    x = torch.tensor([1, 2, 3], **dd)

    with pytest.raises(TypeError):
        storch.reciprocal(x, eps="0")  # type: ignore[arg-type]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_reciprocal(dtype: torch.dtype) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    x = torch.tensor([-1, 2, 3], **dd)

    out = storch.reciprocal(x)
    assert (torch.isnan(out) == False).all()

    out = storch.reciprocal(x, eps=0.1)
    assert (torch.isnan(out) == False).all()

    out = storch.reciprocal(x, eps=0)
    assert (torch.isnan(out) == False).all()

    out = storch.reciprocal(x, eps=torch.tensor(torch.finfo(dtype).eps))
    assert (torch.isnan(out) == False).all()
