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


def test_pow_fail() -> None:
    dd: DD = {"device": DEVICE, "dtype": torch.float32}
    x = torch.tensor([1, 2, 3], **dd)

    with pytest.raises(TypeError):
        storch.pow(x, 2, eps="0")  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        storch.pow(x, 2, eps=0)

    with pytest.raises(ValueError):
        storch.pow(x, 2, eps=torch.tensor(0.0))

    with pytest.raises(ValueError):
        storch.pow(x, "2")  # type: ignore[arg-type]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("xlist", [[-1, 0, 2], [1, 2, 3]])
def test_pow(dtype: torch.dtype, xlist: list) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}
    x = torch.tensor(xlist, **dd)

    assert (torch.pow(x, 2) == storch.pow(x, 2)).all()

    # positive integer exponents
    out = storch.pow(x, 2, eps=torch.tensor(torch.finfo(dtype).eps))
    assert (torch.isnan(out) == False).all()

    # positive integer exponents with float
    out = storch.pow(x, 2.0)
    assert (torch.isnan(out) == False).all()

    # positive integer scalar tensor exponents
    out = storch.pow(x, torch.tensor(1, **dd))
    assert (torch.isnan(out) == False).all()

    # positive integer tensor exponents
    out = storch.pow(x, torch.tensor([1, 2, 3], **dd))
    assert (torch.isnan(out) == False).all()

    # negative integer exponents
    out = storch.pow(x, -2)
    assert (torch.isnan(out) == False).all()

    # negative integer exponents with float
    out = storch.pow(x, -2.0)
    assert (torch.isnan(out) == False).all()

    # negative integer scalar tensor exponents
    out = storch.pow(x, torch.tensor(-1, **dd))
    assert (torch.isnan(out) == False).all()

    # negative integer tensor exponents
    out = storch.pow(x, torch.tensor([-1, -2, -3], **dd))
    assert (torch.isnan(out) == False).all()

    # positive fractional exponents
    out = storch.pow(x, 0.5)
    assert (torch.isnan(out) == False).all()

    # positive fractional scalar tensor exponents
    out = storch.pow(x, torch.tensor(0.5, **dd))
    assert (torch.isnan(out) == False).all()

    # positive fractional tensor exponents
    out = storch.pow(x, torch.tensor([0.5, 1, 2], **dd))
    assert (torch.isnan(out) == False).all()

    # negative fractional exponents
    out = storch.pow(x, -0.5)
    assert (torch.isnan(out) == False).all()

    # negative fractional scalar tensor exponents
    out = storch.pow(x, torch.tensor(-0.5, **dd))
    assert (torch.isnan(out) == False).all()

    # negative fractional tensor exponents
    out = storch.pow(x, torch.tensor([-0.5, -1, -2], **dd))
    assert (torch.isnan(out) == False).all()

    # zero exponents
    out = storch.pow(x, 0, eps=1.0e-10)
    assert (torch.isnan(out) == False).all()

    out = storch.pow(x, torch.tensor(0, **dd))
    assert (torch.isnan(out) == False).all()

    out = storch.pow(x, torch.tensor([0, 0, 0], **dd), eps=999)
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
