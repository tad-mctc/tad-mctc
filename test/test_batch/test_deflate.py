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
Test the deflating utility functions.
"""

from __future__ import annotations

import pytest
import torch

from tad_mctc.autograd import dgradcheck
from tad_mctc.batch import deflate, pack
from tad_mctc.convert import normalize_device

from ..conftest import DEVICE


def test_fail() -> None:
    tensor = torch.tensor([0, 1, 2, 0, 0, 0])
    with pytest.raises(ValueError):
        deflate(tensor, axis=1)


def test_with_trailing_zeros() -> None:
    tensor = torch.tensor(
        [
            [0, 1, 2, 0, 0, 0],
            [3, 4, 5, 6, 0, 0],
        ]
    )
    expected = torch.tensor(
        [
            [0, 1, 2, 0],
            [3, 4, 5, 6],
        ]
    )
    assert torch.equal(deflate(tensor, value=0, axis=0), expected)


def test_single_system() -> None:
    tensor_single = torch.tensor(
        [
            [0, 1, 0, 0],
            [3, 4, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    )
    expected_single = torch.tensor([[0, 1], [3, 4]])
    assert torch.equal(deflate(tensor_single, value=0), expected_single)


###############################################################################


@pytest.mark.filterwarnings("ignore")
def test_general():
    """General operational test of the `deflate` method."""
    a = torch.tensor(
        [
            [1, 1, 0, 0],
            [2, 2, 2, 0],
            [0, 0, 0, 0],
        ],
        device=DEVICE,
    )

    # Check 1: single system culling, should remove the last column & row.
    check_1 = (deflate(a) == a[:-1, :-1]).all()
    assert check_1, "Failed to correctly deflate a single system"

    # Check 2: batch system culling (axis=0), should remove last column.
    check_2 = (deflate(a, axis=0) == a[:, :-1]).all()
    assert check_2, "Failed to correctly deflate a batch system (axis=0)"

    # Check 3: batch system culling (axis=1), should remove last row.
    check_3 = (deflate(a, axis=1) == a[:-1, :]).all()
    assert check_3, "Failed to correctly deflate a batch system (axis=1)"

    # Check 4: Check value argument is respected, should do nothing here.
    check_4 = (deflate(a, value=-1) == a).all()
    assert check_4, "Failed to ignore an unpadded system"

    # Check 5: ValueError should be raised if axis is specified & tensor is 1d.
    with pytest.raises(ValueError, match="Tensor must be at*"):
        deflate(a[0], axis=0)

    # Check 6: high dimensionality tests (warning: this is dependent on `pack`)
    tensors = [
        torch.full((i, i + 1, i + 2, i + 3, i + 4), i, device=DEVICE)
        for i in range(10)
    ]
    over_packed = pack(tensors, size=torch.Size((20, 19, 23, 20, 30)))
    check_6 = (deflate(over_packed, axis=0) == pack(tensors)).all()
    assert check_6, "Failed to correctly deflate a large batch system"

    # Check 7: ensure the result is placed on the correct device
    check_7 = deflate(a).device == normalize_device(DEVICE)
    assert check_7, "Result was returned on the wrong device"


@pytest.mark.grad
def test_grad():
    """Check the gradient stability of the deflate function."""

    def proxy(tensor):
        # Clean the padding values to prevent unjust failures
        proxy_tensor = torch.zeros_like(tensor)
        proxy_tensor[~mask] = tensor[~mask]
        return deflate(proxy_tensor)

    a = torch.tensor(
        [
            [1, 1, 0, 0],
            [2, 2, 2, 0],
            [0, 0, 0, 0.0],
        ],
        device=DEVICE,
        dtype=torch.double,
        requires_grad=True,
    )

    mask = a == 0
    mask = mask.detach()

    check = dgradcheck(proxy, a, raise_exception=False)
    assert check, "Gradient stability check failed"
