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
Test the sorting utility functions.
"""

from __future__ import annotations

import pytest
import torch

from tad_mctc.autograd import dgradcheck
from tad_mctc.batch import merge, pack
from tad_mctc.convert import normalize_device

from ..conftest import DEVICE


def test_merge():
    """
    Operational tests of the merge function.

    Warnings:
        This test is depended upon the `pack` function. Thus it will fail if
        the `pack` function is in error.
    """
    # Check 1: ensure the expected result is returned
    a = [
        torch.full((i, i + 1, i + 2, i + 3, i + 4), float(i), device=DEVICE)
        for i in range(6)
    ]

    merged = merge([pack(a[:2]), pack(a[2:4]), pack(a[4:])])
    packed = pack(a)
    check_1 = (merged == packed).all()
    assert check_1, "Merge attempt failed"

    # Check 2: test axis argument's functionality
    merged = merge(
        [pack(a[:2], axis=1), pack(a[2:4], axis=1), pack(a[4:], axis=1)], axis=1
    )
    packed = pack(a, axis=1)
    check_2 = (merged == packed).all()
    assert check_2, "Merge attempt failed when axis != 0"

    # Check 3: device persistence check
    check_3 = packed.device == normalize_device(DEVICE)
    assert check_3, "Device persistence check failed"


@pytest.mark.grad
def test_merge_grad():
    """Checks gradient stability of the merge function."""

    def proxy(a_in, b_in):
        # Clean padding values
        a_proxy = torch.zeros_like(a_in)
        b_proxy = torch.zeros_like(b_in)
        a_proxy[~a_mask] = a_in[~a_mask]
        b_proxy[~b_mask] = b_in[~b_mask]
        return merge([a_proxy, b_proxy])

    a = torch.tensor(
        [[0, 1, 0], [2, 3, 4.0]],
        device=DEVICE,
        dtype=torch.double,
        requires_grad=True,
    )

    b = torch.tensor(
        [[5, 6, 7, 0], [8, 9, 10, 11.0]],
        device=DEVICE,
        dtype=torch.double,
        requires_grad=True,
    )

    a_mask = (a == 0).detach()
    b_mask = (b == 0).detach()

    check = dgradcheck(proxy, (a, b), raise_exception=False)
    assert check, "Gradient stability check failed"
