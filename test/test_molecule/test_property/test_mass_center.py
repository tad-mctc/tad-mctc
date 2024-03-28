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
