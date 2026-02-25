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
Test the batch-agnostic drop-in functions.
"""

from __future__ import annotations

import torch

from tad_mctc.batch import eye

from ..conftest import DEVICE


def test_eye_single():
    shape = (3, 3)
    value = 1.0
    tensor = torch.empty(shape, device=DEVICE)

    result = eye(shape, value=value, device=DEVICE)

    assert result.shape == shape
    assert result.device == tensor.device
    assert result.dtype == tensor.dtype

    # Check the diagonal values and off-diagonal values
    assert torch.all(result.diag() == value)
    assert torch.all(result[result != value] == 0)


def test_eye_batch():
    shape = (5, 3, 3)
    value = 1.0
    tensor = torch.empty(shape, device=DEVICE)

    result = eye(shape, value=value, device=DEVICE)

    assert result.shape == shape
    assert result.device == tensor.device
    assert result.dtype == tensor.dtype

    # Check each identity matrix in the batch
    for i in range(shape[0]):
        assert torch.all(result[i].diag() == value)
        assert torch.all(result[i][result[i] != value] == 0)
