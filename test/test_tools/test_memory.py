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
Test memory functions.
"""

from __future__ import annotations

import pytest
import torch

from tad_mctc.tools import memory_tensor

dtype_cases = [
    (torch.float64, 8),
    (torch.float32, 4),
    (torch.float16, 2),
    (torch.int64, 8),
    (torch.int32, 4),
    (torch.int16, 2),
    (torch.int8, 1),
    (torch.uint8, 1),
]


@pytest.mark.parametrize("dtype, byte_size", dtype_cases)
def test_memory_tensor(dtype: torch.dtype, byte_size: int) -> None:
    size = (10, 10, 10)
    expected_memory_mb = (10 * 10 * 10 * byte_size) / (1024**2)

    estimated_memory = memory_tensor(size, dtype)
    assert pytest.approx(estimated_memory, 0.0001) == expected_memory_mb


def test_unsupported_dtype() -> None:
    size = (10, 10, 10)
    unsupported_dtype = torch.complex128

    with pytest.raises(ValueError) as e_info:
        memory_tensor(size, unsupported_dtype)

    assert f"{unsupported_dtype}" in str(e_info.value)
