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
Test the unpacking utility functions.
"""
from __future__ import annotations

import pytest
import torch

from tad_mctc.batch import deflate, unpack


def test_deflate_fail() -> None:
    tensor = torch.tensor([0, 1, 2, 0, 0, 0])
    with pytest.raises(ValueError):
        deflate(tensor, axis=1)


def test_deflate_with_trailing_zeros() -> None:
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


def test_deflate_single_system() -> None:
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


# Tests for unpack function
def test_unpack_multiple_tensors() -> None:
    packed_tensor = torch.tensor(
        [
            [1, 2, 3, 0, 0],
            [4, 5, 0, 0, 0],
            [6, 7, 8, 9, 0],
        ]
    )
    unpacked_tensors = unpack(packed_tensor, value=0, axis=0)

    assert len(unpacked_tensors) == 3
    assert torch.equal(unpacked_tensors[0], torch.tensor([1, 2, 3]))
    assert torch.equal(unpacked_tensors[1], torch.tensor([4, 5]))
    assert torch.equal(unpacked_tensors[2], torch.tensor([6, 7, 8, 9]))


def test_unpack_empty_tensor() -> None:
    packed_empty = torch.tensor([[]])
    unpacked_empty = unpack(packed_empty, value=0, axis=0)
    assert len(unpacked_empty) == 1
    assert torch.equal(unpacked_empty[0], torch.tensor([]))


def test_unpack_1d_tensor() -> None:
    packed_empty = torch.tensor([1, 0])
    unpacked_empty = unpack(packed_empty, value=0, axis=0)
    assert len(unpacked_empty) == 1
    assert torch.equal(unpacked_empty[0], torch.tensor([1]))


def test_unpack_2d_tensor() -> None:
    packed_empty = torch.tensor([[1, 0]])
    unpacked_empty = unpack(packed_empty, value=0, axis=0)
    assert len(unpacked_empty) == 1
    assert torch.equal(unpacked_empty[0], torch.tensor([1]))
