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

import torch

from tad_mctc.batch import deflate, unpack
from tad_mctc.convert import normalize_device

from ..conftest import DEVICE


def test_multiple_tensors() -> None:
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


def test_empty_tensor() -> None:
    packed_empty = torch.tensor([[]])
    unpacked_empty = unpack(packed_empty, value=0, axis=0)
    assert len(unpacked_empty) == 1
    assert torch.equal(unpacked_empty[0], torch.tensor([]))


def test_1d_tensor() -> None:
    packed_empty = torch.tensor([1, 0])
    unpacked_empty = unpack(packed_empty, value=0, axis=0)
    assert len(unpacked_empty) == 1
    assert torch.equal(unpacked_empty[0], torch.tensor([1]))


def test_2d_tensor() -> None:
    packed_empty = torch.tensor([[1, 0]])
    unpacked_empty = unpack(packed_empty, value=0, axis=0)
    assert len(unpacked_empty) == 1
    assert torch.equal(unpacked_empty[0], torch.tensor([1]))


###############################################################################


def test_general():
    """
    Ensures unpack functions as intended.

    Notes
    -----
    The ``unpack`` function does not require an in-depth test as it is just
    a wrapper for the ``deflate`` method. Hence, no grad check exists.

    Warnings
    --------
    This test and the method that is being tested are both dependent on
    the ``deflate`` method.
    """
    # Check 1: Unpacking without padding
    a = torch.tensor([[0, 1, 2, 0], [3, 4, 5, 0], [0, 0, 1, 0]], device=DEVICE)

    # Check 1: ensure basic results are correct
    check_1 = all((i == deflate(j)).all() for i, j in zip(unpack(a), a))
    assert check_1, "Failed to unpack"

    # Check 2: ensure axis declaration is obeyed
    check_2 = all(
        (i == deflate(j)).all() for i, j in zip(unpack(a, axis=1), a.T)
    )
    assert check_2, 'Failed to respect "axis" declaration'

    # Check 3: device persistence check.
    check_3 = all(i.device == normalize_device(DEVICE) for i in unpack(a))
    assert check_3, "Device persistence check failed"
