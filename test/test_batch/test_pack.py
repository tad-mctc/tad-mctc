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
Test the packing utility functions.
"""
from __future__ import annotations

import torch

from tad_mctc.batch import pack

mol1 = torch.tensor([1, 1])  # H2
mol2 = torch.tensor([8, 1, 1])  # H2O


def test_single_tensor() -> None:
    # dummy test: only give single tensor
    assert (mol1 == pack(mol1)).all()


def test_standard() -> None:
    # standard packing
    ref = torch.tensor(
        [
            [1, 1, 0],  # H2
            [8, 1, 1],  # H2O
        ],
    )
    packed = pack([mol1, mol2])
    assert (packed == ref).all()


def test_axis() -> None:
    ref = torch.tensor(
        [
            [1, 1, 0],  # H2
            [8, 1, 1],  # H2O
        ],
    )

    # different axis
    packed = pack([mol1, mol2], axis=-1)
    assert (packed == ref.T).all()


def test_size() -> None:
    ref = torch.tensor(
        [
            [1, 1, 0, 0],  # H2
            [8, 1, 1, 0],  # H2O
        ],
    )

    # one additional column of padding
    packed = pack([mol1, mol2], size=[4])
    assert (packed == ref).all()


def test_return_mask() -> None:
    packed, mask = pack(
        [
            torch.tensor([1.0]),
            torch.tensor([2.0, 2.0]),
            torch.tensor([3.0, 3.0, 3.0]),
        ],
        return_mask=True,
    )

    ref_packed = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [2.0, 2.0, 0.0],
            [3.0, 3.0, 3.0],
        ]
    )

    ref_mask = torch.tensor(
        [
            [True, False, False],
            [True, True, False],
            [True, True, True],
        ]
    )

    assert (packed == ref_packed).all()
    assert (mask == ref_mask).all()


def test_return_mask_axis() -> None:
    packed, mask = pack(
        [
            torch.tensor([1.0]),
            torch.tensor([2.0, 2.0]),
            torch.tensor([3.0, 3.0, 3.0]),
        ],
        axis=-1,
        return_mask=True,
    )

    ref_packed = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [2.0, 2.0, 0.0],
            [3.0, 3.0, 3.0],
        ]
    )

    ref_mask = torch.tensor(
        [
            [True, False, False],
            [True, True, False],
            [True, True, True],
        ]
    )

    # different axis
    assert (packed == ref_packed.T).all()
    assert (mask == ref_mask.T).all()
