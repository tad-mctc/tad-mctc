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
Test the utility functions.
"""

from __future__ import annotations

import torch

from tad_mctc import batch


def test_real_triples_single() -> None:
    numbers = torch.tensor([8, 1, 1])  # H2O
    size = numbers.shape[0]

    ref = torch.full((size, size, size), True)
    mask = batch.real_triples(numbers, mask_diagonal=False, mask_self=False)
    assert (mask == ref).all()

    ref *= ~torch.diag_embed(torch.ones(size, dtype=torch.bool))
    mask = batch.real_pairs(numbers, mask_diagonal=True)
    assert (mask == ref).all()


def test_real_triples_batch() -> None:
    numbers = torch.tensor(
        [
            [1, 1, 0],  # H2
            [8, 1, 1],  # H2O
        ],
    )

    ref = torch.tensor(
        [
            [
                [
                    [True, True, False],
                    [True, True, False],
                    [False, False, False],
                ],
                [
                    [True, True, False],
                    [True, True, False],
                    [False, False, False],
                ],
                [
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                ],
            ],
            [
                [
                    [True, True, True],
                    [True, True, True],
                    [True, True, True],
                ],
                [
                    [True, True, True],
                    [True, True, True],
                    [True, True, True],
                ],
                [
                    [True, True, True],
                    [True, True, True],
                    [True, True, True],
                ],
            ],
        ]
    )
    mask = batch.real_triples(numbers, mask_diagonal=False, mask_self=False)
    assert (mask == ref).all()

    ref = torch.tensor(
        [
            [
                [
                    [False, True, False],
                    [True, False, False],
                    [False, False, False],
                ],
                [
                    [False, True, False],
                    [True, False, False],
                    [False, False, False],
                ],
                [
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                ],
            ],
            [
                [
                    [False, True, True],
                    [True, False, True],
                    [True, True, False],
                ],
                [
                    [False, True, True],
                    [True, False, True],
                    [True, True, False],
                ],
                [
                    [False, True, True],
                    [True, False, True],
                    [True, True, False],
                ],
            ],
        ]
    )
    mask = batch.real_triples(numbers, mask_diagonal=True, mask_self=False)
    assert (mask == ref).all()


def test_real_triples_self_single() -> None:
    numbers = torch.tensor([8, 1, 1])  # H2O

    ref = torch.tensor(
        [
            [
                [False, False, False],
                [False, False, True],
                [False, True, False],
            ],
            [
                [False, False, True],
                [False, False, False],
                [True, False, False],
            ],
            [
                [False, True, False],
                [True, False, False],
                [False, False, False],
            ],
        ],
        dtype=torch.bool,
    )

    mask = batch.real_triples(numbers, mask_self=True)
    assert (mask == ref).all()


def test_real_triples_self_batch() -> None:
    numbers = torch.tensor(
        [
            [1, 1, 0],  # H2
            [8, 1, 1],  # H2O
        ],
    )

    ref = torch.tensor(
        [
            [
                [
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                ],
                [
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                ],
                [
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                ],
            ],
            [
                [
                    [False, False, False],
                    [False, False, True],
                    [False, True, False],
                ],
                [
                    [False, False, True],
                    [False, False, False],
                    [True, False, False],
                ],
                [
                    [False, True, False],
                    [True, False, False],
                    [False, False, False],
                ],
            ],
        ]
    )

    mask = batch.real_triples(numbers, mask_self=True)
    assert (mask == ref).all()
