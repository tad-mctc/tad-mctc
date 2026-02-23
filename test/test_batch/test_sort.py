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

import numpy as np
import torch

from tad_mctc.batch import pack, pargsort, psort
from tad_mctc.convert import normalize_device

from ..conftest import DEVICE


def test_sort():
    """
    Ensures that the ``psort`` and ``pargsort`` methods work as intended.

    Notes
    -----
    A separate check is not needed for the ``pargsort`` method as ``psort``
    just wraps around it.
    """
    # Test on with multiple different dimensions
    for d in range(1, 4):
        tensors = [
            torch.rand((*[i] * d,), device=DEVICE)
            for i in np.random.randint(3, 10, (10,))
        ]

        packed, mask = pack(tensors, return_mask=True)

        pred = psort(packed, mask).values
        ref = pack([i.sort().values for i in tensors])

        check_1 = (pred == ref).all()
        assert check_1, "Values were incorrectly sorted"

        check_2 = pred.device == normalize_device(DEVICE)
        assert check_2, "Device persistence check failed"

    # standard sorting
    t = torch.tensor([[1.0, 4.0, 2.0, 3.0], [1.0, 4.0, 2.0, 0.0]])
    ref = torch.tensor([[1.0, 2.0, 3.0, 4.0], [0.0, 1.0, 2.0, 4.0]])
    pred = psort(t).values
    check_3 = (pred == ref).all()
    assert check_3, "Values were incorrectly sorted"


def test_pargsort():
    """Normal `torch.argsort`."""
    t = torch.tensor(
        [
            [1.0, 4.0, 2.0, 3.0],
            [1.0, 4.0, 2.0, 0.0],
        ],
        device=DEVICE,
    )
    ref = torch.tensor(
        [
            [0, 2, 3, 1],
            [3, 0, 2, 1],
        ],
        device=DEVICE,
    )

    pred = pargsort(t)
    check_1 = (pred == ref).all()
    assert check_1, "Values were incorrectly sorted"
