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

from tad_mctc.batch.mask import real_atoms
from tad_mctc.batch.mask.jit import real_atoms_traced


def test_real_atoms() -> None:
    numbers = torch.tensor(
        [
            [1, 1, 0, 0, 0],  # H2
            [6, 1, 1, 1, 1],  # CH4
        ],
    )
    ref = torch.tensor(
        [
            [True, True, False, False, False],  # H2
            [True, True, True, True, True],  # CH4
        ],
    )

    mask = real_atoms(numbers)
    assert (mask == ref).all()

    jmask = real_atoms_traced(numbers)
    assert (mask == jmask).all()
