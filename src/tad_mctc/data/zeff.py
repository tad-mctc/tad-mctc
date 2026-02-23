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
Data: Charges
=============

This module contains the following constants:
- number of core electrons for GFN
- effective nuclear charges from the def2-ECPs (DFT-D4 reference polarizibilities)
- charge of the valence shell (dipole moment in GFN)
"""

from __future__ import annotations

import torch

__all__ = ["ECORE", "ZEFF", "ZVALENCE"]


def ECORE(
    device: torch.device | None = None, dtype: torch.dtype = torch.int8
) -> torch.Tensor:
    """Number of core electrons."""
    # fmt: off
    _ECORE = [
        0,                                                         # dummy
        0,  0,                                                     # 1-2
        2,  2,  2,  2,  2,  2,  2,  2,                             # 3-10
        10, 10, 10, 10, 10, 10, 10, 10,                             # 11-18
        18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18,                 # 19-29
        28, 28, 28, 28, 28, 28, 28,                                 # 30-36
        36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36,                 # 37-47
        46, 46, 46, 46, 46, 46, 46,                                 # 48-54
        54, 54,                                                     # 55-56
        54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, # 57-71
        68, 68, 68, 68, 68, 68, 68, 68,                             # 72-79
        78, 78, 78, 78, 78, 78, 78,                                 # 80-86
    ]

    # fmt: on
    return torch.tensor(_ECORE, dtype=dtype, device=device, requires_grad=False)


def ZEFF(
    device: torch.device | None = None, dtype: torch.dtype = torch.int8
) -> torch.Tensor:
    """Effective nuclear charges from the def2-ECPs."""

    # fmt: off
    _ZEFF = [
        0,                                                      # None
        1,                                                 2,   # H-He
        3, 4,                               5, 6, 7, 8, 9,10,   # Li-Ne
        11,12,                              13,14,15,16,17,18,   # Na-Ar
        19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,   # K-Kr
        9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,   # Rb-Xe
        9,10,11,30,31,32,33,34,35,36,37,38,39,40,41,42,43,      # Cs-Lu
        12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,            # Hf-Rn
        # just copy and paste from above
        9,10,11,30,31,32,33,34,35,36,37,38,39,40,41,42,43,      # Fr-Lr
        12,13,14,15,16,17,18,19,20,21,22,23,24,25,26             # Rf-Og
    ]
    # fmt: on

    return torch.tensor(_ZEFF, dtype=dtype, device=device, requires_grad=False)


def ZVALENCE(
    device: torch.device | None = None, dtype: torch.dtype = torch.int8
) -> torch.Tensor:
    """Charge of the valence shell."""

    # fmt: off
    _ZVALENCE = [
        0,                                                         # dummy
        1,  2,                                                     # 1-2
        1,  2,  3,  4,  5,  6,  7,  8,                             # 3-10
        1,  2,  3,  4,  5,  6,  7,  8,                             # 11-18
        1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11,                 # 19-29
        2,  3,  4,  5,  6,  7,  8,                                 # 30-36
        1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11,                 # 37-47
        2,  3,  4,  5,  6,  7,  8,                                 # 48-54
        1,  2,                                                     # 55-56
        3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3, # 56-71
        4,  5,  6,  7,  8,  9, 10, 11,                             # 72-79
        2,  3,  4,  5,  6,  7,  8,                                 # 80-86
    ]
    # fmt: on
    return torch.tensor(
        _ZVALENCE, dtype=dtype, device=device, requires_grad=False
    )
