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
Data: Electronegativities
=========================

Pauling electronegativities, used for the covalent coordination number.
"""
import torch

__all__ = ["PAULING"]


# fmt: off
PAULING = torch.tensor([
    0.00,  # None
    2.20,3.00,  # H,He
    0.98,1.57,2.04,2.55,3.04,3.44,3.98,4.50,  # Li-Ne
    0.93,1.31,1.61,1.90,2.19,2.58,3.16,3.50,  # Na-Ar
    0.82,1.00,  # K,Ca
    1.36,1.54,1.63,1.66,1.55,  # Sc-
    1.83,1.88,1.91,1.90,1.65,  # -Zn
    1.81,2.01,2.18,2.55,2.96,3.00,  # Ga-Kr
    0.82,0.95,  # Rb,Sr
    1.22,1.33,1.60,2.16,1.90,  # Y-
    2.20,2.28,2.20,1.93,1.69,  # -Cd
    1.78,1.96,2.05,2.10,2.66,2.60,  # In-Xe
    0.79,0.89,  # Cs,Ba
    1.10,1.12,1.13,1.14,1.15,1.17,1.18,  # La-Eu
    1.20,1.21,1.22,1.23,1.24,1.25,1.26,  # Gd-Yb
    1.27,1.30,1.50,2.36,1.90,  # Lu-
    2.20,2.20,2.28,2.54,2.00,  # -Hg
    1.62,2.33,2.02,2.00,2.20,2.20,  # Tl-Rn
    0.79,0.90,  # Fr,Ra
    1.10,1.30,1.50,1.38,1.36,1.28,1.30,  # Ac-Am
    1.30,1.30,1.30,1.30,1.30,1.30,1.30,  # Cm-No
    1.30, # Lr
    # only dummies below
    1.50,1.50,1.50,1.50,  # Rf-
    1.50,1.50,1.50,1.50,1.50,  # Rf-Cn
    1.50,1.50,1.50,1.50,1.50,1.50  # Nh-Og
])
# fmt: on
"""Pauling electronegativities, used for the covalent coordination number."""
