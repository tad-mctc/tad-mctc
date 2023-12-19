# This file is part of tad-mctc.
#
# SPDX-Identifier: LGPL-3.0
# Copyright (C) 2023 Marvin Friede
#
# tad-mctc is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# tad_mctc is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with tad-mctc. If not, see <https://www.gnu.org/licenses/>.
"""
Atomic data: Charges
====================

This module contains the following constants:
- number of core electrons for GFN
- effective nuclear charges from the def2-ECPs (DFT-D4 reference
  polarizibilities)
- charge of the valence shell (dipole moment in GFN)
"""

import torch

__all__ = ["ECORE", "ZEFF", "ZVALENCE"]


# fmt: off
ECORE = torch.tensor([
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
])
"""Number of core electrons of all atoms."""


ZEFF = torch.tensor([
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
])
"""Effective nuclear charges."""


ZVALENCE = torch.tensor([
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
])
"""Charge of the valence shell."""
# fmt: on
