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
Coordination number: Defaults
=============================

Parameters for the counting functions and default cutoffs.
"""

__all__ = [
    "KCN_EEQ",
    "KCN_D3",
    "KA",
    "KB",
    "R_SHIFT",
    "CUTOFF_D3",
    "CUTOFF_D4",
    "CUTOFF_EEQ",
    "CUTOFF_EEQ_MAX",
    "CUTOFF_EEQBC",
    "CUTOFF_GFN2",
    "KCN_EEQ_EN",
    "KCN_D4",
    "D4_K4",
    "D4_K5",
    "D4_K6",
]

KCN_EEQ = 7.5
"""Steepness of counting function in EEQ model (7.5)."""

KCN_D3 = 16.0
"""Steepness of counting function in GFN1."""

KCN_EEQ_EN = 2.60
"""Steepness of counting function in the EN-weighted EEQ model (2.60)."""

KA = 10.0
"""Steepness of first counting function in GFN2."""

KB = 20.0
"""Steepness of second counting function in GFN2."""

R_SHIFT = 2.0
"""Offset of the second counting function in GFN2."""


CUTOFF_D3 = 25.0
"""Coordination number cutoff within DFT-D3 (25.0)."""

CUTOFF_D4 = 30.0
"""Coordination number cutoff (30.0)."""

CUTOFF_EEQ = 25.0
"""Coordination number cutoff within EEQ (25.0)."""

CUTOFF_EEQ_MAX = 8.0
"""Maximum coordination number (8.0)."""

CUTOFF_EEQBC = 25.0
"""Coordination number cutoff within EEQBC (25.0)."""

CUTOFF_GFN2 = 25.0
"""Coordination number cutoff within GFN2-xTB (25.0)."""


KCN_D4 = 7.5
"""Steepness of counting function (7.5)."""

D4_K4 = 4.10451
"""Parameter for electronegativity scaling."""

D4_K5 = 19.08857
"""Parameter for electronegativity scaling."""

D4_K6 = 2 * 11.28174**2  # 254.56
"""Parameter for electronegativity scaling."""
