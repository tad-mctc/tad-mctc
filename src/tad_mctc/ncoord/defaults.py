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
Parameters
==========

Parameters for the counting functions and default cutoffs.
"""

KCN_EEQ = 7.5
"""Steepness of counting function in EEQ model (7.5)."""

KCN_D3 = 16.0
"""GFN1: Steepness of counting function."""

KA = 10.0
"""GFN2: Steepness of first counting function."""

KB = 20.0
"""GFN2: Steepness of second counting function."""

R_SHIFT = 2.0
"""GFN2: Offset of the second counting function."""


CUTOFF_D3 = 25.0
"""Coordination number cutoff within DFT-D3 (25.0)."""

CUTOFF_D4 = 30.0
"""Coordination number cutoff (30.0)."""

CUTOFF_EEQ = 25.0
"""Coordination number cutoff within EEQ (25.0)."""

CUTOFF_EEQ_MAX = 8.0
"""Maximum coordination number (8.0)."""


KCN_D4 = 7.5
"""Steepness of counting function (7.5)."""

D4_K4 = 4.10451
"""Parameter for electronegativity scaling."""

D4_K5 = 19.08857
"""Parameter for electronegativity scaling."""

D4_K6 = 2 * 11.28174**2  # 254.56
"""Parameter for electronegativity scaling."""
