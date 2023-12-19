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
