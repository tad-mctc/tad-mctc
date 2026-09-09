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
Units: Length
=============

This module contains conversions for units of length.
"""

from __future__ import annotations

import math

from .codata import CODATA

__all__ = ["AU2METER", "METER2AU", "AA2METER", "METER2AA", "AA2AU", "AU2AA"]


# The Bohr radius is derived from Planck's constant, the electron mass, the
# speed of light and the fine-structure constant, exactly as mctc-lib (and
# hence the s-dftd3 Fortran reference) derives it, rather than taken from
# qcelemental's independently tabulated CODATA value for it. CODATA rounds
# each of these to a similar number of significant digits, so a value
# re-derived from the others does not exactly reproduce the separately
# rounded tabulation; the gap is a relative 2.8e-12, far below the constant's
# actual measurement uncertainty, but the Axilrod-Teller-Muto damping term's
# ``(r0/r)**((alp+2)/3)`` exponent amplified it into a ~2e-11 disagreement
# with the reference. Deriving it the same way removes that amplification.
_HBAR = CODATA.planck_constant / (2.0 * math.pi)

AU2METER = _HBAR / (
    CODATA.electron_mass * CODATA.speed_of_light_in_vacuum
    * CODATA.fine_structure_constant
)
"""
Conversion from bohr (a.u.) to meter.
This equals: 1 bohr = 5.291772109044924e-11 m.
"""

METER2AU = 1.0 / AU2METER
"""Conversion from meter to atomic units."""


AA2METER = 1e-10
"""Factor for conversion from Angstrom to meter (1e-10)."""

METER2AA = 1.0 / AA2METER
"""Factor for conversion from meter to Angstrom (1e10)."""


AA2AU = AA2METER * METER2AU
"""
Factor for conversion from Angstrom to atomic units (bohr).
This equals: 1 Angstrom = 1.8897261246204407 a.u.
"""

AU2AA = 1.0 / AA2AU
"""Factor for conversion from atomic units (bohr) to Angstrom."""
