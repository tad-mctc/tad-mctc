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
Units: Spectroscopy
===================

This module contains conversions for units used in spectroscopy.
"""

from __future__ import annotations

from .codata import CODATA
from .energy import AU2JOULE, COULOMB2AU
from .length import AU2AA, METER2AU
from .mass import AU2AMU

__all__ = [
    "AU2RCM",
    "RCM2AU",
    "DEBYE2AU",
    "AU2DEBYE",
    # IR
    "AU2KMMOL",
    "KMMOL2AU",
    "AU2DAAAMU",
    "DAAAMU2AU",
    # Raman
    "AU2AA4AMU",
    "AA4AMU2AU",
]


AU2RCM = AU2JOULE / (CODATA.h * CODATA.c) * 1e-2
"""
Conversion from hartree (atomic units of energy) to reciprocal centimeters
(cm⁻¹). Used in spectroscopy to express wavenumbers.
"""

RCM2AU = 1.0 / AU2RCM
"""
Conversion from reciprocal centimeters (cm⁻¹) to hartree (atomic units of
energy).
"""


DEBYE2AU = 1e-21 / CODATA.c * COULOMB2AU * METER2AU
"""
Conversion from Debye (unit of electric dipole moment) to atomic units.
1 Debye in SI units (C·m) = 0.393430 a.u.
"""

AU2DEBYE = 1.0 / DEBYE2AU
"""Conversion from atomic units to Debye."""

# IR

AU2DAAAMU = (AU2DEBYE / AU2AA) ** 2 / AU2AMU
"""Conversion for IR intensity from atomic units to Debye^2/Angstrom^2/amu."""

DAAAMU2AU = 1.0 / AU2DAAAMU
"""Conversion for IR intensity from Debye^2/Angstrom^2/amu to atomic units."""

AU2KMMOL = AU2DAAAMU * 42.256
"""
Conversion factor for IR intensity from atomic units to km/mol. This involves
converting the square of the dipole moment from Debye to atomic units, and
adjusting for mass and the standard reporting units of IR intensity.
"""

KMMOL2AU = 1.0 / AU2KMMOL
"""Conversion factor for IR intensity from km/mol to atomic units."""

# RAMAN

AU2AA4AMU = AU2AA**4 / AU2AMU
"""Conversion for Raman intensity from atomic units to Angstrom^4/amu."""

AA4AMU2AU = 1.0 / AU2AA4AMU
"""Conversion for Raman intensity from Angstrom^4/amu to atomic units."""
