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
Units: Spectroscopy
===================

This module contains conversions for units usd in spectroscopy.
"""
from __future__ import annotations

from .codata import CODATA
from .energy import AU2JOULE, COULOMB2AU
from .length import AA2AU, METER2AU
from .mass import AU2AMU

__all__ = ["AU2RCM", "RCM2AU", "DEBYE2AU", "AU2DEBYE", "AU2KMMOL", "KMMOL2AU"]


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


AU2KMMOL = (DEBYE2AU / AA2AU) ** 2 / AU2AMU * 42.256
"""
Conversion factor for IR intensity from atomic units to km/mol. This involves
converting the square of the dipole moment from Debye to atomic units, and
adjusting for mass and the standard reporting units of IR intensity.
"""

KMMOL2AU = 1.0 / AU2KMMOL
"""Conversion factor for IR intensity from km/mol to atomic units."""
