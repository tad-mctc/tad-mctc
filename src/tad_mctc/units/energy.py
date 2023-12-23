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
Units: Energy
=============

This module contains conversions for units of energy.
"""
from __future__ import annotations

from .codata import CODATA, get_constant
from .length import AA2AU

# fmt: off
__all__ = [
    "AU2JOULE", "JOULE2AU",
    "CAL2JOULE", "JOULE2CAL",
    "KCAL2JOULE", "JOULE2KCAL",
    "AU2KCAL", "KCAL2AU",
    "EV2JOULE", "JOULE2EV",
    "EV2AU", "AU2EV",
    "K2JOULE", "K2AU",
    "AU2COULOMB", "COULOMB2AU",
    "VOLT2AU", "AU2VOLT",
    "VAA2AU", "AU2VAA",
]
# fmt: on


# SI

AU2JOULE = get_constant("atomic unit of energy")
"""Factor for conversion from atomic units to Joule (J = kg·m²·s⁻²)."""

JOULE2AU = 1.0 / AU2JOULE
"""
Factor for conversion from Joule (J = kg·m²·s⁻²) to atomic units.
This equals: 1 Joule = 2.294e+17 Hartree.
Could also be calculated as: CODATA.me * CODATA.c**2 * CODATA.alpha**2
"""

# other

CAL2JOULE = 4.184
"""
Factor for conversion from Calorie to Joule.
This equals: 1 calorie = 4.184 joules.
"""

JOULE2CAL = 1.0 / CAL2JOULE
"""Factor for conversion from Joule to Calorie."""

KCAL2JOULE = CAL2JOULE * 1000
"""Factor for conversion from kilo Calorie to Joule."""

JOULE2KCAL = 1.0 / KCAL2JOULE
"""Factor for conversion from Joule to kilo Calorie."""

AU2KCAL = AU2JOULE * JOULE2KCAL
"""Factor for conversion from atomic units to kilo Calorie."""

KCAL2AU = 1.0 / AU2KCAL
"""Factor for conversion from kilo Calorie to atomic units."""


EV2JOULE = CODATA.e
"""
Factor for conversion from electron volts to Joule.
This equals the elementary charge.
"""

JOULE2EV = 1.0 / EV2JOULE
"""Factor for conversion from Joule to electron volts."""

EV2AU = EV2JOULE * JOULE2AU
"""Factor for conversion from eletronvolt to atomic units."""

AU2EV = 1.0 / EV2AU
"""Factor for conversion from atomic units to eletronvolt."""


K2JOULE = CODATA.kb
"""
Factor for conversion from Kelvin to Joule.
This equals the Boltzmann constant.
"""

K2AU = 3.166808578545117e-06
"""Factor for conversion from Kelvin to atomic units."""


AU2COULOMB = CODATA.e
"""
Factor for conversion from atomic units to Coulomb.
This equals the elementary charge.
"""

COULOMB2AU = 1.0 / AU2COULOMB
"""Factor for conversion from Coulomb to atomic units."""

VOLT2AU = JOULE2AU * COULOMB2AU
"""Factor for conversion from V = J/C to atomic units."""

AU2VOLT = 1.0 / VOLT2AU
"""Factor for conversion from atomic units to V = J/C."""

VAA2AU = VOLT2AU / AA2AU
"""Factor for conversion from V/Å = J/(C·Å) to atomic units."""

AU2VAA = 1.0 / VAA2AU
"""Factor for conversion from atomic units to V/Å = J/(C·Å)."""
