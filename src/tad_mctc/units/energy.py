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
Units: Energy
=============

This module contains conversions for units of energy.
"""

from __future__ import annotations

from .codata import CODATA, get_constant
from .length import AA2AU

# fmt: off
__all__ = [
    "AU2COULOMB", "COULOMB2AU",
    "AU2EV", "EV2AU",
    "AU2JOULE", "JOULE2AU",
    "AU2KCAL", "KCAL2AU",
    "AU2KCALMOL", "KCALMOL2AU",
    "AU2KELVIN", "KELVIN2AU",
    "AU2VAA", "VAA2AU",
    "AU2VOLT", "VOLT2AU",
    #
    "JOULE2CAL", "CAL2JOULE",
    "JOULE2KCAL", "KCAL2JOULE",
    "JOULE2EV", "EV2JOULE",
    "JOULE2KELVIN", "KELVIN2JOULE",
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

AU2KCALMOL = AU2KCAL * CODATA.na
"""Factor for conversion from atomic units to kilo Calorie per mole."""

KCALMOL2AU = 1.0 / AU2KCALMOL
"""Factor for conversion from kilo Calorie per mole to atomic units."""


EV2JOULE = get_constant("electron volt-joule relationship")
"""
Factor for conversion from electron volts to Joule.
This equals the elementary charge.
"""

JOULE2EV = 1.0 / EV2JOULE
"""Factor for conversion from Joule to electron volts."""

EV2AU = EV2JOULE * JOULE2AU
"""
Factor for conversion from eletronvolt to atomic units.
(electron volt-hartree relationship)
"""

AU2EV = 1.0 / EV2AU
"""Factor for conversion from atomic units to eletronvolt."""


KELVIN2JOULE = CODATA.kb
"""
Factor for conversion from Kelvin to Joule.
This equals the Boltzmann constant.
"""

JOULE2KELVIN = 1.0 / KELVIN2JOULE
"""Factor for conversion from Joule to Kelvin."""


KELVIN2AU = get_constant("kelvin-hartree relationship")
"""Factor for conversion from Kelvin to atomic units."""

AU2KELVIN = get_constant("hartree-kelvin relationship")
"""Factor for conversion from Kelvin to atomic units."""


AU2COULOMB = get_constant("elementary charge")
"""
Factor for conversion from atomic units to Coulomb.
This equals the elementary charge.
"""

COULOMB2AU = 1.0 / AU2COULOMB
"""Factor for conversion from Coulomb to atomic units."""

VOLT2AU = JOULE2AU / COULOMB2AU
"""Factor for conversion from V = J/C to atomic units."""

AU2VOLT = 1.0 / VOLT2AU
"""Factor for conversion from atomic units to V = J/C."""

VAA2AU = VOLT2AU / AA2AU
"""Factor for conversion from V/Å = J/(C·Å) to atomic units."""

AU2VAA = 1.0 / VAA2AU
"""Factor for conversion from atomic units to V/Å = J/(C·Å)."""
