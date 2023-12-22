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
Units: Mass
===========

This module contains conversions for units of mass.
"""
from .codata import CODATA, get_constant

# SI

AU2KG = CODATA.me
"""Atomic unit of mass (electron mass) to kilograms."""

KG2AU = 1.0 / AU2KG
"""Kilograms to atomic units of mass (electron mass)."""

# other

AMU2KG = get_constant("unified atomic mass unit")
"""Atomic mass unit (amu, Dalton) to kilograms."""

KG2AMU = 1.0 / AMU2KG
"""Kilograms to atomic mass units (amu, Dalton)."""


AMU2AU = AMU2KG / AU2KG
"""Atomic mass unit (amu) to atomic units of mass (electron mass)"""

AU2AMU = 1.0 / AMU2AU
"""Atomic units of mass (electron mass) to atomic mass unit (amu)"""


AU2GMOL = 1e3 * CODATA.me * CODATA.na
"""Atomic units of mass (electron mass, au) to grams per mole (g/mol)"""

GMOL2AU = 1.0 / AU2GMOL
"""Grams per mole (g/mol) to atomic units of mass (electron mass, au)"""

print(AMU2KG, KG2AMU, AMU2AU, AU2AMU, AU2GMOL, GMOL2AU)
