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
Units: Mass
===========

This module contains conversions for units of mass.
"""

from __future__ import annotations

from .codata import CODATA, get_constant

# fmt: off
__all__ = [
    "AU2KG", "KG2AU",
    "AMU2KG", "KG2AMU",
    "AMU2AU", "AU2AMU",
    "AU2GMOL", "GMOL2AU"
]
# fmt: on


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
