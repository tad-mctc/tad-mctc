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
Units: Length
=============

This module contains conversions for units of length.
"""
from __future__ import annotations

from .codata import CODATA

__all__ = ["AU2METER", "METER2AU", "AA2METER", "AA2AU"]


AU2METER = CODATA.bohr
"""
Conversion from bohr (a.u.) to meter.
This equals: 1 bohr = 5.29177210903e-11 m.
"""

METER2AU = 1.0 / AU2METER
"""Conversion from meter to atomic units."""

AA2METER = 1e-10
"""Factor for conversion from Angstrom to meter (1e-10)."""

AA2AU = AA2METER * METER2AU
"""
Factor for conversion from angstrom to atomic units (bohr).
This equals: 1 Angstrom = 1.8897261246204404 a.u.
"""
