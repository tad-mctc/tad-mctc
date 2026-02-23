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

from .codata import get_constant

__all__ = ["AU2METER", "METER2AU", "AA2METER", "METER2AA", "AA2AU", "AU2AA"]


AU2METER = get_constant("bohr radius")
"""
Conversion from bohr (a.u.) to meter.
This equals: 1 bohr = 5.29177210903e-11 m.
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
This equals: 1 Angstrom = 1.8897261246204404 a.u.
"""

AU2AA = 1.0 / AA2AU
"""Factor for conversion from atomic units (bohr) to Angstrom."""
