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
Units: Time
===========

This module contains conversions for units of time.
"""
from __future__ import annotations

from .codata import get_constant

__all__ = ["AU2SECOND", "SECOND2AU"]


AU2SECOND = get_constant("atomic unit of time")
"""Conversion from atomic units to seconds. The atomic unit of time (s)."""

SECOND2AU = 1.0 / AU2SECOND
"""Conversion from seconds to atomic units of time."""
