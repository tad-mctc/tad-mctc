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
Data
====

This module contains various arrays with atomic data:
 - Pauling electronegativities
 - masses
 - covalent radii
 - effective nuclear charges
 - periodic table

Note that the first element of all tensors is a dummy to allow indexing by the
atomic numbers.
"""
from .en import *
from .mass import *
from .pse import *
from .radii import *
from .zeff import *
