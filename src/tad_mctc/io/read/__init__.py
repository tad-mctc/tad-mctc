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
I/O: Read
=========

This module contains functions for reading files.

The simplest interface is provided by the `read` function, which tries to infer
the file type from the file name or extension.

Example
-------
>>> from tad_mctc.io import read
>>> path = "mol.xyz"
>>> numbers, positions = read.read(path)
"""
from .dotfiles import *
from .frompath import *
from .orca import *
from .qcschema import *
from .reader import *
from .tblite import *
from .turbomole import *
from .xyz import *
