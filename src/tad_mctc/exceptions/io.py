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
Exceptions: I/O
===============

Exceptions and warnings related to input and output operations.
"""

__all__ = ["EmptyFileError", "FormatError", "FormatErrorORCA", "FormatErrorTM"]


class EmptyFileError(RuntimeError):
    """Error for an empty file."""


class FormatError(RuntimeError):
    """Error for wrong format of a file."""


class FormatErrorORCA(FormatError):
    """Format error for an ORCA file."""


class FormatErrorTM(FormatError):
    """Format error for a Turbomole file."""
