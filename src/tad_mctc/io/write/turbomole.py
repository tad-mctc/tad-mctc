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
I/O Write: XYZ
==============

Writer for standard XYZ files.
See https://en.wikipedia.org/wiki/XYZ_file_format.
"""
from __future__ import annotations

from ...data import pse
from ...typing import IO, Any, Tensor
from ..checks import content_checks, shape_checks
from .writer import create_path_writer

__all__ = ["write_turbomole", "write_turbomole_to_path"]


def write_turbomole(
    fileobj: IO[Any], numbers: Tensor, positions: Tensor, **kwargs: Any
) -> None:
    """
    Write atomic coordinates in Turbomole format to a file object.
    The input positions are expected to be in atomic units (bohrs).

    Parameters
    ----------
    fileobj : IO[Any]
        The file-like object to which the Turbomole data will be written.
    numbers : Tensor
        A 1D tensor containing atomic numbers.
    positions : Tensor
        A 2D tensor of shape (n_atoms, 3) containing atomic positions in atomic
        units (bohrs).
    fmt : str, optional
        Format string for the position coordinates.
    """
    assert shape_checks(numbers, positions)
    assert content_checks(numbers, positions)

    fmt = kwargs.pop("fmt", "%20.14f")

    fileobj.write("$coord\n")

    for num, pos in zip(numbers.tolist(), positions.tolist()):
        symbol = pse.Z2S[num]

        # Convert 'X' to 'Q' for Turbomole ghost atoms
        symbol = symbol if symbol != "X" else "Q"

        fileobj.write(
            f"{fmt % pos[0]}  {fmt % pos[1]}  {fmt % pos[2]}      {symbol.lower()} \n"
        )

    fileobj.write("$end\n")


write_turbomole_to_path = create_path_writer(write_turbomole)
