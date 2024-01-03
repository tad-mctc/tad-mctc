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
from ...units import length
from ..checks import content_checks, shape_checks
from .writer import create_path_writer

__all__ = ["write_xyz", "write_xyz_to_path"]


def write_xyz(
    fileobj: IO[Any],
    numbers: Tensor,
    positions: Tensor,
    **kwargs: Any,
) -> None:
    """
    Write atomic coordinates in XYZ format to a file object. The input
    positions are expected to be in atomic units (bohrs).

    Parameters
    ----------
    fileobj : IO[Any]
        The file-like object to which the XYZ data will be written.
    numbers : Tensor
        A 1D tensor containing atomic numbers or symbols.
    positions : Tensor
        A 2D tensor of shape (n_atoms, 3) containing atomic positions in
        atomic units (bohrs).
    fmt : str, optional
        Format string for the position coordinates.
    comment : str, optional
        A comment string for the XYZ file header.

    Raises
    ------
    ValueError
        If the comment contains line breaks.
    """
    positions = positions * length.AU2AA
    assert shape_checks(numbers, positions)
    assert content_checks(numbers, positions)

    fmt = kwargs.pop("fmt", "%20.14f")
    comment = kwargs.pop("comment", "").rstrip()

    if "\n" in comment:
        raise ValueError("Comment line should not have line breaks.")

    fileobj.write(f"{len(numbers)}\n{comment}\n")
    for num, pos in zip(numbers, positions):
        symbol = pse.Z2S[int(num.item())].title()
        fileobj.write(f"{symbol:<2} {fmt % pos[0]} {fmt % pos[1]} {fmt % pos[2]}\n")


write_xyz_to_path = create_path_writer(write_xyz)
