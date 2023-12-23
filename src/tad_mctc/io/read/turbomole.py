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
Read: Turbomole
===============

Reader for Turbomole coordinate files (*coord*).
"""
from __future__ import annotations

import torch

from ...batch import pack
from ...data import pse
from ...typing import DD, IO, Any, get_default_dtype
from ..checks import content_checks, shape_checks
from .reader import create_path_reader

__all__ = [
    "read_coord",
    "read_turbomole",
    "read_coord_from_path",
    "read_turbomole_from_path",
]


def read_turbomole(
    fileobj: IO[Any],
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    dtype_int: torch.dtype = torch.long,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Reads a Turbomole coord file and returns atomic numbers and positions as
    tensors.

    Parameters
    ----------
    fileobj : IO[Any]
        The file-like object to read from.
    device : torch.device | None, optional
        Device to store the tensor on. Defaults to `None`.
    dtype : torch.dtype | None, optional
        Floating point data type of the tensor. Defaults to `None`.
    dtype_int : torch.dtype, optional
        Integer data type of the tensor. Defaults to `torch.long`.

    Returns
    -------
    (Tensor, Tensor)
        Tensors of atomic numbers and positions. Positions is a tensor of shape
        (nat, 3) in atomic units.
    """
    # Initialize as None
    numbers = None
    positions = None

    # data types
    dd: DD = {
        "device": device,
        "dtype": dtype if dtype is not None else get_default_dtype(),
    }
    ddi: DD = {"device": device, "dtype": dtype_int}

    lines = fileobj.readlines()

    # Find $coord section
    # does not necessarily have to be the first $<something> in file...
    start = -1
    for i, line in enumerate(lines):
        if line.strip().startswith("$coord"):
            start = i + 1
            break

    if start == -1:
        raise ValueError("No $coord section found in file.")

    # Parse coord section
    i = start
    while i < len(lines):
        # Start of new section or end
        if lines[i].startswith("$"):
            break

        symbols = []
        coords = []
        while i < len(lines) and not lines[i].startswith("$"):
            x, y, z, symbol = lines[i].split()[:4]
            symbols.append(symbol.title().replace("Q", "X"))
            coords.append([float(x), float(y), float(z)])
            i += 1

        numbers = torch.tensor([pse.S2Z[symbol] for symbol in symbols], **ddi)
        positions = torch.tensor(coords, **dd)

        assert shape_checks(numbers, positions)
        assert content_checks(numbers, positions)

    # Check if data was actually parsed
    if numbers is None or positions is None:
        raise ValueError("No valid data found in the file.")

    return numbers, positions


read_turbomole_from_path = create_path_reader(read_turbomole)

read_coord = read_turbomole

read_coord_from_path = create_path_reader(read_turbomole)
