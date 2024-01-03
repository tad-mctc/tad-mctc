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

Reader for Turbomole coordinate (*coord*), energy, and gradient files.
"""
from __future__ import annotations

import torch

from ...data import pse
from ...exceptions import EmptyFileError, FormatErrorTM
from ...typing import DD, IO, Any, Tensor, get_default_dtype
from ..checks import content_checks, deflatable_check, shape_checks
from .frompath import create_path_reader

__all__ = [
    "read_coord",
    "read_turbomole",
    "read_coord_from_path",
    "read_turbomole_from_path",
    #
    "read_turbomole_energy",
    "read_turbomole_energy_from_path",
]


def read_turbomole(
    fileobj: IO[Any],
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    dtype_int: torch.dtype = torch.long,
    **kwargs: Any,
) -> tuple[Tensor, Tensor]:
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

    lines: list[str] = fileobj.readlines()

    # Find $coord section
    # does not necessarily have to be the first $<something> in file...
    start = -1
    for i, line in enumerate(lines):
        if line.strip().startswith("$coord"):
            start = i + 1
            break

    if start == -1:
        raise FormatErrorTM(f"No $coord section found in file '{fileobj}'.")

    # Parse coord section
    i = start
    while i < len(lines):
        # Start of new section or end
        if lines[i].startswith("$"):
            break

        symbols: list[str] = []
        coords: list[list[float]] = []
        while i < len(lines) and not lines[i].startswith("$"):
            try:
                x, y, z, symbol = lines[i].split()[:4]
            except Exception as e:
                raise FormatErrorTM(f"Cannot read file '{fileobj}'") from e

            symbols.append(symbol.title().replace("Q", "X"))
            coords.append([float(x), float(y), float(z)])
            i += 1

        numbers = torch.tensor([pse.S2Z[symbol] for symbol in symbols], **ddi)
        positions = torch.tensor(coords, **dd)

        assert shape_checks(numbers, positions)
        assert content_checks(numbers, positions)
        assert deflatable_check(positions, fileobj, **kwargs)

    # Check if data was actually parsed
    if numbers is None or positions is None:
        raise EmptyFileError("No valid data found in the file.")

    return numbers, positions


read_turbomole_from_path = create_path_reader(read_turbomole)

read_coord = read_turbomole

read_coord_from_path = create_path_reader(read_turbomole)


################################################################################


def read_turbomole_energy(
    fileobj: IO[Any],
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """
    Read energy file in TM format (energy is three times on second line).

    Parameters
    ----------
    fileobj : IO[Any]
        The file-like object to read from.
    device : torch.device | None, optional
        Device to store the tensor on. Defaults to `None`.
    dtype : torch.dtype | None, optional
        Floating point data type of the tensor. Defaults to `None`.

    Returns
    -------
    Tensor
        Tensor of energy

    Raises
    ------
    ValueError
        File does not conform with the expected format.
    """
    first_line: str = fileobj.readline().strip()
    if not first_line or first_line.split()[0] != "$energy":
        raise FormatErrorTM(f"File '{fileobj}' is not in Turbomole format.")

    second_line: str = fileobj.readline().strip()
    if not second_line:
        raise FormatErrorTM(f"File '{fileobj}' is not in Turbomole format.")

    energy_line = second_line.split()
    if len(energy_line) != 4:
        raise FormatErrorTM(f"File '{fileobj}' is not in Turbomole format.")

    return torch.tensor(float(energy_line[1]), device=device, dtype=dtype)


read_turbomole_energy_from_path = create_path_reader(read_turbomole_energy)
