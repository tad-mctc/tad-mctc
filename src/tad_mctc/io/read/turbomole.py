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
I/O Read: Turbomole
===================

Reader for Turbomole coordinate (*coord*), energy, and gradient files.
"""

from __future__ import annotations

import torch

from ...data import pse
from ...exceptions import EmptyFileError, FormatErrorTM
from ...typing import DD, IO, Any, Tensor, get_default_dtype
from ..checks import content_checks, deflatable_check, shape_checks
from .frompath import create_path_reader

__all__ = ["read_coord", "read_turbomole", "read_turbomole_energy"]


def read_turbomole_fileobj(
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
    device : :class:`torch.device` | None, optional
        Device to store the tensor on. Defaults to `None`.
    dtype : :class:`torch.dtype` | None, optional
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

        assert shape_checks(numbers, positions, allow_batched=False)
        assert content_checks(numbers, positions, allow_batched=False)
        assert deflatable_check(positions, fileobj, **kwargs)

    # Check if data was actually parsed
    if numbers is None or positions is None:
        raise EmptyFileError("No valid data found in the file.")

    return numbers, positions


read_turbomole = create_path_reader(read_turbomole_fileobj)


read_coord = create_path_reader(read_turbomole_fileobj)


################################################################################


def read_turbomole_energy_fileobj(
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
    device : :class:`torch.device` | None, optional
        Device to store the tensor on. Defaults to `None`.
    dtype : :class:`torch.dtype` | None, optional
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


read_turbomole_energy = create_path_reader(read_turbomole_energy_fileobj)
