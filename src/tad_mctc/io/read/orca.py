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
Read: ORCA
==========

Reader for ORCA files.
"""
from __future__ import annotations

import torch

from ...exceptions import EmptyFileError, FormatErrorORCA
from ...typing import IO, Any, Tensor
from .frompath import create_path_reader


def read_orca_engrad(
    fileobj: IO[Any],
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[Tensor, Tensor]:
    """
    Read ORCA's engrad file.

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
    start_grad = -1
    grad = []

    start_energy = -1
    energy = None

    lines: list[str] = fileobj.readlines()
    for i, line in enumerate(lines):
        # energy
        if line.startswith("# The current total energy in Eh"):
            start_energy = i + 2

        if i == start_energy:
            l = line.strip()
            if len(l) == 0:
                raise FormatErrorORCA(f"No energy found in {fileobj}.")
            energy = torch.tensor(float(l), device=device, dtype=dtype)
            start_energy = -1

        # gradient
        if line.startswith("# The current gradient in Eh/bohr"):
            start_grad = i + 2

        if i == start_grad:
            # abort if we hit the next "#"
            if line.startswith("#"):
                break

            l = line.strip()
            if len(l) == 0:
                raise FormatErrorORCA(f"No gradient found in {fileobj}.")

            grad.append(float(l))
            start_grad += 1

    if energy is None:
        raise EmptyFileError(f"File '{fileobj}' appears to be empty.")

    return energy, torch.tensor(grad, device=device, dtype=dtype).reshape(-1, 3)


read_orca_engrad_from_path = create_path_reader(read_orca_engrad)
