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
I/O Read: CHRG/UHF
==================

Reader for the `.CHRG` and `.UHF` files.
"""
from __future__ import annotations

import torch

from ...exceptions import EmptyFileError, FormatError
from ...typing import IO, Any, Tensor
from .frompath import create_path_reader_dotfiles

__all__ = [
    "read_chrg",
    "read_chrg_from_path",
    "read_uhf",
    "read_uhf_from_path",
    "read_spin",
    "read_spin_from_path",
]


def read_dotfile(
    fileobj: IO[Any],
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """
    Read a dotfile (`.CHRG` or `.UHF`) file.

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
        Tensor of the value in the file.

    Raises
    ------
    EmptyFileError
        File is empty (at least the first line).
    FormatError
        File does not conform with the expected format.
    """
    line: str = fileobj.readline().strip()

    if len(line) == 0:
        raise EmptyFileError(f"File '{fileobj}' is empty.")

    try:
        num = int(line)
    except ValueError as e:
        raise FormatError(
            f"Content '{line}' of '{fileobj}' does not represent an integer."
        ) from e

    return torch.tensor(num, device=device, dtype=dtype)


read_chrg = read_uhf = read_spin = read_dotfile


read_chrg_from_path = create_path_reader_dotfiles(read_dotfile, name=".CHRG")
read_uhf_from_path = create_path_reader_dotfiles(read_dotfile, name=".UHF")
read_spin_from_path = create_path_reader_dotfiles(read_dotfile, name=".UHF")
