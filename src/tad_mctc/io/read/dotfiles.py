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
I/O Read: CHRG/UHF
==================

Reader for the `.CHRG` and `.UHF` files.
"""

from __future__ import annotations

import torch

from ...exceptions import EmptyFileError, FormatError
from ...typing import IO, Any, Tensor
from .frompath import create_path_reader_dotfiles

__all__ = ["read_chrg", "read_uhf", "read_spin"]


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
    device : :class:`torch.device` | None, optional
        Device to store the tensor on. Defaults to `None`.
    dtype : :class:`torch.dtype` | None, optional
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


read_chrg = create_path_reader_dotfiles(read_dotfile, name=".CHRG")
"""
Read a ``.CHRG`` file.

Parameters
----------
file : PathLike
    Path of file containing the structure.
mode : str, optional
    Mode in which the file is opened. Defaults to ``"r"``.
encoding : str, optional
    Encoding for file. Defaults to ``"utf-8"``.
device : :class:`torch.device` | None, optional
    Device to store the tensor on. Defaults to ``None``.
dtype : :class:`torch.dtype` | None, optional
    Floating point data type of the tensor. Defaults to ``None``.

Returns
-------
Tensor
    Value stored in the file as tensor.

Raises
------
EmptyFileError
    File is empty (at least the first line).
FormatError
    File does not conform with the expected format.
"""

read_uhf = create_path_reader_dotfiles(read_dotfile, name=".UHF")
"""
Read a ``.UHF`` file.

Parameters
----------
file : PathLike
    Path of file containing the structure.
mode : str, optional
    Mode in which the file is opened. Defaults to ``"r"``.
encoding : str, optional
    Encoding for file. Defaults to ``"utf-8"``.
device : :class:`torch.device` | None, optional
    Device to store the tensor on. Defaults to ``None``.
dtype : :class:`torch.dtype` | None, optional
    Floating point data type of the tensor. Defaults to ``None``.

Returns
-------
Tensor
    Value stored in the file as tensor.

Raises
------
EmptyFileError
    File is empty (at least the first line).
FormatError
    File does not conform with the expected format.
"""

read_spin = create_path_reader_dotfiles(read_dotfile, name=".UHF")
"""
Read a ``.UHF`` file.

Parameters
----------
file : PathLike
    Path of file containing the structure.
mode : str, optional
    Mode in which the file is opened. Defaults to ``"r"``.
encoding : str, optional
    Encoding for file. Defaults to ``"utf-8"``.
device : :class:`torch.device` | None, optional
    Device to store the tensor on. Defaults to ``None``.
dtype : :class:`torch.dtype` | None, optional
    Floating point data type of the tensor. Defaults to ``None``.

Returns
-------
Tensor
    Value stored in the file as tensor.

Raises
------
EmptyFileError
    File is empty (at least the first line).
FormatError
    File does not conform with the expected format.
"""
