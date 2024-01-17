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
I/O Read: From Path
===================

A convenience function to create readers that take a path instead of a stream.

Example
-------
>>> from tad_mctc.io import read
>>> read_xyz_from_path = read.create_path_reader(read.read_xyz)
>>> path = ...
>>> numbers, positions = read_xyz_from_path(path)
"""
from __future__ import annotations

from pathlib import Path
from typing import runtime_checkable

import torch

from ...typing import IO, Any, Literal, PathLike, Protocol, Tensor

__all__ = ["create_path_reader", "create_path_reader_dotfiles"]


@runtime_checkable
class ReaderFunction(Protocol):
    def __call__(
        self,
        fileobj: IO[Any],
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor | tuple[Tensor, Tensor]:
        ...


@runtime_checkable
class FileReaderFunction(Protocol):
    def __call__(
        self,
        filepath: PathLike,
        mode: str = "r",
        encoding: str = "utf-8",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor | tuple[Tensor, Tensor]:
        ...


def create_path_reader(reader_function: ReaderFunction) -> FileReaderFunction:
    """
    Creates a function that reads data from a specified file path using a given reader function.

    Parameters
    ----------
    reader_function : ReaderFunction
        The function used to read and process the file contents.

    Returns
    -------
    FileReaderFunction
        A function that takes a file path, mode, device, and dtype, and returns
        the processed data.
    """

    def read_from_path(
        filepath: PathLike,
        mode: str = "r",
        encoding: str = "utf-8",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **kwargs: Any,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """
        Reads the file from the specified path.

        Parameters
        ----------
        file : PathLike
            Path of file containing the structure.
        mode : str, optional
            Mode in which the file is opened. Defaults to `"r"`.
        encoding : str, optional
            Encoding for file. Defaults to `"utf-8"`.
        device : torch.device | None, optional
            Device to store the tensor on. Defaults to `None`.
        dtype : torch.dtype | None, optional
            Floating point data type of the tensor. Defaults to `None`.

        Returns
        -------
        Tensor | tuple[Tensor, Tensor]
            Returned tensor or tensors.

        Raises
        ------
        FileNotFoundError
            The file specified in `filepath` cannot be found.
        """
        path = Path(filepath)

        # Check if the file exists
        if not path.exists():
            raise FileNotFoundError(f"The file '{path}' does not exist.")

        with open(path, mode=mode, encoding="utf-8") as fileobj:
            return reader_function(fileobj, device, dtype, **kwargs)

    return read_from_path


################################################################################


@runtime_checkable
class ReaderFunctionTensor(Protocol):
    def __call__(
        self,
        fileobj: IO[Any],
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        ...


@runtime_checkable
class FileReaderFunctionTensor(Protocol):
    def __call__(
        self,
        filepath: PathLike,
        mode: str = "r",
        encoding: str = "utf-8",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        ...


def create_path_reader_dotfiles(
    reader_function: ReaderFunctionTensor, name: Literal[".CHRG", ".UHF"]
) -> FileReaderFunctionTensor:
    """
    Creates a function that reads data from a specified file path using a given reader function.

    Parameters
    ----------
    reader_function : ReaderFunction
        The function used to read and process the file contents.
    name: Literal[".CHRG", ".UHF"]
        Name of the dotfile to be read.

    Returns
    -------
    FileReaderFunction
        A function that takes a file path, mode, device, and dtype, and returns
        the processed data.
    """
    # return default if file is not found (must be integer to allow integer
    # dtypes from PyTorch, e.g., 0.0 fails with torch.long)
    DEFAULT_VALUE = 0

    def read_from_path(
        filepath: PathLike,
        mode: str = "r",
        encoding: str = "utf-8",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        """
        Reads the file from the specified path.

        Parameters
        ----------
        file : PathLike
            Path of file containing the structure.
        mode : str, optional
            Mode in which the file is opened. Defaults to `"r"`.
        encoding : str, optional
            Encoding for file. Defaults to `"utf-8"`.
        device : torch.device | None, optional
            Device to store the tensor on. Defaults to `None`.
        dtype : torch.dtype | None, optional
            Floating point data type of the tensor. Defaults to `None`.

        Returns
        -------
        Tensor
            Value stored in the file as tensor.
        """
        path = Path(filepath)

        # possibly coordinate file given -> search dotfile in same directory
        if path.is_file():
            if path.name not in (".CHRG", ".UHF"):
                path = path.parent / name

        if path.is_dir():
            path = path / name

        # Check if the file now exists
        if not path.exists():
            return torch.tensor(DEFAULT_VALUE, device=device, dtype=dtype)

        with open(path, mode=mode, encoding=encoding) as fileobj:
            return reader_function(fileobj, device, dtype)

    return read_from_path
