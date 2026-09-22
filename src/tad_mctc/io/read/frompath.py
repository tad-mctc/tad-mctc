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
I/O Read: From Path
===================

Turn a reader that takes an open file into one that takes a path.

Example
-------
>>> from tad_mctc.io.read.frompath import create_path_reader
>>> from tad_mctc.io.read.xyz import read_xyz_fileobj
>>> read_xyz = create_path_reader(read_xyz_fileobj)
>>> structure = read_xyz("mol.xyz")  # doctest: +SKIP
"""

from __future__ import annotations

from pathlib import Path
from typing import IO, Any, Literal, Protocol, TypeVar

import torch

from ...typing import PathLike, Tensor

__all__ = ["create_path_reader", "create_path_reader_dotfiles"]


T_co = TypeVar("T_co", covariant=True)
"""What the wrapped reader returns (a `Structure`, a `Tensor`, ...)."""


class ReaderFunction(Protocol[T_co]):
    """A reader that takes an open file."""

    def __call__(
        self,
        fileobj: IO[Any],
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> T_co: ...


class FileReaderFunction(Protocol[T_co]):
    """A reader that takes a path (see :func:`create_path_reader`)."""

    def __call__(
        self,
        filepath: PathLike,
        mode: str = "r",
        encoding: str = "utf-8",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **kwargs: Any,
    ) -> T_co: ...


class DotfileReaderFunction(Protocol):
    """A ``.CHRG``/``.UHF`` reader that takes a path (see
    :func:`create_path_reader_dotfiles`)."""

    def __call__(
        self,
        filepath: PathLike,
        mode: str = "r",
        encoding: str = "utf-8",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor: ...


def create_path_reader(
    reader_function: ReaderFunction[T_co],
) -> FileReaderFunction[T_co]:
    """
    Create a reader that opens the file at a path and passes it to
    ``reader_function``, returning whatever that returns.

    Parameters
    ----------
    reader_function : ReaderFunction[T_co]
        The reader that parses the open file.

    Returns
    -------
    FileReaderFunction[T_co]
        A function taking a path, the open ``mode``/``encoding``, ``device``
        and ``dtype``, plus any keyword arguments ``reader_function``
        accepts.
    """

    def read_from_path(
        filepath: PathLike,
        mode: str = "r",
        encoding: str = "utf-8",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **kwargs: Any,
    ) -> T_co:
        """
        Read the file at the given path.

        Parameters
        ----------
        filepath : PathLike
            Path of the file.
        mode : str, optional
            Mode in which the file is opened. Defaults to ``"r"``.
        encoding : str, optional
            Encoding of the file. Defaults to ``"utf-8"``.
        device : :class:`torch.device` | None, optional
            Device to store the tensors on. Defaults to ``None``.
        dtype : :class:`torch.dtype` | None, optional
            Floating point data type of the tensors. Defaults to ``None``.

        Returns
        -------
        T_co
            What the wrapped reader returns.

        Raises
        ------
        FileNotFoundError
            The file specified in ``filepath`` cannot be found.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"The file '{path}' does not exist.")

        with open(path, mode=mode, encoding=encoding) as fileobj:
            return reader_function(fileobj, device, dtype, **kwargs)

    return read_from_path


def create_path_reader_dotfiles(
    reader_function: ReaderFunction[Tensor], name: Literal[".CHRG", ".UHF"]
) -> DotfileReaderFunction:
    """
    Create a reader for a ``.CHRG``/``.UHF`` sidecar file. Unlike
    :func:`create_path_reader`, the path may also be the structure file or
    its directory (the sidecar is then looked up next to it), and a missing
    sidecar gives zero instead of an error.

    Parameters
    ----------
    reader_function : ReaderFunction[Tensor]
        The reader that parses the open sidecar file.
    name : Literal[".CHRG", ".UHF"]
        Name of the sidecar file.

    Returns
    -------
    DotfileReaderFunction
        A function taking a path, the open ``mode``/``encoding``, ``device``
        and ``dtype``.
    """
    # must be an integer so that integer dtypes work (0.0 fails for long)
    default_value = 0

    def read_from_path(
        filepath: PathLike,
        mode: str = "r",
        encoding: str = "utf-8",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        """
        Read the sidecar file for the given path.

        Parameters
        ----------
        filepath : PathLike
            Path of the sidecar file, of the structure file next to it, or
            of their directory.
        mode : str, optional
            Mode in which the file is opened. Defaults to ``"r"``.
        encoding : str, optional
            Encoding of the file. Defaults to ``"utf-8"``.
        device : :class:`torch.device` | None, optional
            Device to store the tensor on. Defaults to ``None``.
        dtype : :class:`torch.dtype` | None, optional
            Data type of the tensor. Defaults to ``None``.

        Returns
        -------
        Tensor
            Value stored in the file, or zero if there is no such file.
        """
        path = Path(filepath)

        # possibly coordinate file given -> search dotfile in same directory
        if path.is_file():
            if path.name not in (".CHRG", ".UHF"):
                path = path.parent / name

        if path.is_dir():
            path = path / name

        if not path.exists():
            return torch.tensor(default_value, device=device, dtype=dtype)

        with open(path, mode=mode, encoding=encoding) as fileobj:
            return reader_function(fileobj, device, dtype)

    return read_from_path
