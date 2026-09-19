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

A convenience function to create readers that take a path instead of a stream.

Example
-------
>>> from tad_mctc.io import read
>>> read_xyz = read.create_path_reader(read.read_xyz_fileobj)
>>> path = ...
>>> numbers, positions = read_xyz(path)
"""

from __future__ import annotations

from pathlib import Path
from typing import IO, Any, Literal, Protocol, runtime_checkable

import torch

from ...typing import PathLike, Tensor

__all__ = [
    "create_path_reader",
    "create_path_reader_cjson",
    "create_path_reader_dotfiles",
    "create_path_reader_json",
    "create_path_reader_lattice",
    "create_path_reader_periodic",
]

CJSONResult = tuple[
    Tensor, Tensor, Tensor | None, Tensor | None, Tensor | None, Tensor | None
]
"""Return shape for a reader that can independently carry a periodic
lattice (`lattice`, `periodic`) and bond connectivity (`bonds`,
`bond_orders`) -- unlike every other multi-field reader in this package,
these four are not all-or-nothing together, so the tuple is always this
fixed length with `None` standing in for whichever pieces the file didn't
have (currently only cjson; see `read.cjson.read_cjson_fileobj`)."""

JSONResult = (
    tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor, Tensor] | CJSONResult
)
"""Return shape for the general JSON sniff-and-dispatch reader
(`read.json.read_json_fileobj`): whichever of its three delegates'
shapes actually matched -- plain qcschema/cjson-without-anything-set
(2-tuple), pymatgen (2-or-4-tuple), or cjson (the fixed 6-tuple
`CJSONResult`)."""


@runtime_checkable
class ReaderFunction(Protocol):
    """Type annotation for a reader function."""

    def __call__(
        self,
        fileobj: IO[Any],
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor | tuple[Tensor, Tensor]: ...


@runtime_checkable
class FileReaderFunction(Protocol):
    """Type annotation for a file reader function."""

    def __call__(
        self,
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
            Mode in which the file is opened. Defaults to ``"r"``.
        encoding : str, optional
            Encoding for file. Defaults to ``"utf-8"``.
        device : :class:`torch.device` | None, optional
            Device to store the tensor on. Defaults to ``None``.
        dtype : :class:`torch.dtype` | None, optional
            Floating point data type of the tensor. Defaults to ``None``.

        Returns
        -------
        Tensor | tuple[Tensor, Tensor]
            Returned tensor or tensors.

        Raises
        ------
        FileNotFoundError
            The file specified in ``filepath`` cannot be found.
        """
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
            Mode in which the file is opened. Defaults to ``"r"``.
        encoding : str, optional
            Encoding for file. Defaults to ``"utf-8"``.
        device : :class:`torch.device` | None, optional
            Device to store the tensor on. Defaults to ``None``.
        dtype : :class:`torch.dtype` | None, optional
            Floating point data type of the tensor. Defaults to ``None``.

        Returns
        -------
        Tensor | tuple[Tensor, Tensor]
            Returned tensor or tensors.

        Raises
        ------
        FileNotFoundError
            The file specified in ``filepath`` cannot be found.
        """
        path = Path(filepath)

        # Check if the file exists
        if not path.exists():
            raise FileNotFoundError(f"The file '{path}' does not exist.")

        with open(path, mode=mode, encoding=encoding) as fileobj:
            return reader_function(fileobj, device, dtype, **kwargs)

    return read_from_path


################################################################################


@runtime_checkable
class ReaderFunctionLattice(Protocol):
    """Type annotation for a reader function that also returns a lattice
    (currently only :func:`tad_mctc.io.read.vasp.read_poscar_fileobj`)."""

    def __call__(
        self,
        fileobj: IO[Any],
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]: ...


@runtime_checkable
class FileReaderFunctionLattice(Protocol):
    """Type annotation for a file reader function that also returns a
    lattice."""

    def __call__(
        self,
        filepath: PathLike,
        mode: str = "r",
        encoding: str = "utf-8",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **kwargs: Any,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Reads the file from the specified path.

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
        (Tensor, Tensor, Tensor)
            Atomic numbers, positions and lattice vectors.

        Raises
        ------
        FileNotFoundError
            The file specified in ``filepath`` cannot be found.
        """
        ...


def create_path_reader_lattice(
    reader_function: ReaderFunctionLattice,
) -> FileReaderFunctionLattice:
    """
    Creates a function that reads data from a specified file path using a
    given reader function that returns a lattice alongside numbers and
    positions. Kept separate from :func:`create_path_reader` so that the
    latter's return type stays a plain ``Tensor | tuple[Tensor, Tensor]``
    for every non-periodic reader, instead of a 2-vs-3-tuple union that
    would make every caller's unpacking ambiguous to the type checker.

    Parameters
    ----------
    reader_function : ReaderFunctionLattice
        The function used to read and process the file contents.

    Returns
    -------
    FileReaderFunctionLattice
        A function that takes a file path, mode, device, and dtype, and
        returns the processed data.
    """

    def read_from_path(
        filepath: PathLike,
        mode: str = "r",
        encoding: str = "utf-8",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **kwargs: Any,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Reads the file from the specified path.

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
        (Tensor, Tensor, Tensor)
            Atomic numbers, positions and lattice vectors.

        Raises
        ------
        FileNotFoundError
            The file specified in ``filepath`` cannot be found.
        """
        path = Path(filepath)

        # Check if the file exists
        if not path.exists():
            raise FileNotFoundError(f"The file '{path}' does not exist.")

        with open(path, mode=mode, encoding=encoding) as fileobj:
            return reader_function(fileobj, device, dtype, **kwargs)

    return read_from_path


################################################################################


@runtime_checkable
class ReaderFunctionTensor(Protocol):
    """Type annotation for a reader function that returns a tensor."""

    def __call__(
        self,
        fileobj: IO[Any],
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor: ...


@runtime_checkable
class FileReaderFunctionTensor(Protocol):
    """Type annotation for a file reader function that returns a tensor."""

    def __call__(
        self,
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
        """
        ...


def create_path_reader_dotfiles(
    reader_function: ReaderFunctionTensor, name: Literal[".CHRG", ".UHF"]
) -> FileReaderFunctionTensor:
    """
    Creates a function that reads data from a specified file path using a
    given reader function.

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
    default_value = 0

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
            return torch.tensor(default_value, device=device, dtype=dtype)

        with open(path, mode=mode, encoding=encoding) as fileobj:
            return reader_function(fileobj, device, dtype)

    return read_from_path


################################################################################


@runtime_checkable
class ReaderFunctionPeriodic(Protocol):
    """Type annotation for a reader function that returns a lattice and a
    periodicity mask alongside numbers and positions for a periodic file,
    or the plain (numbers, positions) pair otherwise (currently only
    :func:`tad_mctc.io.read.turbomole.read_turbomole_fileobj`)."""

    def __call__(
        self,
        fileobj: IO[Any],
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor, Tensor]: ...


@runtime_checkable
class FileReaderFunctionPeriodic(Protocol):
    """Type annotation for a file reader function that returns a lattice
    and a periodicity mask alongside numbers and positions for a periodic
    file, or the plain (numbers, positions) pair otherwise."""

    def __call__(
        self,
        filepath: PathLike,
        mode: str = "r",
        encoding: str = "utf-8",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **kwargs: Any,
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Reads the file from the specified path.

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
        (Tensor, Tensor) | (Tensor, Tensor, Tensor, Tensor)
            Atomic numbers and positions, plus lattice vectors and a
            periodicity mask for a periodic file.

        Raises
        ------
        FileNotFoundError
            The file specified in ``filepath`` cannot be found.
        """
        ...


def create_path_reader_periodic(
    reader_function: ReaderFunctionPeriodic,
) -> FileReaderFunctionPeriodic:
    """
    Creates a function that reads data from a specified file path using a
    given reader function whose return arity depends on whether the file
    turned out to be periodic. Kept separate from :func:`create_path_reader`
    and :func:`create_path_reader_lattice` for the same reason as the
    latter: sharing a Protocol type across readers with different return
    arities makes every caller's unpacking ambiguous to the type checker.

    Parameters
    ----------
    reader_function : ReaderFunctionPeriodic
        The function used to read and process the file contents.

    Returns
    -------
    FileReaderFunctionPeriodic
        A function that takes a file path, mode, device, and dtype, and
        returns the processed data.
    """

    def read_from_path(
        filepath: PathLike,
        mode: str = "r",
        encoding: str = "utf-8",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **kwargs: Any,
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Reads the file from the specified path.

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
        (Tensor, Tensor) | (Tensor, Tensor, Tensor, Tensor)
            Atomic numbers and positions, plus lattice vectors and a
            periodicity mask for a periodic file.

        Raises
        ------
        FileNotFoundError
            The file specified in ``filepath`` cannot be found.
        """
        path = Path(filepath)

        # Check if the file exists
        if not path.exists():
            raise FileNotFoundError(f"The file '{path}' does not exist.")

        with open(path, mode=mode, encoding=encoding) as fileobj:
            return reader_function(fileobj, device, dtype, **kwargs)

    return read_from_path


################################################################################


@runtime_checkable
class ReaderFunctionCJSON(Protocol):
    """Type annotation for a reader function that independently carries a
    periodic lattice and bond connectivity (see `CJSONResult`)."""

    def __call__(
        self,
        fileobj: IO[Any],
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> CJSONResult: ...


@runtime_checkable
class FileReaderFunctionCJSON(Protocol):
    """Type annotation for a file reader function that independently
    carries a periodic lattice and bond connectivity."""

    def __call__(
        self,
        filepath: PathLike,
        mode: str = "r",
        encoding: str = "utf-8",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **kwargs: Any,
    ) -> CJSONResult:
        """
        Reads the file from the specified path.

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
        CJSONResult
            Atomic numbers and positions, plus lattice vectors,
            periodicity mask, bond indices and bond orders -- the latter
            four each independently ``None`` if the file didn't have it.

        Raises
        ------
        FileNotFoundError
            The file specified in ``filepath`` cannot be found.
        """
        ...


def create_path_reader_cjson(
    reader_function: ReaderFunctionCJSON,
) -> FileReaderFunctionCJSON:
    """
    Creates a function that reads data from a specified file path using a
    given reader function whose lattice/periodicity and bond connectivity
    are independently optional (see `CJSONResult`).

    Parameters
    ----------
    reader_function : ReaderFunctionCJSON
        The function used to read and process the file contents.

    Returns
    -------
    FileReaderFunctionCJSON
        A function that takes a file path, mode, device, and dtype, and
        returns the processed data.
    """

    def read_from_path(
        filepath: PathLike,
        mode: str = "r",
        encoding: str = "utf-8",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **kwargs: Any,
    ) -> CJSONResult:
        """
        Reads the file from the specified path.

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
        CJSONResult
            Atomic numbers and positions, plus lattice vectors,
            periodicity mask, bond indices and bond orders -- the latter
            four each independently ``None`` if the file didn't have it.

        Raises
        ------
        FileNotFoundError
            The file specified in ``filepath`` cannot be found.
        """
        path = Path(filepath)

        # Check if the file exists
        if not path.exists():
            raise FileNotFoundError(f"The file '{path}' does not exist.")

        with open(path, mode=mode, encoding=encoding) as fileobj:
            return reader_function(fileobj, device, dtype, **kwargs)

    return read_from_path


################################################################################


@runtime_checkable
class ReaderFunctionJSON(Protocol):
    """Type annotation for a reader function whose return shape depends
    on which of several JSON schemas the file matched (see `JSONResult`;
    currently only `read.json.read_json_fileobj`)."""

    def __call__(
        self,
        fileobj: IO[Any],
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> JSONResult: ...


@runtime_checkable
class FileReaderFunctionJSON(Protocol):
    """Type annotation for a file reader function whose return shape
    depends on which of several JSON schemas the file matched."""

    def __call__(
        self,
        filepath: PathLike,
        mode: str = "r",
        encoding: str = "utf-8",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **kwargs: Any,
    ) -> JSONResult:
        """
        Reads the file from the specified path.

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
        JSONResult
            Whichever shape the matched delegate schema returns.

        Raises
        ------
        FileNotFoundError
            The file specified in ``filepath`` cannot be found.
        """
        ...


def create_path_reader_json(
    reader_function: ReaderFunctionJSON,
) -> FileReaderFunctionJSON:
    """
    Creates a function that reads data from a specified file path using a
    given sniff-and-dispatch JSON reader function (see `JSONResult`).

    Parameters
    ----------
    reader_function : ReaderFunctionJSON
        The function used to read and process the file contents.

    Returns
    -------
    FileReaderFunctionJSON
        A function that takes a file path, mode, device, and dtype, and
        returns the processed data.
    """

    def read_from_path(
        filepath: PathLike,
        mode: str = "r",
        encoding: str = "utf-8",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **kwargs: Any,
    ) -> JSONResult:
        """
        Reads the file from the specified path.

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
        JSONResult
            Whichever shape the matched delegate schema returns.

        Raises
        ------
        FileNotFoundError
            The file specified in ``filepath`` cannot be found.
        """
        path = Path(filepath)

        # Check if the file exists
        if not path.exists():
            raise FileNotFoundError(f"The file '{path}' does not exist.")

        with open(path, mode=mode, encoding=encoding) as fileobj:
            return reader_function(fileobj, device, dtype, **kwargs)

    return read_from_path
