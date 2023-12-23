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
Write: General
==============

General writer for file from a path.
"""
from __future__ import annotations

from functools import wraps
from pathlib import Path

from ...typing import IO, Any, PathLike, Protocol, Tensor, runtime_checkable

__all__ = ["create_path_writer"]


@runtime_checkable
class WriterFunction(Protocol):
    def __call__(
        self,
        fileobj: IO[Any],
        numbers: Tensor,
        positions: Tensor,
        **kwargs: Any,
    ) -> None:
        ...


@runtime_checkable
class FileWriterFunction(Protocol):
    def __call__(
        self,
        filepath: PathLike,
        numbers: Tensor,
        positions: Tensor,
        mode: str = "w",
        fmt: str = "%22.15f",
        **kwargs: Any,
    ) -> None:
        ...


def create_path_writer(writer_function: WriterFunction) -> FileWriterFunction:
    """
    Creates a function that writes data to a specified file path using a given writer function.

    Parameters
    ----------
    writer_function : WriterFunction
        The function used to write the file contents.

    Returns
    -------
    FileWriterFunction
        A function that takes a file path, numbers, positions, mode, comment, and format string, and writes the data to the file.
    """

    @wraps(writer_function)
    def write_to_path(
        filepath: PathLike,
        numbers: Tensor,
        positions: Tensor,
        mode: str = "w",
        fmt: str = "%22.15f",
        **kwargs: Any,
    ) -> None:
        path = Path(filepath)

        # Check if the file already exists
        if mode.strip() == "w":
            if path.exists():
                raise FileExistsError(f"The file '{filepath}' already exists.")

        with open(path, mode=mode, encoding="utf-8") as fileobj:
            writer_function(fileobj, numbers, positions, fmt=fmt, **kwargs)

    return write_to_path
