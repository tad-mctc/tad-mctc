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
I/O Write: General
==================

General writer for file from a path.
"""

from __future__ import annotations

from functools import wraps
from pathlib import Path

from ...typing import IO, Any, PathLike, Protocol, Tensor, runtime_checkable

__all__ = ["create_path_writer"]


@runtime_checkable
class WriterFunction(Protocol):
    """Type annotation for a writer function."""

    def __call__(
        self,
        fileobj: IO[Any],
        numbers: Tensor,
        positions: Tensor,
        **kwargs: Any,
    ) -> None: ...


@runtime_checkable
class FileWriterFunction(Protocol):
    """Type annotation for a file writer function."""

    def __call__(
        self,
        filepath: PathLike,
        numbers: Tensor,
        positions: Tensor,
        mode: str = "w",
        fmt: str = "%22.15f",
        overwrite: bool = False,
        **kwargs: Any,
    ) -> None: ...


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
        A function that takes a file path, numbers, positions, mode, comment,
        and format string, and writes the data to the file.
    """

    @wraps(writer_function)
    def write_to_path(
        filepath: PathLike,
        numbers: Tensor,
        positions: Tensor,
        mode: str = "w",
        fmt: str = "%22.15f",
        overwrite: bool = False,
        **kwargs: Any,
    ) -> None:
        path = Path(filepath)

        # Check if the file already exists
        if mode.strip() == "w":
            if path.exists() and overwrite is False:
                raise FileExistsError(
                    f"The file '{filepath}' already exists. If you want to "
                    "overwrite it, set `overwrite=True`."
                )

        with open(path, mode=mode, encoding="utf-8") as fileobj:
            writer_function(fileobj, numbers, positions, fmt=fmt, **kwargs)

    return write_to_path
