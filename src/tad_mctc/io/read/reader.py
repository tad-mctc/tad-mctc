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
Read: General
=============

General reader for file from a path.
"""
from __future__ import annotations

from functools import wraps
from pathlib import Path
from typing import runtime_checkable

import torch

from ...typing import IO, Any, PathLike, Protocol

__all__ = ["create_path_reader"]


@runtime_checkable
class ReaderFunction(Protocol):
    def __call__(
        self,
        fileobj: IO[Any],
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ...


@runtime_checkable
class FileReaderFunction(Protocol):
    def __call__(
        self,
        filepath: PathLike,
        mode: str = "r",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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

    @wraps(reader_function)
    def read_from_path(
        filepath: PathLike,
        mode: str = "r",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        path = Path(filepath)

        # Check if the file exists
        if not path.exists():
            raise FileNotFoundError(f"The file '{path}' does not exist.")

        with open(filepath, mode=mode, encoding="utf-8") as fileobj:
            return reader_function(fileobj, device, dtype)

    return read_from_path