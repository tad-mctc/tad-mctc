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
I/O Write: XYZ
==============

Writer for standard XYZ files.
See https://en.wikipedia.org/wiki/XYZ_file_format.
"""

from __future__ import annotations

from ...data import pse
from ...typing import IO, Any, Tensor
from ..checks import content_checks, shape_checks
from .topath import create_path_writer

__all__ = ["write_turbomole"]


def write_turbomole_fileobj(
    fileobj: IO[Any], numbers: Tensor, positions: Tensor, **kwargs: Any
) -> None:
    """
    Write atomic coordinates in Turbomole format to a file object.
    The input positions are expected to be in atomic units (bohrs).

    Parameters
    ----------
    fileobj : IO[Any]
        The file-like object to which the Turbomole data will be written.
    numbers : Tensor
        A 1D tensor containing atomic numbers.
    positions : Tensor
        A 2D tensor of shape (n_atoms, 3) containing atomic positions in atomic
        units (bohrs).
    fmt : str, optional
        Format string for the position coordinates.
    """
    assert shape_checks(numbers, positions)
    assert content_checks(numbers, positions)

    fmt = kwargs.pop("fmt", "%20.14f")

    fileobj.write("$coord\n")

    for num, pos in zip(numbers.tolist(), positions.tolist()):
        symbol = pse.Z2S[num]

        # Convert 'X' to 'Q' for Turbomole ghost atoms
        symbol = symbol if symbol != "X" else "Q"

        fileobj.write(
            f"{fmt % pos[0]}  {fmt % pos[1]}  {fmt % pos[2]}      {symbol.lower()} \n"
        )

    fileobj.write("$end\n")


write_turbomole = create_path_writer(write_turbomole_fileobj)
