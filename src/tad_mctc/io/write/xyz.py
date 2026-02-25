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
from ...units import length
from ..checks import content_checks, shape_checks
from .topath import create_path_writer

__all__ = ["write_xyz"]


def write_xyz_fileobj(
    fileobj: IO[Any],
    numbers: Tensor,
    positions: Tensor,
    **kwargs: Any,
) -> None:
    """
    Write atomic coordinates in XYZ format to a file object. The input
    positions are expected to be in atomic units (bohrs).

    Parameters
    ----------
    fileobj : IO[Any]
        The file-like object to which the XYZ data will be written.
    numbers : Tensor
        A 1D tensor containing atomic numbers or symbols.
    positions : Tensor
        A 2D tensor of shape ``(nat, 3)`` containing atomic positions in
        atomic units (bohrs).
    fmt : str, optional
        Format string for the position coordinates.
    comment : str, optional
        A comment string for the XYZ file header.

    Raises
    ------
    ValueError
        If the comment contains line breaks.
    """
    positions = positions * length.AU2AA

    fmt = kwargs.pop("fmt", "%20.14f")
    comment = kwargs.pop("comment", "").rstrip()

    if "\n" in comment:
        raise ValueError("Comment line should not have line breaks.")

    def _write(num: Tensor, pos: Tensor) -> None:
        assert shape_checks(num, pos, allow_batched=False)
        assert content_checks(num, pos)

        fileobj.write(f"{len(num)}\n{comment}\n")
        for n, p in zip(num, pos):
            sym = pse.Z2S[int(n.item())].title()
            fileobj.write(f"{sym:<2} {fmt % p[0]} {fmt % p[1]} {fmt % p[2]}\n")

    # single frame
    if len(numbers.shape) == 1 and len(positions.shape) == 2:
        _write(numbers, positions)
        return

    # check shapes
    if len(numbers.shape) != 1 and len(positions.shape) == 2:
        raise ValueError(
            "Invalid shapes: Atomic positions is 2D (not batched), but atomic "
            "numbers are not 1D (batched?)."
        )
    if len(numbers.shape) == 1 and len(positions.shape) != 2:
        raise ValueError(
            "Invalid shapes: Atomic numbers are 1D (not batched), but atomic "
            "positions are not 2D (batched?)."
        )

    # actual batched write
    if len(numbers.shape) == 2 and len(positions.shape) == 3:
        # pylint: disable=import-outside-toplevel
        from tad_mctc.batch import deflate

        for num, pos in zip(numbers, positions):
            _write(deflate(num), deflate(pos))

        return

    raise ValueError(
        f"Invalid shapes of atomic numbers ({numbers.shape}) and positions "
        f"({positions.shape})."
    )


write_xyz = create_path_writer(write_xyz_fileobj)
