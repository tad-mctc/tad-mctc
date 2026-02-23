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
I/O Read: XYZ
=============

Reader for standard XYZ files.
See https://en.wikipedia.org/wiki/XYZ_file_format.
"""

from __future__ import annotations

import torch

from ...batch import pack
from ...data import pse
from ...exceptions import FormatErrorXYZ
from ...typing import DD, IO, Any, Tensor, get_default_dtype
from ...units import length
from ..checks import content_checks, deflatable_check, shape_checks
from .frompath import create_path_reader

__all__ = ["read_xyz", "read_xyz_qm9"]


def read_xyz_fileobj(
    fileobj: IO[Any],
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    dtype_int: torch.dtype = torch.long,
    **kwargs: Any,
) -> tuple[Tensor, Tensor]:
    """
    Reads an XYZ file and returns atomic numbers and positions as tensors.
    Handles multiple structures by batching them together.
    Positions are converted to atomic units (bohrs).

    Parameters
    ----------
    fileobj : IO[Any]
        The file-like object to read from.
    device : :class:`torch.device` | None, optional
        Device to store the tensor on. Defaults to `None`.
    dtype : :class:`torch.dtype` | None, optional
        Floating point data type of the tensor. Defaults to `None`.
    dtype_int : torch.dtype, optional
        Integer data type of the tensor. Defaults to `torch.long`.

    Returns
    -------
    (Tensor, Tensor)
        (Possibly batched) tensors of atomic numbers and positions. Positions
        is a tensor of shape (batch_size, nat, 3) in atomic units.
    """
    line: list[str]
    natoms_line: str

    dd: DD = {
        "device": device,
        "dtype": dtype if dtype is not None else get_default_dtype(),
    }
    ddi: DD = {"device": device, "dtype": dtype_int}

    numbers_images: list[Tensor] = []
    positions_images: list[Tensor] = []

    while True:
        # Stripping here also covers additional trailing new lines; otherwise,
        # a blank line would be interpreted as the start of a new image.
        natoms_line = fileobj.readline().strip()
        if not natoms_line:
            break

        # Check if line is a number (must strip newline character before)
        if not natoms_line.isdigit():
            raise FormatErrorXYZ(
                "The first line in an xyz file should be the number of atoms "
                f"in the structure, but is {repr(natoms_line)}."
            )
        natoms = int(natoms_line)

        # Skip comment line
        fileobj.readline()

        symbols = []
        coords = []
        for _ in range(natoms):
            line = fileobj.readline().split()
            symbol, x, y, z = line[:4]
            symbols.append(symbol.title())
            coords.append([float(x), float(y), float(z)])

        numbers = torch.tensor([pse.S2Z[symbol] for symbol in symbols], **ddi)
        positions = torch.tensor(coords, **dd) * length.AA2AU

        assert shape_checks(numbers, positions, allow_batched=False)
        assert content_checks(numbers, positions, allow_batched=False)
        assert deflatable_check(positions, fileobj, **kwargs)

        numbers_images.append(numbers)
        positions_images.append(positions)

    # if only one image, return its tensors directly
    if len(numbers_images) == 1:
        n = numbers_images[0]
        p = positions_images[0]

        if kwargs.get("batch_agnostic", False):
            return n.unsqueeze(0), p.unsqueeze(0)

        return n, p

    return pack(numbers_images), pack(positions_images)


read_xyz = create_path_reader(read_xyz_fileobj)


def read_xyz_qm9_fileobj(
    fileobj: IO[Any],
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    dtype_int: torch.dtype = torch.long,
) -> tuple[Tensor, Tensor]:
    """
    Reads the XYZ file of the QM9 dataset, which does not conform with the
    standard format, and returns atomic numbers and positions as tensors.
    Handles only a single structure.
    Positions are converted to atomic units (bohrs).

    Parameters
    ----------
    fileobj : IO[Any]
        The file-like object to read from.
    device : :class:`torch.device` | None, optional
        Device to store the tensor on. Defaults to `None`.
    dtype : :class:`torch.dtype` | None, optional
        Floating point data type of the tensor. Defaults to `None`.
    dtype_int : torch.dtype, optional
        Integer data type of the tensor. Defaults to `torch.long`.

    Returns
    -------
    (Tensor, Tensor)
        Tensors of atomic numbers and positions. Positions is a tensor of shape
        (nat, 3) in atomic units.
    """
    line: list[str]
    natoms_line: str

    dd: DD = {
        "device": device,
        "dtype": dtype if dtype is not None else get_default_dtype(),
    }
    ddi: DD = {"device": device, "dtype": dtype_int}

    natoms_line = fileobj.readline()
    natoms = int(natoms_line.strip())

    # Skip comment line
    fileobj.readline()

    symbols = []
    coords = []
    for _ in range(natoms):
        line = fileobj.readline().split()
        symbols.append(line[0].title())
        coords.append([float(x.replace("*^", "e")) for x in line[1:4]])

    numbers = torch.tensor([pse.S2Z[symbol] for symbol in symbols], **ddi)
    positions = torch.tensor(coords, **dd) * length.AA2AU

    assert shape_checks(numbers, positions, allow_batched=False)
    assert content_checks(numbers, positions, allow_batched=False)
    assert deflatable_check(positions, fileobj)

    return numbers, positions


read_xyz_qm9 = create_path_reader(read_xyz_qm9_fileobj)
