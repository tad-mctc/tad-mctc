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
Read: XYZ
=========

Reader for standard XYZ files.
See https://en.wikipedia.org/wiki/XYZ_file_format.
"""
from __future__ import annotations

import torch

from ...batch import pack
from ...data import pse
from ...typing import DD, IO, Any, Tensor, get_default_dtype
from ...units import length
from ..checks import content_checks, shape_checks
from .reader import create_path_reader

__all__ = ["read_xyz", "read_xyz_from_path"]


def read_xyz(
    fileobj: IO[Any],
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    dtype_int: torch.dtype = torch.long,
) -> tuple[Tensor, Tensor]:
    """
    Reads an XYZ file and returns atomic numbers and positions as tensors.
    Handles multiple structures by batching them together.
    Positions are converted to atomic units (bohrs).

    Parameters
    ----------
    fileobj : IO[Any]
        The file-like object to read from.
    device : torch.device | None, optional
        Device to store the tensor on. Defaults to `None`.
    dtype : torch.dtype | None, optional
        Floating point data type of the tensor. Defaults to `None`.
    dtype_int : torch.dtype, optional
        Integer data type of the tensor. Defaults to `torch.long`.

    Returns
    -------
    (Tensor, Tensor)
        (Possibly batched) tensors of atomic numbers and positions. Positions is a tensor of shape (batch_size, nat, 3) in atomic units.
    """
    dd: DD = {
        "device": device,
        "dtype": dtype if dtype is not None else get_default_dtype(),
    }
    ddi: DD = {"device": device, "dtype": dtype_int}

    numbers_images = []
    positions_images = []

    while True:
        natoms_line = fileobj.readline()
        if not natoms_line:
            break
        natoms = int(natoms_line.strip())

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

        assert shape_checks(numbers, positions)
        assert content_checks(numbers, positions)

        numbers_images.append(numbers)
        positions_images.append(positions)

    # if only one image, return its tensors directly
    if len(numbers_images) == 1:
        return numbers_images[0], positions_images[0]

    return pack(numbers_images), pack(positions_images)


read_xyz_from_path = create_path_reader(read_xyz)
