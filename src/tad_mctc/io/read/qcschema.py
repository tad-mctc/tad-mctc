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
I/O Read: JSON
==============

Reader for JSON/QCSchema files.
"""
from __future__ import annotations

import torch

from ...data import pse
from ...typing import DD, IO, Any, Tensor, get_default_dtype
from ..checks import content_checks, deflatable_check, shape_checks
from .frompath import create_path_reader

__all__ = [
    "read_qcschema",
    "read_qcschema_from_path",
]


def read_qcschema(
    fileobj: IO[Any],
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    dtype_int: torch.dtype = torch.long,
    **kwargs: Any,
) -> tuple[Tensor, Tensor]:
    """
    Reads a JSON/QCSchema file with a single structure and returns atomic
    numbers and positions as tensors.
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
        Tensors of atomic numbers and positions. Positions is a tensor of shape
        (nat, 3) in atomic units.
    """
    dd: DD = {
        "device": device,
        "dtype": dtype if dtype is not None else get_default_dtype(),
    }
    ddi: DD = {"device": device, "dtype": dtype_int}

    # pylint: disable=import-beyond-top-level
    from json import loads as json_load

    data = json_load(fileobj.read())

    if "molecule" not in data:
        raise KeyError(f"Invalid schema: Key 'molecule' not found in '{fileobj}'.")

    mol = data["molecule"]

    if "symbols" not in mol:
        raise KeyError(f"Invalid schema: Key 'symbols' not found in '{fileobj}'.")
    if "geometry" not in mol:
        raise KeyError(f"Invalid schema: Key 'geometry' not found in '{fileobj}'.")

    geo = mol["geometry"]
    coords = []
    for i in range(0, len(geo), 3):
        coords.append([float(geo[i]), float(geo[i + 1]), float(geo[i + 2])])

    numbers = torch.tensor([pse.S2Z[s] for s in mol["symbols"]], **ddi)
    positions = torch.tensor(coords, **dd)

    assert shape_checks(numbers, positions)
    assert content_checks(numbers, positions)
    assert deflatable_check(positions, fileobj, **kwargs)

    return numbers, positions


read_qcschema_from_path = create_path_reader(read_qcschema)
