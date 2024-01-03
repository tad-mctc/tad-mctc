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
I/O Read: tblite
================

Reader for tblite output files.
"""
from __future__ import annotations

import torch

from ...typing import DD, IO, Any, Tensor, get_default_dtype
from .frompath import create_path_reader

__all__ = ["read_tblite_engrad", "read_tblite_engrad_from_path"]


def _read_tblite_gfn(fileobj: IO[Any]) -> dict[str, str | float]:
    """
    Read energy file from tblite json output.

    Parameters
    ----------
    fileobj : IO[Any]
        The file-like object to read from.

    Returns
    -------
    dict[str, str | float]
        Full tblite JSON output.
    """
    from json import loads as json_load

    return json_load(fileobj.read())


def read_tblite_engrad(
    fileobj: IO[Any],
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[Tensor, Tensor]:
    """
    Read tblite energy and gradient from tblite's JSON output.

    Parameters
    ----------
    fileobj : IO[Any]
        The file-like object to read from.
    device : torch.device | None, optional
        Device to store the tensor on. Defaults to `None`.
    dtype : torch.dtype | None, optional
        Floating point data type of the tensor. Defaults to `None`.

    Returns
    -------
    (Tensor, Tensor)
        Tensors of atomic energies and nuclear gradients.
    """
    dd: DD = {
        "device": device,
        "dtype": dtype if dtype is not None else get_default_dtype(),
    }

    data = _read_tblite_gfn(fileobj)

    energies = torch.tensor(data.get("energies", False), **dd)
    gradient = torch.tensor(data.get("gradient", False), **dd)
    return energies, gradient.reshape(-1, 3)


read_tblite_engrad_from_path = create_path_reader(read_tblite_engrad)
