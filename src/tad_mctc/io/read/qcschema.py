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

__all__ = ["read_qcschema"]


def read_qcschema_fileobj(
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
    dd: DD = {
        "device": device,
        "dtype": dtype if dtype is not None else get_default_dtype(),
    }
    ddi: DD = {"device": device, "dtype": dtype_int}

    # pylint: disable=import-outside-toplevel
    from json import loads as json_load

    data = json_load(fileobj.read())

    if "molecule" not in data:
        raise KeyError(
            f"Invalid schema: Key 'molecule' not found in '{fileobj}'."
        )

    mol = data["molecule"]

    if "symbols" not in mol:
        raise KeyError(
            f"Invalid schema: Key 'symbols' not found in '{fileobj}'."
        )
    if "geometry" not in mol:
        raise KeyError(
            f"Invalid schema: Key 'geometry' not found in '{fileobj}'."
        )

    geo = mol["geometry"]
    coords = []
    for i in range(0, len(geo), 3):
        coords.append([float(geo[i]), float(geo[i + 1]), float(geo[i + 2])])

    numbers = torch.tensor([pse.S2Z[s] for s in mol["symbols"]], **ddi)
    positions = torch.tensor(coords, **dd)

    assert shape_checks(numbers, positions, allow_batched=False)
    assert content_checks(numbers, positions, allow_batched=False)
    assert deflatable_check(positions, fileobj, **kwargs)

    return numbers, positions


read_qcschema = create_path_reader(read_qcschema_fileobj)
