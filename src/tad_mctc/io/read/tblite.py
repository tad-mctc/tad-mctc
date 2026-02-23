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
I/O Read: tblite
================

Reader for tblite output files.
"""

from __future__ import annotations

import torch

from ...typing import DD, IO, Any, Tensor, get_default_dtype
from .frompath import create_path_reader

__all__ = ["read_tblite_engrad"]


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
    # pylint: disable=import-outside-toplevel
    from json import loads as json_load

    return json_load(fileobj.read())


def read_tblite_engrad_fileobj(
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
    device : :class:`torch.device` | None, optional
        Device to store the tensor on. Defaults to `None`.
    dtype : :class:`torch.dtype` | None, optional
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


read_tblite_engrad = create_path_reader(read_tblite_engrad_fileobj)
