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
I/O Read: General
=================

General reader for file from a path.
"""

from __future__ import annotations

from pathlib import Path

import torch

from ...typing import IO, Any, PathLike, Tensor
from .qcschema import read_qcschema_fileobj
from .turbomole import read_turbomole_fileobj
from .xyz import read_xyz_fileobj, read_xyz_qm9_fileobj

__all__ = ["read"]


def read_from_fileobj(
    fileobj: IO[Any],
    ftype: str,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    dtype_int: torch.dtype = torch.long,
    **kwargs: Any,
) -> tuple[Tensor, Tensor]:
    """
    Helper to read the structure from the given file.

    Parameters
    ----------
    fileobj : IO[Any]
        The file-like object to read from.
    ftype : str | None, optional
        File type. Defaults to `None`, i.e., infered from the extension.
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

    Raises
    ------
    NotImplementedError
        Reader for specific file type not implemented.
    ValueError
        Unknown file type.
    """
    # path stored in TextIOWrapper
    fname = Path(fileobj.name).name.lower()

    if ftype in ("xyz", "log"):
        numbers, positions = read_xyz_fileobj(
            fileobj, device=device, dtype=dtype, dtype_int=dtype_int, **kwargs
        )
    elif ftype in ("qm9",):
        numbers, positions = read_xyz_qm9_fileobj(
            fileobj, device=device, dtype=dtype, dtype_int=dtype_int, **kwargs
        )
    elif ftype in ("tmol", "tm", "turbomole") or fname == "coord":
        numbers, positions = read_turbomole_fileobj(
            fileobj, device=device, dtype=dtype, dtype_int=dtype_int, **kwargs
        )
    elif ftype in ("mol", "sdf", "gen", "pdb"):
        raise NotImplementedError(
            f"Filetype '{ftype}' recognized but no reader available."
        )
    elif ftype in ("qchem",):
        raise NotImplementedError(
            f"Filetype '{ftype}' (Q-Chem) recognized but no reader available."
        )
    elif ftype in ("poscar", "contcar", "vasp", "crystal") or fname in (
        "poscar",
        "contcar",
        "vasp",
    ):
        raise NotImplementedError(
            f"Filetype '{ftype}' (VASP/CRYSTAL) recognized but no reader available."
        )
    elif ftype in ("ein", "gaussian"):
        raise NotImplementedError(
            f"Filetype '{ftype}' (Gaussian) recognized but no reader available."
        )
    elif ftype in ("json", "qcschema"):
        numbers, positions = read_qcschema_fileobj(
            fileobj, device=device, dtype=dtype, dtype_int=dtype_int, **kwargs
        )
    else:
        raise ValueError(f"Unknown filetype '{ftype}' in '{fileobj}'.")

    return numbers, positions


def read(
    filepath: PathLike,
    ftype: str | None = None,
    mode: str = "r",
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    dtype_int: torch.dtype = torch.long,
    **kwargs: Any,
) -> tuple[Tensor, Tensor]:
    """
    Helper to read the structure from the given file path.

    Parameters
    ----------
    file : PathLike
        Path of file containing the structure.
    ftype : str | None, optional
        File type. Defaults to `None`, i.e., infered from the extension.
    mode : str, optional
        Mode in which the file is opened. Defaults to `"r"`.
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

    Raises
    ------
    FileNotFoundError
        Given file does not exist.
    """
    path = Path(filepath)

    # Check if the file exists
    if not path.exists():
        raise FileNotFoundError(f"The file '{path}' does not exist.")

    if ftype is None:
        ftype = path.suffix.lower()[1:]

    with open(path, mode=mode, encoding="utf-8") as fileobj:
        return read_from_fileobj(
            fileobj,
            ftype,
            device=device,
            dtype=dtype,
            dtype_int=dtype_int,
            **kwargs,
        )
