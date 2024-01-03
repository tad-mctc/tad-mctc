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
I/O Read: General
=================

General reader for file from a path.
"""
from __future__ import annotations

from pathlib import Path

import torch

from ...typing import IO, Any, PathLike, Tensor
from .qcschema import read_qcschema
from .turbomole import read_turbomole
from .xyz import read_xyz, read_xyz_qm9

__all__ = ["read", "read_from_path"]


def read(
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
    device : torch.device | None, optional
        Device to store the tensor on. Defaults to `None`.
    dtype : torch.dtype | None, optional
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
        numbers, positions = read_xyz(
            fileobj, device=device, dtype=dtype, dtype_int=dtype_int, **kwargs
        )
    elif ftype in ("qm9",):
        numbers, positions = read_xyz_qm9(
            fileobj, device=device, dtype=dtype, dtype_int=dtype_int, **kwargs
        )
    elif ftype in ("tmol", "tm", "turbomole") or fname == "coord":
        numbers, positions = read_turbomole(
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
        numbers, positions = read_qcschema(
            fileobj, device=device, dtype=dtype, dtype_int=dtype_int, **kwargs
        )
    else:
        raise ValueError(f"Unknown filetype '{ftype}' in '{fileobj}'.")

    return numbers, positions


def read_from_path(
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
    device : torch.device | None, optional
        Device to store the tensor on. Defaults to `None`.
    dtype : torch.dtype | None, optional
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
        File given does not exist.
    """
    path = Path(filepath)

    # Check if the file exists
    if not path.exists():
        raise FileNotFoundError(f"The file '{path}' does not exist.")

    if ftype is None:
        ftype = path.suffix.lower()[1:]

    with open(path, mode=mode, encoding="utf-8") as fileobj:
        return read(
            fileobj,
            ftype,
            device=device,
            dtype=dtype,
            dtype_int=dtype_int,
            **kwargs,
        )
