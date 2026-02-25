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
I/O Write: General
==================

General writer for files to a path.
"""

from __future__ import annotations

from pathlib import Path

from ...typing import Any, PathLike, Tensor
from .turbomole import write_turbomole_fileobj
from .xyz import write_xyz_fileobj

__all__ = ["write"]


def write(
    filepath: PathLike,
    numbers: Tensor,
    positions: Tensor,
    ftype: str | None = None,
    mode: str = "w",
    overwrite: bool = False,
    **kwargs: Any,
) -> None:
    """
    Write the structure to the given file path.

    Parameters
    ----------
    filepath : PathLike
        File path to write the structure to.
    numbers : Tensor
        Tensor of atomic numbers.
    positions : Tensor
        Tensor of atomic positions (shape: ``(nat, 3)`` in atomic units).
    ftype : str | None, optional
        File type. If None, the file extension is used to infer the type.
    mode : str, optional
        Mode in which the file is opened. Defaults to ``"w"``.
    overwrite : bool, optional
        If False and the file exists, a FileExistsError is raised.
    **kwargs : Any
        Additional arguments for the specific writer.

    Raises
    ------
    FileExistsError
        If the file exists and overwrite is False.
    ValueError
        If the file type is unknown.
    NotImplementedError
        If the file type is recognized but no writer is available.
    """
    path = Path(filepath)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"The file '{filepath}' already exists. "
            "If you want to overwrite it, set `overwrite=True`."
        )

    # Infer file type from file extension if not provided.
    if ftype is None and path.suffix:
        ftype = path.suffix.casefold()[1:]

    fname = path.name.casefold()

    if ftype in ("xyz", "log"):
        writer = write_xyz_fileobj
    elif ftype in ("qm9",):
        raise NotImplementedError(
            f"Filetype '{ftype}' (QM9 XYZ) recognized but no writer available."
        )
    elif ftype in ("tmol", "tm", "turbomole") or fname == "coord":
        writer = write_turbomole_fileobj
    elif ftype in ("mol", "sdf", "gen", "pdb"):
        raise NotImplementedError(
            f"Filetype '{ftype}' recognized but no writer available."
        )
    elif ftype in ("qchem",):
        raise NotImplementedError(
            f"Filetype '{ftype}' (Q-Chem) recognized but no writer available."
        )
    elif ftype in ("poscar", "contcar", "vasp", "crystal") or fname in (
        "poscar",
        "contcar",
        "vasp",
    ):
        raise NotImplementedError(
            f"Filetype '{ftype}' (VASP/CRYSTAL) recognized but no writer available."
        )
    elif ftype in ("ein", "gaussian"):
        raise NotImplementedError(
            f"Filetype '{ftype}' (Gaussian) recognized but no writer available."
        )
    elif ftype in ("json", "qcschema"):
        raise NotImplementedError(
            f"Filetype '{ftype}' (QCSchema) recognized but no writer available."
        )
    else:
        raise ValueError(f"Unknown filetype '{ftype}' in '{fname}'.")

    # Open the file and call the selected writer.
    with open(path, mode=mode, encoding="utf-8") as fileobj:
        writer(fileobj, numbers, positions, **kwargs)
