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
from typing import IO, Any

import torch

from ...typing import PathLike, Tensor
from ..structure import Structure
from .aims import read_aims_fileobj
from .cjson import read_cjson_fileobj
from .ctfile import read_molfile_fileobj, read_sdf_fileobj
from .dotfiles import read_chrg, read_uhf
from .frompath import JSONResult
from .gaussian import read_gaussian_fileobj
from .genformat import read_genformat_fileobj
from .json import read_json_fileobj
from .pdbfile import read_pdb_fileobj
from .pymatgen import read_pymatgen_fileobj
from .qchem import read_qchem_fileobj
from .qcschema import read_qcschema_fileobj
from .turbomole import read_turbomole_fileobj
from .vasp import read_poscar_fileobj
from .xyz import read_xyz_fileobj, read_xyz_qm9_fileobj

__all__ = ["read", "read_structure"]


def read_from_fileobj(
    fileobj: IO[Any],
    ftype: str,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    dtype_int: torch.dtype = torch.long,
    **kwargs: Any,
) -> (
    tuple[Tensor, Tensor]
    | tuple[Tensor, Tensor, Tensor]
    | tuple[Tensor, Tensor, Tensor, Tensor]
    | JSONResult
):
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
    (Tensor, Tensor) | (Tensor, Tensor, Tensor) | (Tensor, Tensor, Tensor, Tensor)
        (Possibly batched) tensors of atomic numbers and positions. Positions
        is a tensor of shape (batch_size, nat, 3) in atomic units. A VASP
        POSCAR/CONTCAR file additionally carries lattice vectors as a third
        tensor of shape (3, 3), also in atomic units. A periodic Turbomole
        coord file (``$periodic`` > 0) additionally carries a periodicity
        mask as a fourth tensor of shape (3,).

    Raises
    ------
    NotImplementedError
        Reader for specific file type not implemented.
    ValueError
        Unknown file type.
    """
    # path stored in TextIOWrapper
    fname = Path(fileobj.name).name.lower()

    if ftype in ("xyz", "log", "extxyz"):
        return read_xyz_fileobj(
            fileobj, device=device, dtype=dtype, dtype_int=dtype_int, **kwargs
        )
    if ftype in ("qm9",):
        return read_xyz_qm9_fileobj(
            fileobj, device=device, dtype=dtype, dtype_int=dtype_int, **kwargs
        )
    # mctc-lib's `get_filetype` maps both the *extension* ".coord" and the
    # bare *filename* "coord" (no extension) to its `tmol` filetype -- the
    # extension case was missing here.
    if ftype in ("tmol", "tm", "turbomole", "coord") or fname == "coord":
        return read_turbomole_fileobj(
            fileobj, device=device, dtype=dtype, dtype_int=dtype_int, **kwargs
        )
    if ftype in ("aims",) or fname == "geometry.in":
        return read_aims_fileobj(
            fileobj, device=device, dtype=dtype, dtype_int=dtype_int, **kwargs
        )
    if ftype in ("mol",):
        return read_molfile_fileobj(
            fileobj, device=device, dtype=dtype, dtype_int=dtype_int, **kwargs
        )
    if ftype in ("sdf",):
        return read_sdf_fileobj(
            fileobj, device=device, dtype=dtype, dtype_int=dtype_int, **kwargs
        )
    if ftype in ("gen",):
        return read_genformat_fileobj(
            fileobj, device=device, dtype=dtype, dtype_int=dtype_int, **kwargs
        )
    if ftype in ("pdb",):
        return read_pdb_fileobj(
            fileobj, device=device, dtype=dtype, dtype_int=dtype_int, **kwargs
        )
    if ftype in ("qchem",):
        return read_qchem_fileobj(
            fileobj, device=device, dtype=dtype, dtype_int=dtype_int, **kwargs
        )
    if ftype in ("poscar", "contcar", "vasp") or fname in (
        "poscar",
        "contcar",
    ):
        return read_poscar_fileobj(
            fileobj, device=device, dtype=dtype, dtype_int=dtype_int, **kwargs
        )
    if ftype in ("crystal",):
        raise NotImplementedError(
            f"Filetype '{ftype}' (CRYSTAL) recognized but no reader available."
        )
    if ftype in ("ein", "gaussian"):
        return read_gaussian_fileobj(
            fileobj, device=device, dtype=dtype, dtype_int=dtype_int, **kwargs
        )
    if ftype in ("qcschema", "qcjson"):
        return read_qcschema_fileobj(
            fileobj, device=device, dtype=dtype, dtype_int=dtype_int, **kwargs
        )
    if ftype in ("json",):
        return read_json_fileobj(
            fileobj, device=device, dtype=dtype, dtype_int=dtype_int, **kwargs
        )
    if ftype in ("pmgjson", "pymatgen"):
        return read_pymatgen_fileobj(
            fileobj, device=device, dtype=dtype, dtype_int=dtype_int, **kwargs
        )
    if ftype in ("cjson",):
        return read_cjson_fileobj(
            fileobj, device=device, dtype=dtype, dtype_int=dtype_int, **kwargs
        )

    raise ValueError(f"Unknown filetype '{ftype}' in '{fileobj}'.")


def read(
    filepath: PathLike,
    ftype: str | None = None,
    mode: str = "r",
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    dtype_int: torch.dtype = torch.long,
    **kwargs: Any,
) -> (
    tuple[Tensor, Tensor]
    | tuple[Tensor, Tensor, Tensor]
    | tuple[Tensor, Tensor, Tensor, Tensor]
    | JSONResult
):
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
    (Tensor, Tensor) | (Tensor, Tensor, Tensor) | (Tensor, Tensor, Tensor, Tensor)
        (Possibly batched) tensors of atomic numbers and positions. Positions
        is a tensor of shape (batch_size, nat, 3) in atomic units. A VASP
        POSCAR/CONTCAR file additionally carries lattice vectors as a third
        tensor of shape (3, 3), also in atomic units. A periodic Turbomole
        coord file (``$periodic`` > 0) additionally carries a periodicity
        mask as a fourth tensor of shape (3,).

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


def read_structure(
    filepath: PathLike,
    ftype: str | None = None,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    dtype_int: torch.dtype = torch.long,
    check_coldfusion: bool = False,
) -> Structure:
    """
    Read a structure from a file, picking up its ``.CHRG``/``.UHF``
    sidecar files from the same directory if present. Mirrors mctc-lib's
    ``read_structure``; replaces the removed ``Mol.from_path``.

    Parameters
    ----------
    filepath : PathLike
        Path of the file containing the structure.
    ftype : str | None, optional
        File type. Defaults to ``None``, i.e., inferred from the extension.
    device : :class:`torch.device` | None, optional
        Device to store the tensors on. Defaults to ``None``.
    dtype : :class:`torch.dtype` | None, optional
        Floating point data type of the tensors. Defaults to ``None``.
    dtype_int : :class:`torch.dtype`, optional
        Integer data type of the tensors. Defaults to ``torch.long``.
    check_coldfusion : bool, optional
        Run the interatomic-distance sanity check while reading. Defaults
        to ``False``: this check builds its own neighbour list (see
        :func:`tad_mctc.io.checks.coldfusion_check`) on the CPU, before
        ``device`` is applied, so it can dominate read time for a large
        structure; pass ``True`` to opt in for an untrusted geometry.

    Returns
    -------
    Structure
        The structure's atomic numbers, positions, charge, number of
        unpaired electrons and, for a periodic file (VASP POSCAR/CONTCAR,
        always; a Turbomole/aims/genformat/pymatgen file with periodicity
        declared), its lattice and periodicity mask. A cjson file
        additionally carries bond connectivity/order if present.

    Raises
    ------
    FileNotFoundError
        Given file does not exist.
    """
    result = read(
        filepath,
        ftype=ftype,
        device=device,
        dtype=dtype,
        dtype_int=dtype_int,
        check_coldfusion=check_coldfusion,
    )
    periodic = None
    bonds = None
    bond_orders = None
    if len(result) == 6:
        numbers, positions, lattice, periodic, bonds, bond_orders = result
    elif len(result) == 4:
        numbers, positions, lattice, periodic = result
    elif len(result) == 3:
        numbers, positions, lattice = result
    else:
        numbers, positions = result
        lattice = None

    charge = read_chrg(filepath, device=device, dtype=dtype)
    uhf = read_uhf(filepath, device=device, dtype=dtype_int)
    # `uhf` counts unpaired electrons, so it stays an integer tensor
    # (matching the mstore mirror's `torch.tensor(1)`-style records),
    # unlike `charge`, which keeps `Mol.charge`'s float representation.

    return Structure(
        numbers=numbers,
        positions=positions,
        charge=charge,
        uhf=uhf,
        lattice=lattice,
        periodic=periodic,
        bonds=bonds,
        bond_orders=bond_orders,
    )
