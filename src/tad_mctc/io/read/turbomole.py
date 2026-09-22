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
I/O Read: Turbomole
===================

Reader for Turbomole coordinate (*coord*), energy, and gradient files.

Periodicity (``$periodic``/``$lattice``/``$cell``) mirrors mctc-lib's
``mctc_io_read_turbomole`` (``src/mctc/io/read/turbomole.f90``): a system
can be periodic along 0 to 3 axes (molecule, wire, slab, or bulk), with the
lattice given either as explicit vectors (``$lattice``) or as length/angle
cell parameters (``$cell``, converted to vectors the same way mctc-lib's
``cell_to_dlat`` does). Unlike a VASP POSCAR, most coord files are plain
molecules, so the returned structure carries a lattice and periodicity
mask only when the file declares one or more periodic axes.

For a wire or slab, each non-periodic axis gets a 1 bohr placeholder
lattice vector, so the cell always has a non-zero volume. For ``$cell``
this is what mctc-lib does too; for ``$lattice`` mctc-lib leaves those
rows zero.
"""

from __future__ import annotations

import math
from typing import IO, Any

import torch

from ...convert import symbol_to_number
from ...exceptions import EmptyFileError, FormatErrorTM
from ...typing import Tensor
from ...units import length
from ..structure import Structure
from ._cell import cell_to_lattice
from ._finalize import finalize_geometry, resolve_dd
from .frompath import create_path_reader

__all__ = ["read_coord", "read_turbomole", "read_turbomole_energy"]

# number of real values needed for $cell / $lattice at each periodicity
_CELLPAR_COUNT = {1: 1, 2: 3, 3: 6}


def read_turbomole_fileobj(
    fileobj: IO[Any],
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    dtype_int: torch.dtype = torch.long,
    **kwargs: Any,
) -> Structure:
    """
    Reads a Turbomole coord file into a structure, with lattice vectors and
    a periodicity mask if the file declares one or more periodic axes via
    ``$periodic``.

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
    Structure
        Atomic numbers and positions (shape ``(nat, 3)``, bohr). If the
        file declares a periodic axis, also the lattice (rows are lattice
        vectors, bohr) and the periodicity mask.

    Raises
    ------
    EmptyFileError
        No ``$coord`` section (or no atoms in it) was found.
    FormatErrorTM
        The file does not conform with the expected Turbomole format, or
        mixes ``$periodic``/``$lattice``/``$cell`` inconsistently.
    """
    dd, ddi = resolve_dd(device, dtype, dtype_int)

    lines: list[str] = fileobj.readlines()

    has_coord = has_periodic = has_lattice = has_cell = False
    cartesian = True
    coord_in_bohr = True
    lattice_in_bohr = True
    periodic_dims = 0
    symbols: list[str] = []
    coords: list[list[float]] = []
    lattice_tokens: list[str] = []
    lattice_line_count = 0
    cell_tokens: list[str] = []

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        if not line.startswith("$"):
            i += 1
            continue

        tag_tokens = line.split()
        tag = tag_tokens[0]

        if tag == "$end":
            break

        if tag == "$coord":
            if has_coord:
                raise FormatErrorTM(
                    f"Duplicated $coord data group in '{fileobj}'."
                )
            has_coord = True
            cartesian = "frac" not in tag_tokens
            coord_in_bohr = "angs" not in tag_tokens
            i += 1
            while i < n and not lines[i].startswith("$"):
                try:
                    x, y, z, symbol = lines[i].split()[:4]
                    coords.append([float(x), float(y), float(z)])
                except Exception as e:
                    raise FormatErrorTM(
                        f"Cannot read coordinates from '{fileobj}'."
                    ) from e
                symbols.append(symbol.title().replace("Q", "X"))
                i += 1
            continue

        if tag == "$periodic":
            if has_periodic:
                raise FormatErrorTM(
                    f"Duplicated $periodic data group in '{fileobj}'."
                )
            has_periodic = True
            try:
                periodic_dims = int(tag_tokens[1])
                if periodic_dims not in (0, 1, 2, 3):
                    raise ValueError("periodicity out of range")
            except (IndexError, ValueError) as e:
                raise FormatErrorTM(
                    f"Cannot read periodicity of system from '{fileobj}': "
                    "expected an integer (0 to 3)."
                ) from e
            i += 1
            continue

        if tag == "$lattice":
            if has_lattice:
                raise FormatErrorTM(
                    f"Duplicated $lattice data group in '{fileobj}'."
                )
            has_lattice = True
            lattice_in_bohr = "angs" not in tag_tokens
            i += 1
            lattice_line_count = 0
            while i < n and not lines[i].startswith("$"):
                lattice_tokens.extend(lines[i].split())
                lattice_line_count += 1
                i += 1
            continue

        if tag == "$cell":
            if has_cell:
                raise FormatErrorTM(
                    f"Duplicated $cell data group in '{fileobj}'."
                )
            has_cell = True
            lattice_in_bohr = "angs" not in tag_tokens
            i += 1
            if i < n and not lines[i].startswith("$"):
                cell_tokens = lines[i].split()
                i += 1
            continue

        # unknown tag (e.g. $eht, $redundant, ...): skip its whole group
        i += 1
        while i < n and not lines[i].startswith("$"):
            i += 1

    if not has_coord:
        raise FormatErrorTM(f"No $coord section found in file '{fileobj}'.")
    if not coords:
        raise EmptyFileError(f"No valid data found in the file '{fileobj}'.")

    if has_cell and has_lattice:
        raise FormatErrorTM(
            f"Conflicting $cell and $lattice data groups in '{fileobj}'."
        )
    if not has_periodic or periodic_dims == 0:
        if has_cell or has_lattice:
            raise FormatErrorTM(
                "$cell/$lattice parameters given without a periodic "
                f"($periodic > 0) system in '{fileobj}'."
            )
    elif not (has_cell or has_lattice):
        raise FormatErrorTM(
            f"Periodic system without $cell/$lattice data in '{fileobj}'."
        )
    if not cartesian and periodic_dims == 0:
        raise FormatErrorTM(
            f"Molecular (non-periodic) systems in '{fileobj}' cannot have "
            "fractional coordinates."
        )

    numbers_list = []
    for s in symbols:
        # "X" is this package's own dummy/padding-atom convention (not
        # from mctc-lib, see `convert.pse._resolve_symbol`'s docstring) --
        # resolve it directly rather than through the quirky lookup, which
        # treats a 0 result as unresolved.
        if s == "X":
            numbers_list.append(0)
            continue
        number = symbol_to_number(s)
        if number is None:
            raise FormatErrorTM(f"Unknown element symbol {s!r} in '{fileobj}'.")
        numbers_list.append(number)
    numbers = torch.tensor(numbers_list, **ddi)
    coords_t = torch.tensor(coords, **dd)

    # A non-periodic axis keeps a 1 bohr unit vector as a placeholder, as
    # the `$cell` path below gets from mctc-lib's `cell_to_dlat`. `$lattice`
    # only gives the periodic block, and mctc-lib leaves the rest zero; that
    # singular cell cannot be inverted to fold atoms into the central cell.
    lattice = torch.eye(3, **dd)
    if has_cell:
        conv = 1.0 if lattice_in_bohr else length.AA2AU
        needed = _CELLPAR_COUNT[periodic_dims]
        try:
            values = [float(v) for v in cell_tokens]
            if len(values) < needed:
                raise ValueError("not enough values")
        except ValueError as e:
            raise FormatErrorTM(
                f"Cannot read cell parameters from '{fileobj}': expected "
                f"{needed} real value(s)."
            ) from e

        if periodic_dims == 1:
            cellpar = (values[0] * conv, 1.0, 1.0, 90.0, 90.0, 90.0)
        elif periodic_dims == 2:
            cellpar = (
                values[0] * conv,
                values[1] * conv,
                1.0,
                90.0,
                90.0,
                values[2],
            )
        else:
            cellpar = (
                values[0] * conv,
                values[1] * conv,
                values[2] * conv,
                values[3],
                values[4],
                values[5],
            )
        a, b, c, alpha, beta, gamma = cellpar
        lattice = cell_to_lattice(
            (
                a,
                b,
                c,
                math.radians(alpha),
                math.radians(beta),
                math.radians(gamma),
            ),
            dd,
        )
    elif has_lattice:
        if lattice_line_count != periodic_dims:
            raise FormatErrorTM(
                f"Number of $lattice vectors ({lattice_line_count}) does "
                f"not match periodicity ({periodic_dims}) in '{fileobj}'."
            )
        needed = periodic_dims * periodic_dims
        try:
            values = [float(v) for v in lattice_tokens[:needed]]
            if len(values) < needed:
                raise ValueError("not enough values")
        except ValueError as e:
            raise FormatErrorTM(
                f"Cannot read $lattice vectors from '{fileobj}': expected "
                f"{needed} real value(s)."
            ) from e
        conv = 1.0 if lattice_in_bohr else length.AA2AU
        sub = (
            torch.tensor(values, **dd).reshape(periodic_dims, periodic_dims)
            * conv
        )
        lattice[:periodic_dims, :periodic_dims] = sub

    if cartesian:
        conv = 1.0 if coord_in_bohr else length.AA2AU
        positions = coords_t * conv
    else:
        positions = coords_t.clone()
        positions[:, :periodic_dims] = (
            coords_t[:, :periodic_dims]
            @ lattice[:periodic_dims, :periodic_dims]
        )

    positions = finalize_geometry(numbers, positions, fileobj, **kwargs)

    if periodic_dims == 0:
        return Structure(numbers=numbers, positions=positions)

    periodic = torch.zeros(3, dtype=torch.bool, device=device)
    periodic[:periodic_dims] = True

    return Structure(
        numbers=numbers, positions=positions, lattice=lattice, periodic=periodic
    )


read_turbomole = create_path_reader(read_turbomole_fileobj)


read_coord = create_path_reader(read_turbomole_fileobj)


################################################################################


def read_turbomole_energy_fileobj(
    fileobj: IO[Any],
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """
    Read energy file in TM format (energy is three times on second line).

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
    Tensor
        Tensor of energy

    Raises
    ------
    ValueError
        File does not conform with the expected format.
    """
    first_line: str = fileobj.readline().strip()
    if not first_line or first_line.split()[0] != "$energy":
        raise FormatErrorTM(f"File '{fileobj}' is not in Turbomole format.")

    second_line: str = fileobj.readline().strip()
    if not second_line:
        raise FormatErrorTM(f"File '{fileobj}' is not in Turbomole format.")

    energy_line = second_line.split()
    if len(energy_line) != 4:
        raise FormatErrorTM(f"File '{fileobj}' is not in Turbomole format.")

    return torch.tensor(float(energy_line[1]), device=device, dtype=dtype)


read_turbomole_energy = create_path_reader(read_turbomole_energy_fileobj)
