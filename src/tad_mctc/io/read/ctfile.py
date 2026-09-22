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
I/O Read: MDL Molfile / SDF
============================

Reader for the MDL Molfile (``.mol``, V2000 and V3000) and SDF (``.sdf``)
formats. Mirrors mctc-lib's ``mctc_io_read_ctfile``
(``src/mctc/io/read/ctfile.f90``).

A Molfile never carries a periodic lattice, so the returned structure has
no ``lattice``/``periodic``; ``bonds``/``bond_orders`` are always set,
since the format always declares a bond count in its header (even if 0).

V2000's per-atom isotope/charge/hydrogen-count/valence columns and
V2000's ``M  CHG``/V3000's ``CHG=``/``VAL=``/``HCOUNT=`` atom properties
are SDF metadata mctc-lib attaches to its own ``structure_type%sdf``
side-table; nothing in ``Structure`` corresponds to that, so (like every
other reader's embedded charge/multiplicity) they are skipped entirely
rather than parsed-and-dropped -- there is no per-field validation to
preserve, since none of it is fixed-width-critical to the fields this
reader does propagate (coordinates, symbols, bonds).
"""

from __future__ import annotations

from typing import IO, Any, Iterator

import torch

from ...convert import symbol_to_number
from ...exceptions import FormatErrorCTFile
from ...units import length
from ..structure import Structure
from ._finalize import finalize_geometry, resolve_dd
from .frompath import create_path_reader

__all__ = ["read_molfile", "read_sdf"]


def _next_line(lines: Iterator[str], fileobj: IO[Any], context: str) -> str:
    try:
        return next(lines).rstrip("\n")
    except StopIteration as e:
        raise FormatErrorCTFile(
            f"Unexpected end of file '{fileobj}' while reading {context}."
        ) from e


def _read_header(
    lines: Iterator[str], fileobj: IO[Any]
) -> tuple[int, int, bool]:
    """Read the 4-line Molfile header (title, program info, comment,
    counts line) and return (number_of_atoms, number_of_bonds, is_v3000).
    The title/program-info/comment lines carry no field ``Structure`` has
    a home for, so they are consumed but not returned."""
    _next_line(lines, fileobj, "the title line")
    _next_line(lines, fileobj, "the program info line")
    _next_line(lines, fileobj, "the comment line")
    counts_line = _next_line(lines, fileobj, "the counts line")

    try:
        number_of_atoms = int(counts_line[0:3])
        number_of_bonds = int(counts_line[3:6])
    except ValueError as e:
        raise FormatErrorCTFile(
            f"Cannot read header of molfile from '{fileobj}'."
        ) from e

    version = counts_line[34:39]
    is_v3000 = version == "V3000"
    if version not in ("V2000", "V3000"):
        raise FormatErrorCTFile(
            f"Format version {version!r} in '{fileobj}' is not supported."
        )
    if not is_v3000 and number_of_atoms < 1:
        raise FormatErrorCTFile(
            f"Invalid number of atoms in '{fileobj}': expected a "
            "positive integer."
        )

    return number_of_atoms, number_of_bonds, is_v3000


def _read_v2000_body(
    lines: Iterator[str],
    fileobj: IO[Any],
    number_of_atoms: int,
    number_of_bonds: int,
) -> tuple[list[int], list[list[float]], list[list[int]], list[float]]:
    numbers_list: list[int] = []
    coords: list[list[float]] = []

    for _ in range(number_of_atoms):
        line = _next_line(lines, fileobj, "an atom record")
        try:
            x = float(line[0:10])
            y = float(line[10:20])
            z = float(line[20:30])
        except ValueError as e:
            raise FormatErrorCTFile(
                f"Cannot read coordinates from connection table in "
                f"'{fileobj}'."
            ) from e
        symbol = line[31:34]
        number = symbol_to_number(symbol)
        if number is None:
            raise FormatErrorCTFile(
                f"Cannot map symbol {symbol!r} to an atomic number in "
                f"'{fileobj}'."
            )
        numbers_list.append(number)
        coords.append([x, y, z])

    bonds_list: list[list[int]] = []
    orders_list: list[float] = []
    for _ in range(number_of_bonds):
        line = _next_line(lines, fileobj, "a bond record")
        try:
            atom1 = int(line[0:3])
            atom2 = int(line[3:6])
            bond_type = int(line[6:9])
        except ValueError as e:
            raise FormatErrorCTFile(
                f"Cannot read topology from connection table in "
                f"'{fileobj}'."
            ) from e
        bonds_list.append([atom1 - 1, atom2 - 1])
        orders_list.append(float(bond_type))

    # properties block: only its end marker matters here (see module
    # docstring for why per-property parsing, e.g. "M  CHG", is skipped)
    while True:
        line = _next_line(lines, fileobj, "the properties block")
        if line.startswith("M  END"):
            break

    return numbers_list, coords, bonds_list, orders_list


def _v30_content(line: str) -> str | None:
    if line.startswith("M  V30 "):
        return line[7:]
    return None


def _read_v3000_end(lines: Iterator[str], fileobj: IO[Any], group: str) -> None:
    line = _next_line(lines, fileobj, f"the end of the {group} block")
    content = _v30_content(line)
    tokens = content.split() if content else []
    if tokens[:2] != ["END", group]:
        raise FormatErrorCTFile(
            f"{group} block is not terminated in '{fileobj}'."
        )


def _read_v3000_body(
    lines: Iterator[str], fileobj: IO[Any]
) -> tuple[list[int], list[list[float]], list[list[int]], list[float]]:
    while True:
        line = _next_line(lines, fileobj, "the CTAB header")
        content = _v30_content(line)
        if content is not None and content.split()[:2] == ["BEGIN", "CTAB"]:
            break

    counts_line = _v30_content(
        _next_line(lines, fileobj, "the connection table counts")
    )
    counts_tokens = counts_line.split() if counts_line else []
    if not counts_tokens or counts_tokens[0] != "COUNTS":
        raise FormatErrorCTFile(f"COUNTS header not found in '{fileobj}'.")
    try:
        # mctc-lib reads 5 integer fields after "COUNTS": atom count, bond
        # count, and 3 further fields it never uses (sgroup/3d-obj counts,
        # chiral flag) -- but still requires them to parse as integers, so
        # a malformed field there is a format error even though its value
        # is otherwise dropped.
        number_of_atoms = int(counts_tokens[1])
        number_of_bonds = int(counts_tokens[2])
        for extra in counts_tokens[3:6]:
            int(extra)
    except (IndexError, ValueError) as e:
        raise FormatErrorCTFile(
            f"Cannot read connection table counts from '{fileobj}'."
        ) from e
    if number_of_atoms < 1:
        raise FormatErrorCTFile(
            f"Invalid number of atoms in '{fileobj}': expected a "
            "positive integer."
        )

    numbers_list: list[int] = []
    coords: list[list[float]] = []
    bonds_list: list[list[int]] = []
    orders_list: list[float] = []

    while True:
        line = _next_line(lines, fileobj, "the connection table")
        content = _v30_content(line)
        tokens = content.split() if content else []
        if not tokens:
            continue
        if tokens[:2] == ["END", "CTAB"]:
            break
        if tokens[0] != "BEGIN":
            continue

        group = tokens[1] if len(tokens) > 1 else ""
        if group == "ATOM":
            for _ in range(number_of_atoms):
                atom_content = _v30_content(
                    _next_line(lines, fileobj, "an atom record")
                )
                atoms_tokens = atom_content.split() if atom_content else []
                try:
                    symbol = atoms_tokens[1]
                    x, y, z = (float(v) for v in atoms_tokens[2:5])
                    aamap = int(atoms_tokens[5])
                except (IndexError, ValueError) as e:
                    raise FormatErrorCTFile(
                        f"Cannot read coordinates from '{fileobj}'."
                    ) from e
                if aamap > 0:
                    raise FormatErrorCTFile(
                        f"Mapping atoms is not supported in '{fileobj}'."
                    )
                number = symbol_to_number(symbol)
                if number is None:
                    raise FormatErrorCTFile(
                        f"Cannot map symbol {symbol!r} to an atomic "
                        f"number in '{fileobj}'."
                    )
                numbers_list.append(number)
                coords.append([x, y, z])
            _read_v3000_end(lines, fileobj, "ATOM")
        elif group == "BOND":
            for _ in range(number_of_bonds):
                bond_content = _v30_content(
                    _next_line(lines, fileobj, "a bond record")
                )
                bond_tokens = bond_content.split() if bond_content else []
                try:
                    bond_type = int(bond_tokens[1])
                    atom1 = int(bond_tokens[2])
                    atom2 = int(bond_tokens[3])
                except (IndexError, ValueError) as e:
                    raise FormatErrorCTFile(
                        f"Cannot read bond information from '{fileobj}'."
                    ) from e
                bonds_list.append([atom1 - 1, atom2 - 1])
                orders_list.append(float(bond_type))
            _read_v3000_end(lines, fileobj, "BOND")
        elif group in ("COLLECTION", "SGROUP", "OBJ3D"):
            while True:
                skip_content = _v30_content(
                    _next_line(lines, fileobj, f"the {group} block")
                )
                skip_tokens = skip_content.split() if skip_content else []
                if skip_tokens[:1] == ["END"]:
                    break
        else:
            raise FormatErrorCTFile(
                f"Unknown connection table entry {group!r} in '{fileobj}'."
            )

    return numbers_list, coords, bonds_list, orders_list


def read_molfile_fileobj(
    fileobj: IO[Any],
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    dtype_int: torch.dtype = torch.long,
    **kwargs: Any,
) -> Structure:
    """
    Reads an MDL Molfile (V2000 or V3000) into a structure with bond
    indices and bond orders. Molfiles are never periodic.

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
        Atomic numbers, positions (bohr), bonds and bond orders.

    Raises
    ------
    FormatErrorCTFile
        The file does not conform with the expected Molfile format.
    """
    dd, ddi = resolve_dd(device, dtype, dtype_int)

    lines = iter(fileobj)
    number_of_atoms, number_of_bonds, is_v3000 = _read_header(lines, fileobj)

    if is_v3000:
        numbers_list, coords, bonds_list, orders_list = _read_v3000_body(
            lines, fileobj
        )
    else:
        numbers_list, coords, bonds_list, orders_list = _read_v2000_body(
            lines, fileobj, number_of_atoms, number_of_bonds
        )

    numbers = torch.tensor(numbers_list, **ddi)
    positions = torch.tensor(coords, **dd) * length.AA2AU
    bonds = torch.tensor(bonds_list, **ddi).reshape(-1, 2)
    bond_orders = torch.tensor(orders_list, **dd)

    positions = finalize_geometry(numbers, positions, fileobj, **kwargs)

    return Structure(
        numbers=numbers,
        positions=positions,
        bonds=bonds,
        bond_orders=bond_orders,
    )


def read_sdf_fileobj(
    fileobj: IO[Any],
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    dtype_int: torch.dtype = torch.long,
    **kwargs: Any,
) -> Structure:
    """
    Reads the first record of an SDF file (a Molfile connection table
    plus a trailing key-value data block terminated by ``$$$$``) and
    returns it exactly like :func:`read_molfile_fileobj`. A multi-record
    SDF file's later records are not read, matching mctc-lib's own
    ``read_sdf``.

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
        See :func:`read_molfile_fileobj`.

    Raises
    ------
    FormatErrorCTFile
        The file does not conform with the expected Molfile format, or
        the SDF data block is not terminated with ``$$$$``.
    """
    result = read_molfile_fileobj(
        fileobj, device=device, dtype=dtype, dtype_int=dtype_int, **kwargs
    )

    for line in fileobj:
        if line.startswith("$$$$"):
            return result

    raise FormatErrorCTFile(
        f"Failed while reading SDF key-value pairs in '{fileobj}'."
    )


read_molfile = create_path_reader(read_molfile_fileobj)


read_sdf = create_path_reader(read_sdf_fileobj)
