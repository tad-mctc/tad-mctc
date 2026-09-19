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
I/O Read: PDB
=============

Reader for the Protein Data Bank (PDB) format. Mirrors mctc-lib's
``mctc_io_read_pdb`` (``src/mctc/io/read/pdb.f90``). Named ``pdbfile.py``,
not ``pdb.py``, so that ``from .pdb import ...`` doesn't shadow the
standard library debugger module -- pre-commit's ``debug-statements``
hook flags exactly that import pattern as a leftover breakpoint.

Only ``ATOM``/``HETATM`` records are read (fixed columns, matching the
PDB spec); any other record type (``TER``, ``ANISOU``, ``HEADER``, ...)
is skipped, and reading stops at ``END``. Each record's element-symbol
column (77-78) is tried first; if that is blank or unrecognized, mctc-lib
falls back to the first ``H``/``C``/``N``/``O``/``S``/``P``/``F``
character found in the atom-name column (13-16), which this mirrors.

Per-atom formal charges (column 79-80, e.g. ``"1+"``/``"2-"``) are what
mctc-lib sums into ``struc%charge`` -- unlike Gaussian's/Q-Chem's embedded
charge/multiplicity, mctc-lib's own parsing of this field can never fail
(a missing/malformed charge silently defaults to 0, no error raised), so
there is no validation to faithfully replicate here. Like those other
formats' embedded charge, the value is not propagated: this reader's
tuple API carries no charge field, and ``Structure.charge`` comes from a
separate ``.CHRG`` sidecar file (see ``tad_mctc.io.read.dotfiles``).
"""

from __future__ import annotations

from typing import IO, Any

import torch

from ...convert import symbol_to_number
from ...exceptions import FormatErrorPDB
from ...typing import DD, Tensor, get_default_dtype
from ...units import length
from ..checks import content_checks, deflatable_check, shape_checks
from .frompath import create_path_reader

__all__ = ["read_pdb"]

_FALLBACK_ELEMENTS = "HCNOSPF"


def _read_fixed_float(line: str, start: int, end: int) -> float:
    """Parse a fixed-column numeric field the way Fortran formatted I/O
    does: embedded blanks within the field are ignored (not just leading
    /trailing whitespace), which real PDB files rely on -- mctc-lib's own
    ``token_type(60, 66)`` for the temperature factor overlaps the
    occupancy field by one column, so the field it reads often looks like
    ``"0 39.42"``."""
    return float(line[start:end].replace(" ", ""))


def _resolve_pdb_symbol(
    element_field: str, name_field: str, fileobj: IO[Any]
) -> int:
    number = symbol_to_number(element_field)
    if number is None:
        for char in name_field:
            if char in _FALLBACK_ELEMENTS:
                number = symbol_to_number(char)
                break
    if number is None:
        raise FormatErrorPDB(
            f"Cannot map symbol {element_field!r} to an atomic number in "
            f"'{fileobj}'."
        )
    return number


def read_pdb_fileobj(
    fileobj: IO[Any],
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    dtype_int: torch.dtype = torch.long,
    **kwargs: Any,
) -> tuple[Tensor, Tensor]:
    """
    Reads a PDB file and returns atomic numbers and positions as tensors.

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
        Tensors of atomic numbers and positions. Positions is a tensor of
        shape (nat, 3) in atomic units.

    Raises
    ------
    FormatErrorPDB
        The file does not conform with the expected fixed-column format.
    """
    dd: DD = {
        "device": device,
        "dtype": dtype if dtype is not None else get_default_dtype(),
    }
    ddi: DD = {"device": device, "dtype": dtype_int}

    numbers_list: list[int] = []
    coords: list[list[float]] = []

    for raw_line in fileobj:
        line = raw_line.rstrip("\n")
        if line.startswith("END"):
            break
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue

        if len(line) < 78:
            raise FormatErrorPDB(
                f"Too few entries provided in record from '{fileobj}': "
                f"{line!r}."
            )

        name_field = line[12:16]
        element_field = line[76:78]

        try:
            x = _read_fixed_float(line, 30, 38)
            y = _read_fixed_float(line, 38, 46)
            z = _read_fixed_float(line, 46, 54)
            # occupancy/temperature factor are validated (mctc-lib itself
            # requires them to parse) but never used downstream
            _read_fixed_float(line, 54, 60)
            _read_fixed_float(line, 59, 66)
        except ValueError as e:
            raise FormatErrorPDB(
                f"Cannot read coordinates from record in '{fileobj}': "
                f"{line!r}."
            ) from e

        numbers_list.append(
            _resolve_pdb_symbol(element_field, name_field, fileobj)
        )
        coords.append([x, y, z])

    if not numbers_list:
        raise FormatErrorPDB(f"No atoms found in '{fileobj}'.")

    numbers = torch.tensor(numbers_list, **ddi)
    positions = torch.tensor(coords, **dd) * length.AA2AU

    assert shape_checks(numbers, positions, allow_batched=False)
    assert content_checks(
        numbers,
        positions,
        allow_batched=False,
        check_coldfusion=kwargs.get("check_coldfusion", False),
        coldfusion_cutoff=kwargs.get("coldfusion_cutoff", 2.0),
    )
    assert deflatable_check(positions, fileobj, **kwargs)

    return numbers, positions


read_pdb = create_path_reader(read_pdb_fileobj)
