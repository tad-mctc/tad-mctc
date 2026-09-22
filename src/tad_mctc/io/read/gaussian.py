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
I/O Read: Gaussian
==================

Reader for the Gaussian external-program (``.ein``) format. Mirrors
mctc-lib's ``mctc_io_read_gaussian`` (``src/mctc/io/read/gaussian.f90``).

The format is fixed-column (Fortran ``(4I10)`` header, ``(I10,4F20.12)``
per atom line), so fields are sliced by column rather than split on
whitespace -- real Gaussian output can abut large or negative numbers
without a separating space. Coordinates are already in bohr (Gaussian's
external interface, unlike most geometry formats, never uses Angstrom).

The header also carries a charge and spin multiplicity; both are parsed
(so a malformed header still raises) but, unlike lattice/periodicity in
the VASP/Turbomole/aims readers, not propagated -- see the module's tests
for why: threading them through would collide with the existing
(numbers, positions, lattice, periodic) 4-tuple meaning shared by
Turbomole and aims. A caller that needs them uses ``.CHRG``/``.UHF``
sidecar files, as for every other format.
"""

from __future__ import annotations

from typing import IO, Any

import torch

from ...exceptions import FormatErrorGaussian
from ..structure import Structure
from ._finalize import finalize_geometry, resolve_dd
from .frompath import create_path_reader

__all__ = ["read_gaussian"]


def _read_int(
    line: str, start: int, end: int, fileobj: IO[Any], context: str
) -> int:
    field = line[start:end].strip()
    try:
        return int(field)
    except ValueError as e:
        raise FormatErrorGaussian(
            f"Cannot read {context} from '{fileobj}': {field!r} is not an "
            "integer."
        ) from e


def _read_float(
    line: str, start: int, end: int, fileobj: IO[Any], context: str
) -> float:
    field = line[start:end].strip()
    try:
        return float(field)
    except ValueError as e:
        raise FormatErrorGaussian(
            f"Cannot read {context} from '{fileobj}': {field!r} is not a "
            "real value."
        ) from e


def read_gaussian_fileobj(
    fileobj: IO[Any],
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    dtype_int: torch.dtype = torch.long,
    **kwargs: Any,
) -> Structure:
    """
    Reads a Gaussian external-program (``.ein``) file and returns atomic
    numbers and positions as tensors. Positions are already in bohr.

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
        Atomic numbers and positions (shape ``(nat, 3)``, bohr).

    Raises
    ------
    FormatErrorGaussian
        The file does not conform with the expected fixed-column format.
    """
    dd, ddi = resolve_dd(device, dtype, dtype_int)

    header = fileobj.readline()
    if not header:
        raise FormatErrorGaussian(f"File '{fileobj}' is empty.")

    natoms = _read_int(header, 0, 10, fileobj, "number of atoms")
    # columns 11-20 (calculation mode) are not needed downstream
    _read_int(header, 20, 30, fileobj, "charge")
    _read_int(header, 30, 40, fileobj, "spin multiplicity")

    if natoms <= 0:
        raise FormatErrorGaussian(
            f"Found no atoms in '{fileobj}': expected a positive atom count."
        )

    numbers_list: list[int] = []
    coords: list[list[float]] = []
    for _ in range(natoms):
        line = fileobj.readline()
        if not line:
            break

        iat = _read_int(line, 0, 10, fileobj, "atomic number")
        x = _read_float(line, 10, 30, fileobj, "geometry")
        y = _read_float(line, 30, 50, fileobj, "geometry")
        z = _read_float(line, 50, 70, fileobj, "geometry")
        # columns 71-90 (partial charge) are parsed to match mctc-lib's
        # strictness but not used downstream
        _read_float(line, 70, 90, fileobj, "geometry")

        if iat <= 0:
            raise FormatErrorGaussian(
                f"Invalid atomic number {iat} in '{fileobj}': expected a "
                "positive integer."
            )

        numbers_list.append(iat)
        coords.append([x, y, z])

    if len(numbers_list) != natoms:
        raise FormatErrorGaussian(
            f"Atom count mismatch in '{fileobj}': header declared "
            f"{natoms} atoms but only {len(numbers_list)} were found."
        )

    numbers = torch.tensor(numbers_list, **ddi)
    positions = torch.tensor(coords, **dd)

    positions = finalize_geometry(numbers, positions, fileobj, **kwargs)

    return Structure(numbers=numbers, positions=positions)


read_gaussian = create_path_reader(read_gaussian_fileobj)
