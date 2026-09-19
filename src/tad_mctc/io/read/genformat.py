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
I/O Read: DFTB+ genFormat
=========================

Reader for the DFTB+ ``genFormat`` (``.gen``) format. Mirrors mctc-lib's
``mctc_io_read_genformat`` (``src/mctc/io/read/genformat.f90``) for its
cluster (``C``, non-periodic), supercell (``S``, cartesian + lattice) and
fractional (``F``, fractional + lattice) modes.

mctc-lib's fourth mode, ``H`` (helical), encodes 1D screw-axis symmetry
(translation distance, rotation angle, rotation order) in the same slot
a real lattice vector would occupy for the other modes -- ``Structure``
and the neighbour-list code have no representation for screw symmetry,
so a helical file is rejected with a clear error rather than silently
returned as a bogus lattice vector.

mctc-lib (as of writing) reads the periodic "origin" line (Angstrom, per
the format spec) but never converts it to bohr before subtracting it
from the already-converted positions -- a genuine bug (same unit-mismatch
pattern as the former cjson fractional-coordinate bug), silent only
because real ``.gen`` files almost always use a zero origin. Fixed here
rather than ported as-is, and reported upstream.
"""

from __future__ import annotations

from typing import IO, Any, Iterator

import torch

from ...convert import symbol_to_number
from ...exceptions import FormatErrorGenFormat
from ...typing import DD, Tensor, get_default_dtype
from ...units import length
from ..checks import content_checks, deflatable_check, shape_checks
from .frompath import create_path_reader_periodic

__all__ = ["read_genformat"]


def _iter_meaningful_lines(fileobj: IO[Any]) -> Iterator[str]:
    """Strip ``#``-onward comments and skip resulting blank lines."""
    for line in fileobj:
        line = line.split("#", 1)[0].strip()
        if line:
            yield line


def _next_line(lines: Iterator[str], fileobj: IO[Any], context: str) -> str:
    try:
        return next(lines)
    except StopIteration as e:
        raise FormatErrorGenFormat(
            f"Unexpected end of file '{fileobj}' while reading {context}."
        ) from e


def read_genformat_fileobj(
    fileobj: IO[Any],
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    dtype_int: torch.dtype = torch.long,
    **kwargs: Any,
) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Reads a DFTB+ genFormat (.gen) file and returns atomic numbers and
    positions as tensors, plus lattice vectors and a periodicity mask for
    a supercell/fractional (periodic) file.

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
    (Tensor, Tensor) | (Tensor, Tensor, Tensor, Tensor)
        Tensors of atomic numbers and positions (shape ``(nat, 3)``,
        atomic units). A supercell/fractional file additionally carries a
        lattice tensor (shape ``(3, 3)``, rows are lattice vectors in
        bohr) and an all-``True`` periodicity mask (shape ``(3,)``).

    Raises
    ------
    FormatErrorGenFormat
        The file does not conform with the expected format, or uses the
        unsupported helical (``H``) mode.
    """
    dd: DD = {
        "device": device,
        "dtype": dtype if dtype is not None else get_default_dtype(),
    }
    ddi: DD = {"device": device, "dtype": dtype_int}

    lines = _iter_meaningful_lines(fileobj)

    header = _next_line(lines, fileobj, "the number of atoms").split()
    try:
        natoms = int(header[0])
        if natoms < 1:
            raise ValueError("non-positive atom count")
    except (IndexError, ValueError) as e:
        raise FormatErrorGenFormat(
            f"Could not read number of atoms from '{fileobj}'."
        ) from e

    try:
        mode = header[1].lower()
    except IndexError as e:
        raise FormatErrorGenFormat(
            f"Missing mode identifier in '{fileobj}'."
        ) from e

    if mode == "c":
        cartesian, periodic_flag = True, False
    elif mode == "s":
        cartesian, periodic_flag = True, True
    elif mode == "f":
        cartesian, periodic_flag = False, True
    elif mode == "h":
        raise FormatErrorGenFormat(
            f"Helical (screw-axis) periodicity in '{fileobj}' is not "
            "supported; Structure.lattice has no representation for it."
        )
    else:
        raise FormatErrorGenFormat(
            f"Invalid input version {header[1]!r} in '{fileobj}'."
        )

    # `_iter_meaningful_lines` never yields a blank/comment-only line, so
    # `.split()` here always has at least one token.
    species_tokens = _next_line(lines, fileobj, "the species line").split()
    species_numbers = []
    for token in species_tokens:
        number = symbol_to_number(token)
        if number is None:
            raise FormatErrorGenFormat(
                f"Unknown element symbol {token!r} in '{fileobj}'."
            )
        species_numbers.append(number)

    numbers_list: list[int] = []
    coords: list[list[float]] = []
    for _ in range(natoms):
        tokens = _next_line(lines, fileobj, "atomic coordinates").split()
        try:
            isp = int(tokens[1])
            x, y, z = (float(v) for v in tokens[2:5])
        except (IndexError, ValueError) as e:
            raise FormatErrorGenFormat(
                f"Cannot read coordinates from '{fileobj}'."
            ) from e
        if not 1 <= isp <= len(species_numbers):
            raise FormatErrorGenFormat(
                f"Invalid species index {isp} in '{fileobj}'."
            )
        numbers_list.append(species_numbers[isp - 1])
        coords.append([x, y, z])

    numbers = torch.tensor(numbers_list, **ddi)
    coords_t = torch.tensor(coords, **dd)

    if not periodic_flag:
        positions = coords_t * length.AA2AU

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

    origin_tokens = _next_line(lines, fileobj, "the origin").split()
    try:
        origin = [float(v) for v in origin_tokens[:3]]
        if len(origin) < 3:
            raise ValueError("not enough values")
    except ValueError as e:
        raise FormatErrorGenFormat(
            f"Cannot read origin from '{fileobj}'."
        ) from e

    lattice_rows = []
    for _ in range(3):
        row_tokens = _next_line(lines, fileobj, "a lattice vector").split()
        try:
            row = [float(v) for v in row_tokens[:3]]
            if len(row) < 3:
                raise ValueError("not enough values")
        except ValueError as e:
            raise FormatErrorGenFormat(
                f"Cannot read lattice vector from '{fileobj}'."
            ) from e
        lattice_rows.append(row)

    lattice = torch.tensor(lattice_rows, **dd) * length.AA2AU
    # origin is given in the same Angstrom unit as the coordinates and
    # lattice vectors above (see the module docstring)
    origin_t = torch.tensor(origin, **dd) * length.AA2AU

    if cartesian:
        positions = coords_t * length.AA2AU
    else:
        positions = coords_t @ lattice
    positions = positions - origin_t

    periodic = torch.ones(3, dtype=torch.bool, device=device)

    assert shape_checks(numbers, positions, allow_batched=False)
    assert content_checks(
        numbers,
        positions,
        allow_batched=False,
        check_coldfusion=kwargs.get("check_coldfusion", False),
        coldfusion_cutoff=kwargs.get("coldfusion_cutoff", 2.0),
    )
    assert deflatable_check(positions, fileobj, **kwargs)

    return numbers, positions, lattice, periodic


read_genformat = create_path_reader_periodic(read_genformat_fileobj)
