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
I/O Read: Q-Chem
================

Reader for a Q-Chem ``$molecule`` block. Mirrors mctc-lib's
``mctc_io_read_qchem`` (``src/mctc/io/read/qchem.f90``).

Two atom-line formats are supported, matching Q-Chem itself:

- Cartesian: ``<symbol|number> x y z`` (Angstrom).
- Z-matrix (internal coordinates), one reference atom's worth of data
  richer per atom than the last: a bare first atom (placed at the
  origin), then ``ref1 distance`` for the second atom, ``ref1 distance
  ref2 angle`` for the third, and ``ref1 distance ref2 angle ref3
  dihedral`` for every atom after that -- the same placement scheme as
  mctc-lib's own NERF-style construction (Gram-Schmidt-orthogonalize the
  ref2->ref1 direction against ref3->ref2, rotate by the dihedral around
  their cross product, then by the angle, then scale by the distance).

As with Gaussian's ``.ein`` header, the block's charge/multiplicity line
is parsed (so a malformed one still raises) but not propagated.
"""

from __future__ import annotations

import math
from typing import IO, Any

import torch

from ...convert import symbol_to_number
from ...exceptions import FormatErrorQChem
from ...units import length
from ..structure import Structure
from ._finalize import finalize_geometry, resolve_dd
from .frompath import create_path_reader

__all__ = ["read_qchem"]


def _parse_cartesian_triple(
    tokens: list[str], line: str, fileobj: IO[Any]
) -> list[float]:
    if len(tokens) < 3:
        raise FormatErrorQChem(
            f"Cannot read coordinates from '{fileobj}': {line!r}."
        )
    try:
        values = [float(v) for v in tokens[:3]]
    except ValueError as e:
        raise FormatErrorQChem(
            f"Cannot read coordinates from '{fileobj}': {line!r}."
        ) from e
    return [v * length.AA2AU for v in values]


def _resolve_atom_symbol(token: str, fileobj: IO[Any]) -> int:
    """Resolve an atom token that is either an element symbol or a raw
    atomic number, matching mctc-lib's own symbol-or-integer fallback."""
    number = symbol_to_number(token)
    if number is not None:
        return number
    try:
        number = int(token)
    except ValueError:
        number = None
    if number is None or number <= 0:
        raise FormatErrorQChem(
            f"Cannot map symbol {token!r} to an atomic number in "
            f"'{fileobj}'."
        )
    return number


def _place_zmatrix_atom(
    positions: list[list[float]],
    ij: list[int],
    zm: list[float],
    fileobj: IO[Any],
) -> list[float]:
    """Place one Z-matrix atom, given the reference indices (1-based) and
    distance/angle/dihedral values already collected for it."""
    i1 = ij[0] - 1
    ref1 = positions[i1]

    if len(zm) == 1:
        dist = zm[0] * length.AA2AU
        return [ref1[0] + dist, ref1[1], ref1[2]]

    if len(zm) == 2:
        dist, angle = zm[0] * length.AA2AU, math.radians(zm[1])
        sign = ij[1] - ij[0]
        return [
            ref1[0] + dist * math.cos(angle) * sign,
            ref1[1] + dist * math.sin(angle),
            ref1[2],
        ]

    dist, angle, dihedral = (
        zm[0] * length.AA2AU,
        math.radians(zm[1]),
        math.radians(zm[2]),
    )
    i2, i3 = ij[1] - 1, ij[2] - 1
    if not (0 <= i2 < len(positions)) or not (0 <= i3 < len(positions)):
        raise FormatErrorQChem(
            f"Invalid Z-matrix reference atom index in '{fileobj}'."
        )
    # pinned to float64 regardless of the caller's requested dtype: these
    # are pure-Python doubles coming out of `positions_list`, and passing
    # them through torch's (float32) default dtype would silently throw
    # away precision before the result is cast back via `dd` on return
    p1, p2, p3 = (
        torch.tensor(ref1, dtype=torch.float64),
        torch.tensor(positions[i2], dtype=torch.float64),
        torch.tensor(positions[i3], dtype=torch.float64),
    )

    a12 = p2 - p1
    a12 = a12 / torch.linalg.norm(a12)
    a32 = p2 - p3
    a32 = a32 - a12 * torch.dot(a32, a12)
    a32 = a32 / torch.linalg.norm(a32)

    vec = a32 * math.cos(dihedral) + torch.linalg.cross(a12, a32) * math.sin(
        dihedral
    )
    vec = a12 * math.cos(angle) - vec * math.sin(angle)
    vec = dist / torch.linalg.norm(vec) * vec

    return (p1 + vec).tolist()


def read_qchem_fileobj(
    fileobj: IO[Any],
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    dtype_int: torch.dtype = torch.long,
    **kwargs: Any,
) -> Structure:
    """
    Reads a Q-Chem ``$molecule`` block and returns atomic numbers and
    positions as tensors, from either Cartesian or Z-matrix input.

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
    FormatErrorQChem
        The file does not conform with the expected format.
    """
    dd, ddi = resolve_dd(device, dtype, dtype_int)

    for line in fileobj:
        tokens = line.split()
        if tokens and tokens[0].lower() == "$molecule":
            break
    else:
        raise FormatErrorQChem(f"No '$molecule' block found in '{fileobj}'.")

    header = fileobj.readline().split()
    try:
        int(header[0])
        int(header[1])
    except (IndexError, ValueError) as e:
        raise FormatErrorQChem(
            f"Failed to read charge and multiplicity from '{fileobj}'."
        ) from e

    numbers_list: list[int] = []
    positions_list: list[list[float]] = []

    # The first atom line decides the mode for the *entire* block: if it
    # carries a coordinate triple, every atom is Cartesian; if it is a
    # bare symbol, every subsequent atom is a Z-matrix entry. mctc-lib
    # never mixes the two within one $molecule block.
    line = fileobj.readline()
    tokens = line.split()
    if not tokens or tokens[0].lower() == "$end":
        raise FormatErrorQChem(f"No atoms found in '{fileobj}'.")

    numbers_list.append(_resolve_atom_symbol(tokens[0], fileobj))
    rest = tokens[1:]
    cartesian = bool(rest)
    if cartesian:
        positions_list.append(_parse_cartesian_triple(rest, line, fileobj))
    else:
        positions_list.append([0.0, 0.0, 0.0])

    for line in fileobj:
        tokens = line.split()
        if not tokens:
            continue
        if tokens[0].lower() == "$end":
            break

        numbers_list.append(_resolve_atom_symbol(tokens[0], fileobj))
        rest = tokens[1:]

        if cartesian:
            positions_list.append(_parse_cartesian_triple(rest, line, fileobj))
            continue

        # Z-matrix entry: (index, value) pairs, one per already-established
        # reference atom -- 1 for the second atom, 2 for the third, 3 for
        # every atom from the fourth onward.
        nrefs = min(len(positions_list), 3)
        if len(rest) < 2 * nrefs:
            raise FormatErrorQChem(
                f"Cannot read Z-matrix entry from '{fileobj}': {line!r}."
            )
        try:
            ij = [int(rest[2 * k]) for k in range(nrefs)]
            zm = [float(rest[2 * k + 1]) for k in range(nrefs)]
        except ValueError as e:
            raise FormatErrorQChem(
                f"Cannot read Z-matrix entry from '{fileobj}': {line!r}."
            ) from e

        if ij[0] > len(positions_list):
            raise FormatErrorQChem(
                f"Invalid Z-matrix reference atom index {ij[0]} in "
                f"'{fileobj}': must refer to an earlier atom."
            )

        positions_list.append(
            _place_zmatrix_atom(positions_list, ij, zm, fileobj)
        )
    else:
        raise FormatErrorQChem(
            f"Failed to read molecule block from '{fileobj}': unexpected "
            "end of input."
        )

    numbers = torch.tensor(numbers_list, **ddi)
    positions = torch.tensor(positions_list, **dd)

    positions = finalize_geometry(numbers, positions, fileobj, **kwargs)

    return Structure(numbers=numbers, positions=positions)


read_qchem = create_path_reader(read_qchem_fileobj)
