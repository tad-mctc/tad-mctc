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
I/O Read: FHI-aims
==================

Reader for FHI-aims ``geometry.in`` files. Mirrors mctc-lib's
``mctc_io_read_aims`` (``src/mctc/io/read/aims.f90``).

Periodicity is implicit in the number of ``lattice_vector`` lines (0 to 3)
rather than an explicit declaration like Turbomole's ``$periodic``; the
returned structure carries a lattice and periodicity mask only once at
least one ``lattice_vector`` line is present.

Element symbols follow mctc-lib's own quirky ``to_number``: only the first
two alphabetic characters are kept (so ``18O`` -> ``O``, ``C*`` -> ``C``),
and the isotope labels ``D``/``T`` resolve to hydrogen.
"""

from __future__ import annotations

from typing import IO, Any

import torch

from ...convert import symbol_to_number
from ...exceptions import FormatErrorAIMS
from ...units import length
from ..structure import Structure
from ._finalize import finalize_geometry, resolve_dd
from .frompath import create_path_reader

__all__ = ["read_aims"]


def _aims_symbol_to_number(symbol: str, fileobj: IO[Any]) -> int:
    number = symbol_to_number(symbol)
    if number is None:
        raise FormatErrorAIMS(
            f"Unknown element symbol '{symbol}' in '{fileobj}'."
        )
    return number


def read_aims_fileobj(
    fileobj: IO[Any],
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    dtype_int: torch.dtype = torch.long,
    **kwargs: Any,
) -> Structure:
    """
    Reads an FHI-aims ``geometry.in`` file into a structure, with lattice
    vectors and a periodicity mask if the file declares one or more
    ``lattice_vector`` lines.

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
        file declares a ``lattice_vector``, also the lattice (rows are
        lattice vectors, bohr) and the periodicity mask.

    Raises
    ------
    FormatErrorAIMS
        The file does not conform with the expected ``geometry.in`` format.
    """
    dd, ddi = resolve_dd(device, dtype, dtype_int)

    numbers_list: list[int] = []
    cart_rows: list[list[float]] = []
    frac_rows: list[list[float]] = []
    lattice_rows: list[list[float]] = []

    for line in fileobj:
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        tokens = line.split()
        keyword = tokens[0]

        if keyword in ("atom", "atom_frac"):
            try:
                x, y, z = (float(v) for v in tokens[1:4])
                symbol = tokens[4]
            except (IndexError, ValueError) as e:
                raise FormatErrorAIMS(
                    f"Cannot read coordinates from '{fileobj}': {line!r}."
                ) from e

            numbers_list.append(_aims_symbol_to_number(symbol, fileobj))
            if keyword == "atom_frac":
                frac_rows.append([x, y, z])
                cart_rows.append([0.0, 0.0, 0.0])
            else:
                cart_rows.append([x, y, z])
                frac_rows.append([0.0, 0.0, 0.0])

        elif keyword == "lattice_vector":
            if len(lattice_rows) >= 3:
                raise FormatErrorAIMS(
                    f"Too many lattice vectors in '{fileobj}': a fourth "
                    "'lattice_vector' line was found."
                )
            try:
                lattice_rows.append([float(v) for v in tokens[1:4]])
            except (IndexError, ValueError) as e:
                raise FormatErrorAIMS(
                    f"Cannot read lattice vector from '{fileobj}': {line!r}."
                ) from e

        else:
            raise FormatErrorAIMS(
                f"Unexpected keyword {keyword!r} in '{fileobj}'."
            )

    if not numbers_list:
        raise FormatErrorAIMS(f"No atoms found in '{fileobj}'.")

    numbers = torch.tensor(numbers_list, **ddi)
    cart = torch.tensor(cart_rows, **dd) * length.AA2AU
    frac = torch.tensor(frac_rows, **dd)

    periodic_dims = len(lattice_rows)
    if periodic_dims == 0:
        positions = finalize_geometry(numbers, cart, fileobj, **kwargs)
        return Structure(numbers=numbers, positions=positions)

    # A non-periodic axis keeps a 1 bohr unit vector as a placeholder, as in
    # the Turbomole reader: mctc-lib leaves it zero, but that singular cell
    # cannot be inverted to fold atoms into the central cell.
    lattice = torch.eye(3, **dd)
    lattice[:periodic_dims] = torch.tensor(lattice_rows, **dd) * length.AA2AU

    # mctc-lib's own simplification: the fractional -> cartesian transform
    # only ever uses the top-left (periodic_dims, periodic_dims) block of
    # the lattice, even though each lattice_vector line carries a full
    # 3-component vector (relevant for a 1D/2D-periodic system whose axis
    # is not aligned with x/y/z); any fractional component beyond
    # periodic_dims is instead treated as a literal cartesian value.
    positions = cart.clone()
    positions[:, :periodic_dims] += (
        frac[:, :periodic_dims] @ lattice[:periodic_dims, :periodic_dims]
    )
    positions[:, periodic_dims:] += frac[:, periodic_dims:] * length.AA2AU

    periodic = torch.zeros(3, dtype=torch.bool, device=device)
    periodic[:periodic_dims] = True

    positions = finalize_geometry(numbers, positions, fileobj, **kwargs)

    return Structure(
        numbers=numbers, positions=positions, lattice=lattice, periodic=periodic
    )


read_aims = create_path_reader(read_aims_fileobj)
