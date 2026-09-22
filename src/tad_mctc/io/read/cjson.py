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
I/O Read: Chemical JSON (cjson)
================================

Reader for the Chemical JSON (Avogadro) format. Mirrors mctc-lib's
``mctc_io_read_cjson`` (``src/mctc/io/read/cjson.F90``).

A periodic lattice and bond connectivity are independently optional here:
a cjson file can have neither, either, or both, and the returned structure
leaves whichever it lacks unset.

mctc-lib (as of writing) applies its Angstrom->bohr conversion to the raw
coordinate array unconditionally, before the fractional->cartesian lattice
transform -- so a fractional coordinate gets scaled by that factor a
second time via the lattice, which is already in bohr. Confirmed against
real periodic cjson data (rutile TiO2) to be a genuine bug, not a
deliberate convention: it stretches every periodic bond by a factor of
``AA2AU`` (~1.8897). Fixed here rather than ported as-is -- diverging
from mctc-lib's current output for the fractional-coordinate case -- and
reported upstream; see the reader's tests for the worked-out regression
case.
"""

from __future__ import annotations

import json
import math
from numbers import Real
from typing import IO, Any

import torch

from ...data import pse
from ...exceptions import FormatErrorCJSON
from ...typing import DD
from ...units import length
from ..structure import Structure
from ._cell import cell_to_lattice
from ._finalize import finalize_geometry, resolve_dd
from .frompath import create_path_reader

__all__ = ["read_cjson"]


def _is_number(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _get_alias(data: dict[str, Any], key1: str, key2: str) -> Any:
    """mctc-lib's own ``cjson_get_*`` fallback: try ``key1``, then
    ``key2`` (cjson's schema allows both a camelCase and a spaced key for
    several fields)."""
    if key1 in data:
        return data[key1]
    return data.get(key2)


def read_cjson_from_dict(
    data: Any,
    fileobj: IO[Any],
    dd: DD,
    ddi: DD,
    device: torch.device | None = None,
    **kwargs: Any,
) -> Structure:
    """
    Builds a structure (with lattice/periodicity and bonds/bond_orders if
    present) from an already-parsed cjson JSON object, factored
    out of :func:`read_cjson_fileobj` so a sniff-and-dispatch caller
    (:func:`tad_mctc.io.read.json.read_json_fileobj`) can reuse it
    without re-parsing the same JSON text a second time.

    Raises
    ------
    FormatErrorCJSON
        The parsed JSON does not conform with the expected cjson schema.
    """
    if not isinstance(data, dict):
        raise FormatErrorCJSON(
            f"Invalid JSON object in '{fileobj}': expected a JSON object."
        )

    schema_version = _get_alias(data, "chemicalJson", "chemical json")
    if not _is_int(schema_version):
        raise FormatErrorCJSON(
            f"Could not read schema version from '{fileobj}': expected an "
            "integer value."
        )
    if schema_version not in (0, 1):
        raise FormatErrorCJSON(
            f"Invalid schema version {schema_version!r} in '{fileobj}': "
            "expected 0 or 1."
        )

    unit_cell = _get_alias(data, "unitCell", "unit cell")
    lattice = None
    if unit_cell is not None:
        if not isinstance(unit_cell, dict):
            raise FormatErrorCJSON(
                f"Could not read unit cell from '{fileobj}': expected an "
                "object."
            )
        try:
            cellpar_aa = tuple(
                float(unit_cell[key])
                for key in ("a", "b", "c", "alpha", "beta", "gamma")
            )
        except (KeyError, TypeError, ValueError) as e:
            raise FormatErrorCJSON(
                f"Could not read unit cell parameters from '{fileobj}': "
                "expected float values."
            ) from e
        a, b, c, alpha, beta, gamma = cellpar_aa
        lattice = cell_to_lattice(
            (
                a * length.AA2AU,
                b * length.AA2AU,
                c * length.AA2AU,
                math.radians(alpha),
                math.radians(beta),
                math.radians(gamma),
            ),
            dd,
        )

    atoms = data.get("atoms")
    if not isinstance(atoms, dict):
        raise FormatErrorCJSON(f"Could not read atoms from '{fileobj}'.")

    elements = atoms.get("elements")
    numbers_raw = elements.get("number") if isinstance(elements, dict) else None
    if not isinstance(numbers_raw, list) or not all(
        _is_int(v) for v in numbers_raw
    ):
        raise FormatErrorCJSON(
            f"Could not read atomic numbers from '{fileobj}'."
        )
    if not all(1 <= v <= pse.MAX_ELEMENT for v in numbers_raw):
        raise FormatErrorCJSON(
            f"Invalid atomic number in '{fileobj}': expected a value "
            f"between 1 and {pse.MAX_ELEMENT}."
        )

    coords_obj = atoms.get("coords")
    if not isinstance(coords_obj, dict):
        raise FormatErrorCJSON(
            f"Could not read coordinates from '{fileobj}': expected an "
            "object."
        )

    geo = coords_obj.get("3d")
    cartesian = geo is not None
    if not cartesian and lattice is not None:
        geo = _get_alias(coords_obj, "3dFractional", "3d fractional")
    if not isinstance(geo, list) or not all(_is_number(v) for v in geo):
        raise FormatErrorCJSON(
            f"Could not read coordinates from '{fileobj}': expected an "
            "array of real values."
        )

    if 3 * len(numbers_raw) != len(geo):
        raise FormatErrorCJSON(
            f"Number of coordinates ({len(geo)}) and atomic numbers "
            f"({len(numbers_raw)}) do not match in '{fileobj}'."
        )

    numbers = torch.tensor(numbers_raw, **ddi)
    geo_t = torch.tensor([float(v) for v in geo], **dd).reshape(-1, 3)
    # fractional coordinates are dimensionless -- only the (already bohr)
    # lattice carries a unit, so Angstrom->bohr applies to cartesian
    # coordinates only (see the module docstring)
    if cartesian:
        positions = geo_t * length.AA2AU
    else:
        # fractional coordinates are only read above when a lattice is set
        assert lattice is not None
        positions = geo_t @ lattice

    bonds = None
    bond_orders = None
    bonds_obj = data.get("bonds")
    if bonds_obj is not None:
        if not isinstance(bonds_obj, dict):
            raise FormatErrorCJSON(
                f"Could not read bonds from '{fileobj}': expected an " "object."
            )
        connections = bonds_obj.get("connections")
        index_list = (
            connections.get("index") if isinstance(connections, dict) else None
        )
        if index_list is not None:
            if (
                not isinstance(index_list, list)
                or len(index_list) % 2 != 0
                or not all(_is_int(v) for v in index_list)
            ):
                raise FormatErrorCJSON(
                    f"Could not read bond connectivity from '{fileobj}'."
                )
            nbond = len(index_list) // 2

            order = bonds_obj.get("order")
            if order is None:
                order = [1] * nbond
            if (
                not isinstance(order, list)
                or len(order) != nbond
                or not all(_is_number(v) for v in order)
            ):
                raise FormatErrorCJSON(
                    "Number of bond orders and connectivity indices must "
                    f"match in '{fileobj}'."
                )

            bonds = torch.tensor(index_list, **ddi).reshape(nbond, 2)
            bond_orders = torch.tensor([float(v) for v in order], **dd)

    periodic = (
        torch.ones(3, dtype=torch.bool, device=device)
        if lattice is not None
        else None
    )

    positions = finalize_geometry(numbers, positions, fileobj, **kwargs)

    return Structure(
        numbers=numbers,
        positions=positions,
        lattice=lattice,
        periodic=periodic,
        bonds=bonds,
        bond_orders=bond_orders,
    )


def read_cjson_fileobj(
    fileobj: IO[Any],
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    dtype_int: torch.dtype = torch.long,
    **kwargs: Any,
) -> Structure:
    """
    Reads a Chemical JSON file into a structure, with lattice vectors, a
    periodicity mask, bond indices and bond orders where the file has
    them.

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
        Atomic numbers and positions (bohr), plus lattice and periodicity
        mask, and bonds and bond orders, each only if the file has them.

    Raises
    ------
    FormatErrorCJSON
        The file is not valid JSON, or is valid JSON that does not
        conform with the expected cjson schema.
    """
    dd, ddi = resolve_dd(device, dtype, dtype_int)

    try:
        data = json.load(fileobj)
    except json.JSONDecodeError as e:
        raise FormatErrorCJSON(f"Invalid JSON in '{fileobj}': {e}") from e

    return read_cjson_from_dict(data, fileobj, dd, ddi, device=device, **kwargs)


read_cjson = create_path_reader(read_cjson_fileobj)
