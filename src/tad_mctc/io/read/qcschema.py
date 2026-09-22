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
I/O Read: JSON
==============

Reader for JSON/QCSchema files. Mirrors mctc-lib's ``mctc_io_read_qcschema``
(``src/mctc/io/read/qcschema.F90``):

- ``schema_name`` defaults to ``"qcschema_molecule"`` and must be either
  that or ``"qcschema_input"``.
- ``"qcschema_input"`` (only ``schema_version: 1`` supported at that level)
  wraps a ``"molecule"`` object, which must itself be a
  ``"qcschema_molecule"`` -- its own ``schema_version``/``schema_name`` are
  read the same way, one level down.
- For a ``"qcschema_molecule"``, ``schema_version`` defaults to 2 and
  selects the layout: version 1 nests ``symbols``/``geometry`` (and
  everything else) under a ``"molecule"`` key; version 2 places them
  directly at that level (the flat layout e.g. QCElemental emits).
- ``extras.periodic.lattice`` (a flat 9-element array), if present, makes
  the structure periodic in all 3 dimensions, since QCSchema has no
  partial-periodicity concept. Without it, the structure has no lattice.
"""

from __future__ import annotations

from typing import IO, Any

import torch

from ...convert import symbol_to_number
from ...typing import DD
from ..structure import Structure
from ._finalize import finalize_geometry, resolve_dd
from .frompath import create_path_reader

__all__ = ["read_qcschema"]


def _resolve_qcschema_molecule(data: Any, fileobj: IO[Any]) -> dict[str, Any]:
    """
    Unwrap ``data`` down to the ``qcschema_molecule``-shaped dict holding
    ``symbols``/``geometry`` (and optionally ``extras``), following
    mctc-lib's ``schema_name``/``schema_version`` dispatch exactly (see the
    module docstring). Raises ``KeyError`` for any invalid/unsupported
    schema, matching this reader's existing (pre-dispatch) convention of
    plain built-in exceptions rather than a dedicated format-error class.
    """
    if not isinstance(data, dict):
        raise KeyError(
            f"Invalid schema: expected a JSON object at the root of "
            f"'{fileobj}'."
        )

    schema_version = data.get("schema_version", 2)
    schema_name = data.get("schema_name", "qcschema_molecule")

    if schema_name not in ("qcschema_molecule", "qcschema_input"):
        raise KeyError(
            f"Invalid schema name {schema_name!r} in '{fileobj}': expected "
            "'qcschema_molecule' or 'qcschema_input'."
        )

    if schema_name == "qcschema_input":
        if schema_version != 1:
            raise KeyError(
                f"Unsupported schema version {schema_version!r} for "
                f"'qcschema_input' in '{fileobj}': expected 1."
            )
        if "molecule" not in data or not isinstance(data["molecule"], dict):
            raise KeyError(
                f"Invalid schema: Key 'molecule' not found in '{fileobj}'."
            )
        child = data["molecule"]
        child_schema_name = child.get("schema_name", "qcschema_molecule")
        if child_schema_name != "qcschema_molecule":
            raise KeyError(
                f"Invalid schema name {child_schema_name!r} in "
                f"'{fileobj}': expected 'qcschema_molecule'."
            )
        root = child
        schema_version = child.get("schema_version", 2)
    else:
        root = data

    if schema_version == 1:
        if "molecule" not in root or not isinstance(root["molecule"], dict):
            raise KeyError(
                f"Invalid schema: Key 'molecule' not found in '{fileobj}'."
            )
        mol: dict[str, Any] = root["molecule"]
    elif schema_version == 2:
        mol = root
    else:
        raise KeyError(
            f"Unsupported schema version {schema_version!r} for "
            f"'qcschema_molecule' in '{fileobj}': expected 1 or 2."
        )

    return mol


def read_qcschema_from_dict(
    data: Any,
    fileobj: IO[Any],
    dd: DD,
    ddi: DD,
    **kwargs: Any,
) -> Structure:
    """
    Builds a structure from an already-parsed QCSchema
    JSON object, factored out of :func:`read_qcschema_fileobj` so a
    sniff-and-dispatch caller (:func:`tad_mctc.io.read.json.read_json_fileobj`)
    can reuse it without re-parsing the same JSON text a second time.
    """
    mol = _resolve_qcschema_molecule(data, fileobj)

    if "symbols" not in mol:
        raise KeyError(
            f"Invalid schema: Key 'symbols' not found in '{fileobj}'."
        )
    if "geometry" not in mol:
        raise KeyError(
            f"Invalid schema: Key 'geometry' not found in '{fileobj}'."
        )

    geo = mol["geometry"]
    coords = []
    for i in range(0, len(geo), 3):
        coords.append([float(geo[i]), float(geo[i + 1]), float(geo[i + 2])])

    numbers_list = []
    for s in mol["symbols"]:
        number = symbol_to_number(s) if isinstance(s, str) else None
        if number is None:
            raise KeyError(f"Unknown element symbol {s!r} in '{fileobj}'.")
        numbers_list.append(number)

    numbers = torch.tensor(numbers_list, **ddi)
    positions = torch.tensor(coords, **dd)

    positions = finalize_geometry(numbers, positions, fileobj, **kwargs)

    lattice = None
    extras = mol.get("extras")
    if isinstance(extras, dict):
        per = extras.get("periodic")
        if isinstance(per, dict) and "lattice" in per:
            lat = per["lattice"]
            if len(lat) != 9:
                raise ValueError(
                    f"Lattice from 'extras.periodic.lattice' in "
                    f"'{fileobj}' must have 9 elements, got {len(lat)}."
                )
            lattice = torch.tensor([float(v) for v in lat], **dd).reshape(3, 3)

    if lattice is None:
        return Structure(numbers=numbers, positions=positions)

    periodic = torch.ones(3, dtype=torch.bool, device=dd["device"])
    return Structure(
        numbers=numbers, positions=positions, lattice=lattice, periodic=periodic
    )


def read_qcschema_fileobj(
    fileobj: IO[Any],
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    dtype_int: torch.dtype = torch.long,
    **kwargs: Any,
) -> Structure:
    """
    Reads a JSON/QCSchema file with a single structure, with a lattice and
    an all-True periodicity mask if the file declares one via
    ``extras.periodic.lattice``.

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
        file declares ``extras.periodic.lattice``, also the lattice (bohr)
        and an all-True periodicity mask.
    """
    dd, ddi = resolve_dd(device, dtype, dtype_int)

    # pylint: disable=import-outside-toplevel
    from json import loads as json_load

    data = json_load(fileobj.read())

    return read_qcschema_from_dict(data, fileobj, dd, ddi, **kwargs)


read_qcschema = create_path_reader(read_qcschema_fileobj)
