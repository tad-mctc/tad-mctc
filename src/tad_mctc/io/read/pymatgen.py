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
I/O Read: Pymatgen
==================

Reader for pymatgen's monty-serialized JSON ``Molecule``/``Structure``
objects. Mirrors mctc-lib's ``mctc_io_read_pymatgen``
(``src/mctc/io/read/pymatgen.F90``).

Every site's absolute Cartesian ``xyz`` field is read directly, for a
``Molecule`` and a periodic ``Structure`` alike -- mctc-lib never reads a
site's fractional ``abc`` field, so neither does this. As with Gaussian's
and Q-Chem's embedded charge/multiplicity, the top-level ``charge`` (and,
for a ``Molecule``, ``spin_multiplicity``) are validated (a malformed one
still raises) but not propagated.
"""

from __future__ import annotations

import json
from numbers import Real
from typing import IO, Any

import torch

from ...convert import symbol_to_number
from ...exceptions import FormatErrorPymatgen
from ...typing import DD, Tensor, get_default_dtype
from ...units import length
from ..checks import content_checks, deflatable_check, shape_checks
from .frompath import create_path_reader_periodic

__all__ = ["read_pymatgen"]


def _is_number(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def _read_vec3(value: Any) -> list[float] | None:
    if (
        not isinstance(value, list)
        or len(value) != 3
        or not all(_is_number(v) for v in value)
    ):
        return None
    return [float(v) for v in value]


def read_pymatgen_from_dict(
    data: Any,
    fileobj: IO[Any],
    dd: DD,
    ddi: DD,
    device: torch.device | None = None,
    **kwargs: Any,
) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Builds atomic numbers and positions (plus lattice/periodicity for a
    periodic ``Structure``) from an already-parsed pymatgen JSON object,
    factored out of :func:`read_pymatgen_fileobj` so a sniff-and-dispatch
    caller (:func:`tad_mctc.io.read.json.read_json_fileobj`) can reuse it
    without re-parsing the same JSON text a second time.

    Raises
    ------
    FormatErrorPymatgen
        The parsed JSON does not conform with the expected pymatgen
        schema.
    """
    if not isinstance(data, dict):
        raise FormatErrorPymatgen(
            f"Invalid JSON object in '{fileobj}': expected a JSON object."
        )

    module_name = data.get("@module")
    if not isinstance(module_name, str):
        raise FormatErrorPymatgen(
            f"Could not read module name from '{fileobj}'."
        )
    if module_name != "pymatgen.core.structure":
        raise FormatErrorPymatgen(
            f"Invalid pymatgen object in '{fileobj}': expected "
            f"'pymatgen.core.structure', got {module_name!r}."
        )

    class_name = data.get("@class")
    if not isinstance(class_name, str):
        raise FormatErrorPymatgen(
            f"Could not read class name from '{fileobj}'."
        )
    if class_name not in ("Structure", "Molecule"):
        raise FormatErrorPymatgen(
            f"Invalid pymatgen object in '{fileobj}': expected 'Structure' "
            f"or 'Molecule', got {class_name!r}."
        )
    periodic_flag = class_name == "Structure"

    charge = data.get("charge", 0.0)
    if not _is_number(charge):
        raise FormatErrorPymatgen(f"Could not read charge from '{fileobj}'.")

    if not periodic_flag:
        multiplicity = data.get("spin_multiplicity", 1)
        if not isinstance(multiplicity, int) or isinstance(multiplicity, bool):
            raise FormatErrorPymatgen(
                f"Could not read spin multiplicity from '{fileobj}'."
            )
        if multiplicity < 1:
            raise FormatErrorPymatgen(
                f"Invalid spin multiplicity in '{fileobj}': expected an "
                "integer >= 1."
            )

    sites = data.get("sites")
    if not isinstance(sites, list):
        raise FormatErrorPymatgen(
            f"Could not read sites from '{fileobj}': expected an array."
        )

    numbers_list: list[int] = []
    coords: list[list[float]] = []
    for site in sites:
        if not isinstance(site, dict):
            raise FormatErrorPymatgen(f"Could not read site in '{fileobj}'.")

        label = site.get("label")
        if not isinstance(label, str):
            raise FormatErrorPymatgen(
                f"Could not read site label in '{fileobj}'."
            )
        number = symbol_to_number(label)
        if number is None:
            raise FormatErrorPymatgen(
                f"Unknown element symbol {label!r} in '{fileobj}'."
            )

        xyz = _read_vec3(site.get("xyz"))
        if xyz is None:
            raise FormatErrorPymatgen(
                f"Could not read site coordinates in '{fileobj}': expected "
                "an array of 3 real values."
            )

        numbers_list.append(number)
        coords.append(xyz)

    numbers = torch.tensor(numbers_list, **ddi)
    positions = torch.tensor(coords, **dd) * length.AA2AU

    if not periodic_flag:
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

    lattice_obj = data.get("lattice")
    if not isinstance(lattice_obj, dict):
        raise FormatErrorPymatgen(
            f"Could not read lattice from '{fileobj}': expected an object."
        )
    matrix = lattice_obj.get("matrix")
    if not isinstance(matrix, list) or len(matrix) != 3:
        raise FormatErrorPymatgen(
            f"Invalid lattice matrix size in '{fileobj}': expected 3 "
            "lattice vectors."
        )
    lattice_rows = []
    for row in matrix:
        vec = _read_vec3(row)
        if vec is None:
            raise FormatErrorPymatgen(
                f"Could not read lattice vector in '{fileobj}': expected "
                "an array of 3 real values."
            )
        lattice_rows.append(vec)
    lattice = torch.tensor(lattice_rows, **dd) * length.AA2AU

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


def read_pymatgen_fileobj(
    fileobj: IO[Any],
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    dtype_int: torch.dtype = torch.long,
    **kwargs: Any,
) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Reads a pymatgen JSON ``Molecule``/``Structure`` and returns atomic
    numbers and positions as tensors, plus lattice vectors and a
    periodicity mask for a periodic ``Structure``.

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
        atomic units). A ``Structure`` additionally carries a lattice
        tensor (shape ``(3, 3)``, rows are lattice vectors in bohr) and an
        all-``True`` periodicity mask (shape ``(3,)``).

    Raises
    ------
    FormatErrorPymatgen
        The file is not valid JSON, or is valid JSON that does not
        conform with the expected pymatgen schema.
    """
    dd: DD = {
        "device": device,
        "dtype": dtype if dtype is not None else get_default_dtype(),
    }
    ddi: DD = {"device": device, "dtype": dtype_int}

    try:
        data = json.load(fileobj)
    except json.JSONDecodeError as e:
        raise FormatErrorPymatgen(f"Invalid JSON in '{fileobj}': {e}") from e

    return read_pymatgen_from_dict(
        data, fileobj, dd, ddi, device=device, **kwargs
    )


read_pymatgen = create_path_reader_periodic(read_pymatgen_fileobj)
