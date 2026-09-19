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
Test the general JSON sniff-and-dispatch reader, mirroring mctc-lib's
``mctc_io_read_json``: a generic ``.json`` file is routed to the qcschema,
pymatgen, or cjson reader based on which schema-identifying keys it
contains, without re-parsing the file twice.
"""

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest
import torch

from tad_mctc.io import read


def _write(
    data: dict[str, Any],
) -> tuple[tempfile.TemporaryDirectory[str], Path]:
    tmpdir = tempfile.TemporaryDirectory()
    filepath = Path(tmpdir.name) / "mol.json"
    filepath.write_text(json.dumps(data), encoding="utf-8")
    return tmpdir, filepath


def test_sniff_qcschema_by_schema_name() -> None:
    data = {
        "schema_name": "qcschema_molecule",
        "schema_version": 1,
        "molecule": {
            "symbols": ["O", "H", "H"],
            "geometry": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        },
    }
    tmpdir, filepath = _write(data)
    with tmpdir:
        result = read.read_json(filepath)

    assert len(result) == 2
    numbers, positions = result
    assert (numbers == torch.tensor([8, 1, 1])).all()
    assert positions.shape == (3, 3)


def test_sniff_qcschema_default_fallback() -> None:
    """A file with none of the identifying keys falls back to qcschema,
    matching mctc-lib's own default -- this is the shape of the existing
    ``files/mol.json`` fixture used by ``test_reader.py``'s ``test_types``.
    ``schema_version`` is explicit here (mctc-lib defaults an absent one to
    2, the flat/no-wrapper layout) so this fixture's ``molecule``-wrapped
    shape stays schema_version-1-correct rather than relying on a default
    that would actually pick the other layout."""
    data = {
        "schema_version": 1,
        "molecule": {
            "symbols": ["O", "H", "H"],
            "geometry": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        },
    }
    tmpdir, filepath = _write(data)
    with tmpdir:
        result = read.read_json(filepath)

    assert len(result) == 2


def test_sniff_pymatgen_by_module_and_class() -> None:
    data = {
        "@module": "pymatgen.core.structure",
        "@class": "Molecule",
        "sites": [
            {"label": "O", "xyz": [0.0, 0.0, 0.0]},
            {"label": "H", "xyz": [1.0, 0.0, 0.0]},
        ],
    }
    tmpdir, filepath = _write(data)
    with tmpdir:
        result = read.read_json(filepath)

    assert len(result) == 2
    numbers, positions = result
    assert (numbers == torch.tensor([8, 1])).all()
    assert positions.shape == (2, 3)


def test_sniff_cjson_by_chemical_json_key() -> None:
    data = {
        "chemicalJson": 1,
        "atoms": {
            "elements": {"number": [8, 1]},
            "coords": {"3d": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]},
        },
    }
    tmpdir, filepath = _write(data)
    with tmpdir:
        result = read.read_json(filepath)

    assert len(result) == 6
    numbers, positions, lattice, periodic, bonds, bond_orders = result
    assert (numbers == torch.tensor([8, 1])).all()
    assert lattice is None
    assert periodic is None
    assert bonds is None
    assert bond_orders is None


def test_sniff_qcschema_precedence_over_pymatgen() -> None:
    """mctc-lib's own dispatch (json.F90) checks ``schema_name``/
    ``schema_version`` *before* ``@module``/``@class``, so an object that
    (pathologically) carries identifying keys for both schemas must still
    be routed to qcschema."""
    data = {
        "schema_version": 1,
        "@module": "pymatgen.core.structure",
        "@class": "Molecule",
        "molecule": {
            "symbols": ["O", "H", "H"],
            "geometry": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        },
    }
    tmpdir, filepath = _write(data)
    with tmpdir:
        result = read.read_json(filepath)

    assert len(result) == 2
    numbers, positions = result
    assert (numbers == torch.tensor([8, 1, 1])).all()
    assert positions.shape == (3, 3)


def test_sniff_qcschema_by_schema_version_only() -> None:
    """Mirrors mctc-lib's ``test_valid_mol3`` fixture, which carries
    ``schema_version`` without ``schema_name``."""
    data = {
        "schema_version": 1,
        "molecule": {
            "geometry": [
                0.0,
                0.0,
                -0.1294,
                0.0,
                -1.4941,
                1.0274,
                0.0,
                1.4941,
                1.0274,
            ],
            "symbols": ["O", "H", "H"],
            "comment": "Water molecule",
        },
    }
    tmpdir, filepath = _write(data)
    with tmpdir:
        result = read.read_json(filepath)

    assert len(result) == 2
    numbers, _ = result
    assert (numbers == torch.tensor([8, 1, 1])).all()


def test_sniff_cjson_by_spaced_key_alias() -> None:
    data = {
        "chemical json": 0,
        "atoms": {
            "elements": {"number": [6]},
            "coords": {"3d": [0.0, 0.0, 0.0]},
        },
    }
    tmpdir, filepath = _write(data)
    with tmpdir:
        result = read.read_json(filepath)

    assert len(result) == 6


def test_read_fail_notfound() -> None:
    with pytest.raises(FileNotFoundError):
        read.read_json("not found")


def test_read_fail_malformed_json() -> None:
    tmpdir = tempfile.TemporaryDirectory()
    filepath = Path(tmpdir.name) / "mol.json"
    filepath.write_text("{not valid json", encoding="utf-8")
    with tmpdir:
        with pytest.raises(ValueError):
            read.read_json(filepath)
