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
Test the Pymatgen JSON (Structure/Molecule) file reader, mirroring
mctc-lib's ``mctc_io_read_pymatgen`` test cases (fixture content taken
directly from mctc-lib's own ``test_read_pymatgen.f90``).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest
import torch

from tad_mctc.exceptions import FormatErrorPymatgen
from tad_mctc.io import read
from tad_mctc.typing import DD
from tad_mctc.units import length

_VALID_MOL1 = {
    "@module": "pymatgen.core.structure",
    "@class": "Molecule",
    "charge": 0,
    "spin_multiplicity": 1,
    "sites": [
        {
            "name": "O",
            "species": [{"element": "O", "occu": 1}],
            "xyz": [1.1847029, 1.1150792, -0.0344641],
            "properties": {},
            "label": "O",
        },
        {
            "name": "H",
            "species": [{"element": "H", "occu": 1}],
            "xyz": [0.4939088, 0.9563767, 0.6340089],
            "properties": {},
            "label": "H",
        },
        {
            "name": "H",
            "species": [{"element": "H", "occu": 1}],
            "xyz": [2.0242676, 1.0811246, 0.4301417],
            "properties": {},
            "label": "H",
        },
        {
            "name": "O",
            "species": [{"element": "O", "occu": 1}],
            "xyz": [-1.1469443, 0.0697649, 1.1470196],
            "properties": {},
            "label": "O",
        },
        {
            "name": "H",
            "species": [{"element": "H", "occu": 1}],
            "xyz": [-1.2798308, -0.5232169, 1.8902833],
            "properties": {},
            "label": "H",
        },
        {
            "name": "H",
            "species": [{"element": "H", "occu": 1}],
            "xyz": [-1.0641398, -0.4956693, 0.356925],
            "properties": {},
            "label": "H",
        },
        {
            "name": "O",
            "species": [{"element": "O", "occu": 1}],
            "xyz": [-0.1633508, -1.0289346, -1.2401808],
            "properties": {},
            "label": "O",
        },
        {
            "name": "H",
            "species": [{"element": "H", "occu": 1}],
            "xyz": [0.4914771, -0.3248733, -1.0784838],
            "properties": {},
            "label": "H",
        },
        {
            "name": "H",
            "species": [{"element": "H", "occu": 1}],
            "xyz": [-0.5400907, -0.8496512, -2.1052499],
            "properties": {},
            "label": "H",
        },
    ],
    "properties": {},
}

_VALID_SOL1 = {
    "@module": "pymatgen.core.structure",
    "@class": "Structure",
    "charge": 0.0,
    "lattice": {
        "matrix": [
            [5.59003664376222, 0.0, 0.0],
            [0.0, 8.68089159045265, 0.0],
            [0.0, 0.0, 8.68089159045265],
        ],
        "pbc": [True, True, True],
        "a": 5.59003664376222,
        "b": 8.68089159045265,
        "c": 8.68089159045265,
        "alpha": 90.0,
        "beta": 90.0,
        "gamma": 90.0,
        "volume": 421.253303917213,
    },
    "properties": {},
    "sites": [
        {
            "species": [{"element": "Ti", "occu": 1}],
            "abc": [0.0, 0.0, 0.0],
            "properties": {},
            "label": "Ti",
            "xyz": [0.0, 0.0, 0.0],
        },
        {
            "species": [{"element": "Ti", "occu": 1}],
            "abc": [0.5, 0.5, 0.5],
            "properties": {},
            "label": "Ti",
            "xyz": [2.79501832188111, 4.340445795226331, 4.340445795226331],
        },
        {
            "species": [{"element": "O", "occu": 1}],
            "abc": [0.0, 0.3053, 0.3053],
            "properties": {},
            "label": "O",
            "xyz": [0.0, 2.6502762025652005, 2.6502762025652005],
        },
        {
            "species": [{"element": "O", "occu": 1}],
            "abc": [0.0, 0.6947, 0.6947],
            "properties": {},
            "label": "O",
            "xyz": [0.0, 6.03061538788746, 6.03061538788746],
        },
        {
            "species": [{"element": "O", "occu": 1}],
            "abc": [0.5, 0.1947, 0.8053],
            "properties": {},
            "label": "O",
            "xyz": [2.79501832188111, 1.69016959266113, 6.99072199779152],
        },
        {
            "species": [{"element": "O", "occu": 1}],
            "abc": [0.5, 0.8053, 0.1947],
            "properties": {},
            "label": "O",
            "xyz": [2.79501832188111, 6.99072199779152, 1.69016959266113],
        },
    ],
}


def _write(data: Any) -> tuple[tempfile.TemporaryDirectory[str], Path]:
    tmpdir = tempfile.TemporaryDirectory()
    filepath = Path(tmpdir.name) / "mol.pmgjson"
    filepath.write_text(json.dumps(data), encoding="utf-8")
    return tmpdir, filepath


def test_read_molecule() -> None:
    """mctc-lib's ``valid-pymatgen-mol1``."""
    dd: DD = {"device": None, "dtype": torch.double}

    tmpdir, filepath = _write(_VALID_MOL1)
    with tmpdir:
        result = read.read_pymatgen(filepath, **dd)

    assert len(result) == 2
    numbers, positions = result

    assert numbers.shape == (9,)
    assert len(torch.unique(numbers)) == 2
    ref_first = torch.tensor([1.1847029, 1.1150792, -0.0344641], **dd) * (
        length.AA2AU
    )
    assert pytest.approx(ref_first.cpu()) == positions[0].cpu()


def test_read_structure() -> None:
    """mctc-lib's ``valid-pymatgen-sol1``: a periodic ``Structure``, using
    the site's absolute ``xyz`` field directly (mctc-lib never reads the
    fractional ``abc`` field, even for a periodic structure)."""
    dd: DD = {"device": None, "dtype": torch.double}

    tmpdir, filepath = _write(_VALID_SOL1)
    with tmpdir:
        numbers, positions, lattice, periodic = read.read_pymatgen(  # type: ignore[misc]
            filepath, **dd
        )

    assert numbers.shape == (6,)
    assert len(torch.unique(numbers)) == 2
    assert (periodic == torch.tensor([True, True, True])).all()

    ref_lattice = (
        torch.eye(3, **dd)
        * torch.tensor(
            [5.59003664376222, 8.68089159045265, 8.68089159045265], **dd
        ).unsqueeze(-1)
        * length.AA2AU
    )
    assert pytest.approx(ref_lattice.cpu()) == lattice.cpu()

    ref_second = (
        torch.tensor(
            [2.79501832188111, 4.340445795226331, 4.340445795226331], **dd
        )
        * length.AA2AU
    )
    assert pytest.approx(ref_second.cpu()) == positions[1].cpu()


def test_read_fail_notfound() -> None:
    with pytest.raises(FileNotFoundError):
        read.read_pymatgen("not found")


def test_read_fail_malformed_json() -> None:
    """Malformed JSON syntax is a distinct error path from valid JSON
    with the wrong schema."""
    tmpdir = tempfile.TemporaryDirectory()
    filepath = Path(tmpdir.name) / "mol.pmgjson"
    filepath.write_text("{not valid json", encoding="utf-8")
    with tmpdir:
        with pytest.raises(FormatErrorPymatgen):
            read.read_pymatgen(filepath)


def _mutated(**overrides: Any) -> dict[str, Any]:
    data = json.loads(json.dumps(_VALID_MOL1))
    for key, value in overrides.items():
        if value is _DELETE:
            data.pop(key)
        else:
            data[key] = value
    return data


_DELETE = object()


@pytest.mark.parametrize(
    "data",
    [
        _mutated(**{"@module": _DELETE}),
        _mutated(**{"@module": "not.pymatgen"}),
        _mutated(**{"@class": _DELETE}),
        _mutated(**{"@class": "NotAClass"}),
        _mutated(charge="not-a-number"),
        _mutated(spin_multiplicity="not-a-number"),
        _mutated(spin_multiplicity=0),
        _mutated(sites="not-a-list"),
        _mutated(sites=[{"label": "O"}]),  # missing xyz
        _mutated(sites=[{"xyz": [0.0, 0.0, 0.0]}]),  # missing label
        _mutated(sites=[{"label": "Xx", "xyz": [0.0, 0.0, 0.0]}]),
        _mutated(sites=[{"label": "O", "xyz": [0.0, 0.0]}]),  # size 2
        # mctc-lib's "incorrect-sites-entry": a site entry that is itself a
        # JSON array (e.g. a bare ["O", x, y, z] tuple) rather than an
        # object.
        _mutated(
            sites=[
                ["O", 1.1847029, 1.1150792, -0.0344641],
                ["H", 0.4939088, 0.9563767, 0.6340089],
                ["H", 2.0242676, 1.0811246, 0.4301417],
            ]
        ),
        # mctc-lib's "incorrect-label": the label field has the wrong JSON
        # type (a number instead of a string).
        _mutated(
            sites=[{"label": 8, "xyz": [1.1847029, 1.1150792, -0.0344641]}]
        ),
        # mctc-lib's "incorrect-xyz": xyz is a string instead of an array.
        _mutated(
            sites=[
                {"label": "O", "xyz": "1.1847029, 1.1150792, -0.0344641"},
            ]
        ),
        # mctc-lib's "incorrect-xyz-value": one xyz component has the wrong
        # JSON type (a string instead of a number).
        _mutated(
            sites=[
                {"label": "O", "xyz": [1.1847029, "1.1150792", -0.0344641]},
            ]
        ),
    ],
    ids=[
        "missing-module",
        "incorrect-module",
        "missing-class",
        "incorrect-class",
        "incorrect-charge",
        "incorrect-multiplicity",
        "unphysical-multiplicity",
        "incorrect-sites",
        "incorrect-sites-entry-no-xyz",
        "incorrect-sites-entry-no-label",
        "unknown-element",
        "incorrect-xyz-size",
        "incorrect-sites-entry",
        "incorrect-label",
        "incorrect-xyz",
        "incorrect-xyz-value",
    ],
)
def test_read_fail_format(data: dict[str, Any]) -> None:
    tmpdir, filepath = _write(data)
    with tmpdir:
        with pytest.raises(FormatErrorPymatgen):
            read.read_pymatgen(filepath)


def test_read_fail_incorrect_root() -> None:
    """mctc-lib's ``incorrect-root``: the JSON document is an array, not
    an object."""
    tmpdir, filepath = _write([_VALID_MOL1])
    with tmpdir:
        with pytest.raises(FormatErrorPymatgen):
            read.read_pymatgen(filepath)


def _lattice_mutated(matrix: Any = None, lattice: Any = None) -> dict[str, Any]:
    data = json.loads(json.dumps(_VALID_SOL1))
    if lattice is not None:
        data["lattice"] = lattice
    elif matrix is not None:
        data["lattice"]["matrix"] = matrix
    return data


@pytest.mark.parametrize(
    "data",
    [
        # mctc-lib's "incorrect-lattice": lattice.matrix has the wrong JSON
        # type (an object instead of an array).
        _lattice_mutated(matrix={}),
        # mctc-lib's "incorrect-lattice-table": lattice itself has the
        # wrong JSON type (an array instead of an object).
        _lattice_mutated(lattice=[]),
        # mctc-lib's "incorrect-lattice-value": a lattice vector component
        # has the wrong JSON type (a string instead of a number).
        _lattice_mutated(
            matrix=[
                ["5.59003664376222", 0.0, 0.0],
                [0.0, 8.68089159045265, 0.0],
                [0.0, 0.0, 8.68089159045265],
            ]
        ),
        # mctc-lib's "incorrect-lattice-size": each lattice vector has the
        # wrong length (1 component instead of 3).
        _lattice_mutated(
            matrix=[
                [5.59003664376222],
                [8.68089159045265],
                [8.68089159045265],
            ]
        ),
        # mctc-lib's "incorrect-lattice-dim": the matrix has the wrong
        # number of lattice vectors (2 instead of 3).
        _lattice_mutated(
            matrix=[
                [5.59003664376222, 0.0, 0.0],
                [0.0, 8.68089159045265, 0.0],
            ]
        ),
        # mctc-lib's "incorrect-lattice-rank": the matrix is a flat array
        # of numbers instead of an array of 3-vectors.
        _lattice_mutated(
            matrix=[5.59003664376222, 8.68089159045265, 8.68089159045265]
        ),
    ],
    ids=[
        "incorrect-lattice",
        "incorrect-lattice-table",
        "incorrect-lattice-value",
        "incorrect-lattice-size",
        "incorrect-lattice-dim",
        "incorrect-lattice-rank",
    ],
)
def test_read_fail_lattice_format(data: dict[str, Any]) -> None:
    tmpdir, filepath = _write(data)
    with tmpdir:
        with pytest.raises(FormatErrorPymatgen):
            read.read_pymatgen(filepath)
