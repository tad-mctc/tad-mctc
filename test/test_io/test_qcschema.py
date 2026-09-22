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
Test the QCSchema file reader.

The additional tests below port ``mctc-lib``'s
``test/test_read_qcschema.f90`` cases that had no Python equivalent yet
(see git history / PR description for the audit). The reader now follows
mctc-lib's own ``schema_name``/``schema_version`` dispatch (see the
``read_qcschema.qcschema`` module docstring): the ``qcschema_input``
envelope, both the wrapped (``schema_version: 1``) and flat
(``schema_version: 2``) ``qcschema_molecule`` layouts, and periodicity via
``extras.periodic.lattice`` are all supported. ``molecular_charge``,
``molecular_multiplicity`` and ``connectivity`` are read by mctc-lib but
are simply never looked at by the Python reader (this codebase reads
charge/UHF from separate ``.CHRG``/``.UHF`` dotfiles instead, see
``tad_mctc.io.read.dotfiles``), so mctc-lib's malformed-charge/
multiplicity/connectivity/comment/schema-version/schema-name tests have no
Python equivalent to port: the reader never parses those keys, so no
error can occur for them.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest
import torch

from tad_mctc.io import read
from tad_mctc.typing import DD

from ..conftest import DEVICE

sample_list = ["LiH", "H2O"]


def _write(
    data: dict[str, Any],
) -> tuple[tempfile.TemporaryDirectory[str], Path]:
    """Write ``data`` as a JSON file in a fresh temporary directory."""
    tmpdir = tempfile.TemporaryDirectory()
    filepath = Path(tmpdir.name) / "mol.json"
    filepath.write_text(json.dumps(data), encoding="utf-8")
    return tmpdir, filepath


def test_read_fail() -> None:
    with pytest.raises(FileNotFoundError):
        read.read_qcschema("not found")


def test_read_fail_empty() -> None:
    # Create a temporary directory to save the file
    with tempfile.TemporaryDirectory() as tmpdirname:
        filepath = Path(tmpdirname) / "mol.json"
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("")

        with pytest.raises(ValueError):
            read.read_qcschema(filepath)


@pytest.mark.parametrize("file", ["mol1.json", "mol2.json", "mol3.json"])
def test_json(file: str) -> None:
    p = Path(__file__).parent.resolve() / "fail" / file
    with pytest.raises(KeyError):
        read.read_qcschema(p)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_read(dtype: torch.dtype) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    numbers = torch.tensor([8, 1, 1], device=DEVICE)
    positions = torch.tensor(
        [
            [+0.00000000000000, +0.00000000000000, -0.74288549752983],
            [-1.43472674945442, +0.00000000000000, +0.37144274876492],
            [+1.43472674945442, +0.00000000000000, +0.37144274876492],
        ],
        **dd,
    )

    # Create a temporary directory to save the file
    filepath = Path(__file__).parent / "files" / "mol.json"
    with open(filepath, encoding="utf-8") as fp:
        structure = read.qcschema.read_qcschema_fileobj(fp, **dd)
        read_numbers, read_positions = structure.numbers, structure.positions

    # Check if the read data matches the written data
    assert read_numbers.dtype == numbers.dtype
    assert read_numbers.shape == numbers.shape
    assert (numbers == read_numbers).all()
    assert pytest.approx(positions.cpu()) == read_positions.cpu()


def test_valid1_qcschema() -> None:
    """
    Port of mctc-lib's ``test_valid1_qcschema``: ``schema_version: 1`` with
    a nested ``molecule`` object holding ``symbols``/``geometry`` directly
    -- the one shape the Python reader supports natively.
    """
    data = {
        "schema_version": 1,
        "molecule": {
            "geometry": [
                0.0,
                0.0000,
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
        structure = read.read_qcschema(filepath, device=DEVICE)
        numbers, positions = structure.numbers, structure.positions

    assert (numbers == torch.tensor([8, 1, 1], device=DEVICE)).all()
    ref = torch.tensor(
        [
            [0.0, 0.0000, -0.1294],
            [0.0, -1.4941, 1.0274],
            [0.0, 1.4941, 1.0274],
        ],
        device=DEVICE,
    )
    assert pytest.approx(ref.cpu()) == positions.cpu()


def test_valid2_qcschema() -> None:
    """
    Port of mctc-lib's ``test_valid2_qcschema``: a ``qcschema_input``
    (``schema_version: 1``) envelope whose ``molecule`` object is itself a
    ``qcschema_molecule`` (``schema_version: 2``) with ``provenance``,
    ``molecular_charge`` and ``connectivity`` alongside ``symbols``/
    ``geometry``. The Python reader has no schema dispatch, so it just
    reads ``data["molecule"]["symbols"]``/``["geometry"]`` directly and
    silently ignores the other keys -- this happens to work here because
    the inner ``molecule`` object holds ``symbols``/``geometry`` directly
    (unlike ``test_valid4_qcschema`` below, where they sit one level
    deeper).
    """
    data = {
        "schema_version": 1,
        "schema_name": "qcschema_input",
        "driver": "energy",
        "model": {"method": "xtb", "basis": ""},
        "molecule": {
            "schema_version": 2,
            "schema_name": "qcschema_molecule",
            "provenance": {
                "creator": "mctc-lib",
                "version": "0.2.3",
                "routine": "mctc_io_write_qcschema::write_qcschema",
            },
            "symbols": [
                "C",
                "C",
                "C",
                "C",
                "C",
                "C",
                "H",
                "H",
                "H",
                "H",
                "H",
                "H",
                "H",
                "H",
                "H",
                "H",
                "H",
                "C",
                "C",
                "C",
                "C",
                "C",
                "C",
                "H",
                "H",
                "H",
                "H",
                "H",
                "H",
                "H",
                "H",
                "H",
                "H",
                "H",
                "H",
                "H",
                "H",
                "H",
            ],
            "geometry": [
                1.1910941154998063e00,
                8.0445623507578545e-01,
                0.0,
                3.6246828858324265e00,
                -7.2565467293657882e-01,
                0.0,
                6.0068711168320394e00,
                8.8306882464391478e-01,
                0.0,
                8.4393260517381972e00,
                -6.4779797365275837e-01,
                0.0,
                1.0824537843875046e01,
                9.5827990793265394e-01,
                0.0,
                1.3237906549102401e01,
                -5.9564154403544178e-01,
                0.0,
                1.4916738870552548e01,
                5.9753126974621407e-01,
                0.0,
                1.3341085572910570e01,
                -1.8126249017728293e00,
                -1.6605019820556557e00,
                1.3341085572910570e01,
                -1.8126249017728293e00,
                1.6605019820556557e00,
                1.0802428053059009e01,
                2.2054988770423987e00,
                -1.6484077375067128e00,
                1.0802428053059009e01,
                2.2054988770423987e00,
                1.6484077375067128e00,
                8.4618137876963875e00,
                -1.8965287233311212e00,
                1.6491636277910218e00,
                8.4618137876963875e00,
                -1.8965287233311212e00,
                -1.6491636277910218e00,
                5.9874069420110843e00,
                2.1312326566090456e00,
                -1.6493526003620989e00,
                5.9874069420110843e00,
                2.1312326566090456e00,
                1.6493526003620989e00,
                3.6452808960798451e00,
                -1.9740074774727869e00,
                1.6491636277910218e00,
                3.6452808960798451e00,
                -1.9740074774727869e00,
                -1.6491636277910218e00,
                -1.1910941154998063e00,
                -8.0445623507578545e-01,
                0.0,
                -3.6246828858324265e00,
                7.2565467293657882e-01,
                0.0,
                -6.0068711168320394e00,
                -8.8306882464391478e-01,
                0.0,
                -8.4393260517381972e00,
                6.4779797365275837e-01,
                0.0,
                -1.0824537843875046e01,
                -9.5827990793265394e-01,
                0.0,
                -1.3237906549102401e01,
                5.9564154403544178e-01,
                0.0,
                -1.4916738870552548e01,
                -5.9753126974621407e-01,
                0.0,
                -1.3341085572910570e01,
                1.8126249017728293e00,
                1.6605019820556557e00,
                -1.3341085572910570e01,
                1.8126249017728293e00,
                -1.6605019820556557e00,
                -1.0802428053059009e01,
                -2.2054988770423987e00,
                -1.6484077375067128e00,
                -1.0802428053059009e01,
                -2.2054988770423987e00,
                1.6484077375067128e00,
                -8.4618137876963875e00,
                1.8965287233311212e00,
                1.6491636277910218e00,
                -8.4618137876963875e00,
                1.8965287233311212e00,
                -1.6491636277910218e00,
                -5.9874069420110843e00,
                -2.1312326566090456e00,
                -1.6493526003620989e00,
                -5.9874069420110843e00,
                -2.1312326566090456e00,
                1.6493526003620989e00,
                -3.6452808960798451e00,
                1.9740074774727869e00,
                -1.6491636277910218e00,
                -3.6452808960798451e00,
                1.9740074774727869e00,
                1.6491636277910218e00,
                -1.1706850778234652e00,
                -2.0526200670409165e00,
                1.6491636277910218e00,
                -1.1706850778234652e00,
                -2.0526200670409165e00,
                -1.6491636277910218e00,
                1.1706850778234652e00,
                2.0526200670409165e00,
                -1.6491636277910218e00,
                1.1706850778234652e00,
                2.0526200670409165e00,
                1.6491636277910218e00,
            ],
            "molecular_charge": 0,
            "connectivity": [
                [0, 1, 1],
                [1, 2, 1],
                [2, 3, 1],
                [3, 4, 1],
                [4, 5, 1],
                [5, 6, 1],
                [5, 7, 1],
                [5, 8, 1],
                [4, 9, 1],
                [4, 10, 1],
                [3, 11, 1],
                [3, 12, 1],
                [2, 13, 1],
                [2, 14, 1],
                [1, 15, 1],
                [1, 16, 1],
                [0, 17, 1],
                [17, 18, 1],
                [18, 19, 1],
                [19, 20, 1],
                [20, 21, 1],
                [21, 22, 1],
                [22, 23, 1],
                [22, 24, 1],
                [22, 25, 1],
                [21, 26, 1],
                [21, 27, 1],
                [20, 28, 1],
                [20, 29, 1],
                [19, 30, 1],
                [19, 31, 1],
                [18, 32, 1],
                [18, 33, 1],
                [17, 34, 1],
                [17, 35, 1],
                [0, 36, 1],
                [0, 37, 1],
            ],
        },
    }
    tmpdir, filepath = _write(data)
    with tmpdir:
        structure = read.read_qcschema(filepath, device=DEVICE)
        numbers, positions = structure.numbers, structure.positions

    # mirrors mctc-lib's assertions on `struc%nat` and `struc%nid`
    assert numbers.shape == (38,)
    assert positions.shape == (38, 3)
    assert numbers.unique().numel() == 2


def test_valid3_qcschema() -> None:
    """
    Port of mctc-lib's ``test_valid3_qcschema``: ``schema_version: 2``
    places ``symbols``/``geometry`` directly at the JSON root, with no
    top-level ``molecule`` wrapper -- the flat layout e.g. QCElemental
    emits by default.
    """
    data = {
        "schema_version": 2,
        "geometry": [
            0.0,
            0.0000,
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
    }
    tmpdir, filepath = _write(data)
    with tmpdir:
        structure = read.read_qcschema(filepath, device=DEVICE)
        numbers, positions = structure.numbers, structure.positions

    assert (numbers == torch.tensor([8, 1, 1], device=DEVICE)).all()
    assert positions.shape == (3, 3)


def test_valid4_qcschema() -> None:
    """
    Port of mctc-lib's ``test_valid4_qcschema``: doubly-nested
    ``qcschema_input`` -> ``qcschema_molecule`` (``schema_version: 1``) ->
    ``molecule`` layout, one level deeper than ``test_valid2_qcschema``.
    """
    data = {
        "schema_version": 1,
        "schema_name": "qcschema_input",
        "driver": "gradient",
        "model": {"method": "r2scan", "basis": "def2-svp"},
        "molecule": {
            "schema_version": 1,
            "schema_name": "qcschema_molecule",
            "molecule": {
                "symbols": ["O", "H", "H"],
                "geometry": [
                    0.0,
                    0.0000,
                    -0.1294,
                    0.0,
                    -1.4941,
                    1.0274,
                    0.0,
                    1.4941,
                    1.0274,
                ],
                "molecular_charge": 0,
            },
        },
    }
    tmpdir, filepath = _write(data)
    with tmpdir:
        structure = read.read_qcschema(filepath, device=DEVICE)
        numbers, positions = structure.numbers, structure.positions

    assert (numbers == torch.tensor([8, 1, 1], device=DEVICE)).all()
    assert positions.shape == (3, 3)


def test_extras1_qcschema() -> None:
    """
    Port of mctc-lib's ``test_extras1_qcschema``: a flat (``schema_version:
    2``) TiO2 fixture carrying a periodic lattice via
    ``extras.periodic.lattice`` (a flat 9-element array). Since
    mctc-lib's own writer flattens its column-major ``lattice(:, i)`` (i-th
    lattice vector as column ``i``) via Fortran's column-major array
    constructor, each contiguous triple in the flat JSON array is already
    one full lattice vector -- exactly this project's own "rows are
    lattice vectors" convention, so a plain ``reshape(3, 3)`` needs no
    transpose.
    """
    data = {
        "schema_version": 2,
        "schema_name": "qcschema_molecule",
        "provenance": {
            "creator": "mctc-lib",
            "version": "0.4.2",
            "routine": "mctc_io_write_qcschema::write_qcschema",
        },
        "comment": "TiO2 rutile",
        "symbols": ["Ti", "Ti", "O", "O", "O", "O"],
        "atomic_numbers": [22, 22, 8, 8, 8, 8],
        "geometry": [
            0.0000000000000000e00,
            0.0000000000000000e00,
            0.0000000000000000e00,
            5.2818191416515159e00,
            8.2022538117381334e00,
            8.2022538117381334e00,
            6.1333938828927657e-16,
            5.0082961774473045e00,
            5.0082961774473045e00,
            1.3956333869785798e-15,
            1.1396211446028962e01,
            1.1396211446028962e01,
            5.2818191416515150e00,
            3.1939576342908298e00,
            1.3210549989185438e01,
            5.2818191416515150e00,
            1.3210549989185438e01,
            3.1939576342908289e00,
        ],
        "molecular_charge": 0,
        "extras": {
            "periodic": {
                "lattice": [
                    5.5900366437622173e00,
                    0.0000000000000000e00,
                    0.0000000000000000e00,
                    5.3155130499965102e-16,
                    8.6808915904526547e00,
                    0.0000000000000000e00,
                    5.3155130499965102e-16,
                    5.3155130499965102e-16,
                    8.6808915904526547e00,
                ],
            },
        },
    }
    tmpdir, filepath = _write(data)
    with tmpdir:
        structure = read.read_qcschema(filepath, device=DEVICE)
        numbers, positions = structure.numbers, structure.positions
        lattice, periodic = structure.lattice, structure.periodic
        assert lattice is not None and periodic is not None

    assert numbers.shape == (6,)
    assert positions.shape == (6, 3)
    assert numbers.unique().numel() == 2

    ref_lattice = torch.tensor(
        [
            [5.5900366437622173e00, 0.0, 0.0],
            [5.3155130499965102e-16, 8.6808915904526547e00, 0.0],
            [
                5.3155130499965102e-16,
                5.3155130499965102e-16,
                8.6808915904526547e00,
            ],
        ],
        device=DEVICE,
    )
    assert lattice.shape == (3, 3)
    assert pytest.approx(ref_lattice.cpu()) == lattice.cpu()
    assert periodic.dtype == torch.bool
    assert periodic.all()


def test_extras_periodic_without_lattice_qcschema() -> None:
    """``extras.periodic`` present but without a ``lattice`` key is not an
    error -- it just carries no periodic information, same as ``extras``
    being absent entirely."""
    data = {
        "schema_version": 2,
        "schema_name": "qcschema_molecule",
        "symbols": ["O", "H", "H"],
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
        "extras": {"periodic": {}},
    }
    tmpdir, filepath = _write(data)
    with tmpdir:
        result = read.read_qcschema(filepath)

    assert result.lattice is None


def test_invalid_lattice_length_qcschema() -> None:
    """``extras.periodic.lattice`` must have exactly 9 elements."""
    data = {
        "schema_version": 2,
        "schema_name": "qcschema_molecule",
        "symbols": ["O", "H", "H"],
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
        "extras": {"periodic": {"lattice": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]}},
    }
    tmpdir, filepath = _write(data)
    with tmpdir:
        with pytest.raises(ValueError):
            read.read_qcschema(filepath)


def test_invalid_schema_name_qcschema() -> None:
    """A ``schema_name`` that is neither ``qcschema_molecule`` nor
    ``qcschema_input`` is rejected outright."""
    data = {
        "schema_name": "not_a_qcschema",
        "symbols": ["O", "H", "H"],
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
    }
    tmpdir, filepath = _write(data)
    with tmpdir:
        with pytest.raises(KeyError):
            read.read_qcschema(filepath)


def test_invalid_root_schema_version_qcschema() -> None:
    """A ``qcschema_molecule`` document's own ``schema_version`` must be 1
    or 2 (unlike ``test_invalid_root_data_qcschema``, the root here really
    is a JSON object, so this reaches the schema-version check itself)."""
    data = {
        "schema_version": 3,
        "schema_name": "qcschema_molecule",
        "symbols": ["O", "H", "H"],
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
    }
    tmpdir, filepath = _write(data)
    with tmpdir:
        with pytest.raises(KeyError):
            read.read_qcschema(filepath)


def test_qcschema_input_invalid_schema_version_qcschema() -> None:
    """A ``qcschema_input`` document's own ``schema_version`` must be 1."""
    data = {
        "schema_version": 2,
        "schema_name": "qcschema_input",
        "molecule": {
            "schema_name": "qcschema_molecule",
            "symbols": ["O", "H", "H"],
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
        },
    }
    tmpdir, filepath = _write(data)
    with tmpdir:
        with pytest.raises(KeyError):
            read.read_qcschema(filepath)


def test_qcschema_input_missing_molecule_qcschema() -> None:
    """A ``qcschema_input`` document with no ``molecule`` key at all has
    nothing to resolve down to."""
    data = {
        "schema_version": 1,
        "schema_name": "qcschema_input",
        "driver": "gradient",
    }
    tmpdir, filepath = _write(data)
    with tmpdir:
        with pytest.raises(KeyError):
            read.read_qcschema(filepath)


def test_qcschema_input_child_wrong_schema_name_qcschema() -> None:
    """The nested ``molecule`` a ``qcschema_input`` wraps must itself
    declare ``qcschema_molecule`` (or omit ``schema_name``, defaulted to
    it) -- anything else is rejected."""
    data = {
        "schema_version": 1,
        "schema_name": "qcschema_input",
        "molecule": {
            "schema_name": "not_a_qcschema_molecule",
            "symbols": ["O", "H", "H"],
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
        },
    }
    tmpdir, filepath = _write(data)
    with tmpdir:
        with pytest.raises(KeyError):
            read.read_qcschema(filepath)


def test_schema_version_1_missing_molecule_qcschema() -> None:
    """A plain (not ``qcschema_input``-wrapped) ``schema_version: 1``
    document must itself carry a ``molecule`` key."""
    data = {
        "schema_version": 1,
        "schema_name": "qcschema_molecule",
        "symbols": ["O", "H", "H"],
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
    }
    tmpdir, filepath = _write(data)
    with tmpdir:
        with pytest.raises(KeyError):
            read.read_qcschema(filepath)


def test_missing_geometry_qcschema() -> None:
    """``geometry`` is required alongside ``symbols``."""
    data = {
        "schema_version": 2,
        "schema_name": "qcschema_molecule",
        "symbols": ["O", "H", "H"],
    }
    tmpdir, filepath = _write(data)
    with tmpdir:
        with pytest.raises(KeyError):
            read.read_qcschema(filepath)


def test_missing_symbols_qcschema() -> None:
    """
    Port of mctc-lib's ``test_missing_symbols``: only ``atomic_numbers`` is
    given, not ``symbols``. mctc-lib requires ``symbols`` unconditionally
    (no ``atomic_numbers`` fallback), and so does the Python reader.
    """
    data = {
        "schema_version": 1,
        "schema_name": "qcschema_molecule",
        "molecule": {
            "geometry": [
                0.0,
                0.0000,
                -0.1294,
                0.0,
                -1.4941,
                1.0274,
                0.0,
                1.4941,
                1.0274,
            ],
            "atomic_numbers": [8, 1, 1],
        },
    }
    tmpdir, filepath = _write(data)
    with tmpdir:
        with pytest.raises(KeyError):
            read.read_qcschema(filepath)


def test_invalid_symbols_qcschema() -> None:
    """
    Port of mctc-lib's ``test_invalid_symbols``: ``symbols`` given as
    integers instead of element-symbol strings.
    """
    data = {
        "schema_version": 1,
        "schema_name": "qcschema_molecule",
        "molecule": {
            "geometry": [
                0.0,
                0.0000,
                -0.1294,
                0.0,
                -1.4941,
                1.0274,
                0.0,
                1.4941,
                1.0274,
            ],
            "symbols": [8, 1, 1],
            "comment": "Water molecule",
        },
    }
    tmpdir, filepath = _write(data)
    with tmpdir:
        with pytest.raises(KeyError):
            read.read_qcschema(filepath)


def test_invalid_geometry_qcschema() -> None:
    """
    Port of mctc-lib's ``test_invalid_geometry``: ``geometry`` given as a
    list of 3-element lists instead of a flat list of 3*nat floats.
    """
    data = {
        "schema_version": 1,
        "schema_name": "qcschema_molecule",
        "molecule": {
            "geometry": [
                [0.0, 0.0000, -0.1294],
                [0.0, -1.4941, 1.0274],
                [0.0, 1.4941, 1.0274],
            ],
            "symbols": ["O", "H", "H"],
            "comment": "Water molecule",
        },
    }
    tmpdir, filepath = _write(data)
    with tmpdir:
        with pytest.raises(TypeError):
            read.read_qcschema(filepath)


def test_mismatch_geometry_symbols_qcschema() -> None:
    """
    Port of mctc-lib's ``test_mismatch_geometry_symbols``: number of
    ``symbols`` does not match the number of coordinate triples in
    ``geometry``.
    """
    data = {
        "schema_version": 1,
        "schema_name": "qcschema_molecule",
        "molecule": {
            "geometry": [
                0.0,
                0.0000,
                -0.1294,
                0.0,
                -1.4941,
                1.0274,
                0.0,
                1.4941,
                1.0274,
            ],
            "symbols": ["O", "H", "H", "H"],
        },
    }
    tmpdir, filepath = _write(data)
    with tmpdir:
        with pytest.raises(ValueError):
            read.read_qcschema(filepath)


def test_invalid_root_data_qcschema() -> None:
    """
    Port of mctc-lib's ``test_invalid_root_data``: the JSON document's
    root is an array, not an object.
    """
    data = [
        {
            "schema_version": 0,
            "schema_name": "qcschema_molecule",
            "molecule": {
                "geometry": [
                    0.0,
                    0.0000,
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
    ]
    tmpdir = tempfile.TemporaryDirectory()
    filepath = Path(tmpdir.name) / "mol.json"
    filepath.write_text(json.dumps(data), encoding="utf-8")
    with tmpdir:
        with pytest.raises(KeyError):
            read.read_qcschema(filepath)


def test_incomplete_qcschema() -> None:
    """
    Port of mctc-lib's ``test_incomplete``: the JSON text is truncated
    mid-array (syntactically invalid JSON), distinct from
    ``test_read_fail_empty``'s fully-empty file.
    """
    with tempfile.TemporaryDirectory() as tmpdirname:
        filepath = Path(tmpdirname) / "mol.json"
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(
                "{\n"
                '  "schema_version": 1,\n'
                '  "schema_name": "qcschema_molecule",\n'
                '  "molecule": {\n'
                '    "geometry": [\n'
                "      0.0,  0.0000, -0.1294,\n"
                "      0.0, -1.4941,  1.0274,\n"
            )

        with pytest.raises(ValueError):
            read.read_qcschema(filepath)


def test_cjson_qcschema() -> None:
    """
    Port of mctc-lib's ``test_cjson_qcschema``: feeding a Chemical JSON
    (cjson) document directly to the *qcschema* reader (not the generic
    ``.json`` sniff-and-dispatch reader, which would correctly route this
    to the cjson reader instead) must fail, since cjson has no top-level
    ``molecule`` key.
    """
    data = {
        "chemical json": 0,
        "name": "ethane",
        "inchi": "1/C2H6/c1-2/h1-2H3",
        "formula": "C 2 H 6",
        "atoms": {
            "elements": {"number": [1, 6, 1, 1, 6, 1, 1, 1]},
            "coords": {
                "3d": [
                    1.185080,
                    -0.003838,
                    0.987524,
                    0.751621,
                    -0.022441,
                    -0.020839,
                    1.166929,
                    0.833015,
                    -0.569312,
                    1.115519,
                    -0.932892,
                    -0.514525,
                    -0.751587,
                    0.022496,
                    0.020891,
                    -1.166882,
                    -0.833372,
                    0.568699,
                    -1.115691,
                    0.932608,
                    0.515082,
                    -1.184988,
                    0.004424,
                    -0.987522,
                ]
            },
        },
        "bonds": {
            "connections": {
                "index": [0, 1, 1, 2, 1, 3, 1, 4, 4, 5, 4, 6, 4, 7]
            },
            "order": [1, 1, 1, 1, 1, 1, 1],
        },
    }
    tmpdir, filepath = _write(data)
    with tmpdir:
        with pytest.raises(KeyError):
            read.read_qcschema(filepath)
