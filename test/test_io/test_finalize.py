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
Test the geometry checks that every structure reader shares, through the
general :func:`tad_mctc.io.read.read_structure` entry point.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tad_mctc.exceptions import StructureError
from tad_mctc.io import read

FILES = Path(__file__).parent.resolve() / "files"

# (file name, ftype) of the fixture files, one per reader that has one
FIXTURES = [
    ("mol.xyz", None),
    ("qm9.xyz", "qm9"),
    ("coord", "tmol"),
    ("mol.ein", None),
    ("mol.json", None),
    ("mol.mol", None),
    ("POSCAR", "vasp"),
]

# Two hydrogen atoms at the same position, one file per format. The shared
# position is away from the origin, so the padding check stays quiet.
FUSED = {
    "xyz": "2\n\nH 1.0 1.0 1.0\nH 1.0 1.0 1.0\n",
    "qm9": "2\n\nH 1.0 1.0 1.0 0.0\nH 1.0 1.0 1.0 0.0\n",
    "tmol": "$coord\n1.0 1.0 1.0 h\n1.0 1.0 1.0 h\n$end\n",
    "aims": "atom 1.0 1.0 1.0 H\natom 1.0 1.0 1.0 H\n",
    "gen": "2 C\nH\n1 1 1.0 1.0 1.0\n2 1 1.0 1.0 1.0\n",
    "qcjson": (
        '{"schema_version": 1, "molecule": {"symbols": ["H", "H"], '
        '"geometry": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]}}'
    ),
}


def _write_fused(tmp_path: Path, ftype: str) -> Path:
    filepath = tmp_path / "fused"
    filepath.write_text(FUSED[ftype], encoding="utf-8")
    return filepath


@pytest.mark.parametrize("name,ftype", FIXTURES)
def test_read_structure_accepts_check_coldfusion(
    name: str, ftype: str | None
) -> None:
    """Every reader accepts the keyword arguments `read_structure` forwards."""
    structure = read.read_structure(
        FILES / name, ftype=ftype, check_coldfusion=True
    )
    assert structure.numbers.shape[-1] == structure.positions.shape[-2]


@pytest.mark.parametrize("ftype", FUSED)
def test_read_structure_fused_atoms(tmp_path: Path, ftype: str) -> None:
    """Every reader honours `check_coldfusion`, and only when asked."""
    filepath = _write_fused(tmp_path, ftype)

    structure = read.read_structure(filepath, ftype=ftype)
    assert structure.numbers.shape == (2,)

    with pytest.raises(StructureError):
        read.read_structure(filepath, ftype=ftype, check_coldfusion=True)


def test_checks_run_without_asserts(tmp_path: Path) -> None:
    """The checks raise instead of asserting, so `python -O` keeps them."""
    filepath = _write_fused(tmp_path, "xyz")
    script = (
        "from tad_mctc.exceptions import StructureError\n"
        "from tad_mctc.io.read import read_structure\n"
        "try:\n"
        f"    read_structure({str(filepath)!r}, 'xyz', check_coldfusion=True)\n"
        "except StructureError:\n"
        "    raise SystemExit(0)\n"
        "raise SystemExit(1)\n"
    )
    result = subprocess.run(
        [sys.executable, "-O", "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
