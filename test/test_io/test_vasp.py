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
Test the VASP POSCAR/CONTCAR file reader.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from tad_mctc.exceptions import EmptyFileError, FormatErrorVASP, StructureWarning
from tad_mctc.io import read
from tad_mctc.typing import DD
from tad_mctc.units import length

from ..conftest import DEVICE


def _write(content: str) -> tuple[tempfile.TemporaryDirectory[str], Path]:
    tmpdir = tempfile.TemporaryDirectory()
    filepath = Path(tmpdir.name) / "POSCAR"
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    return tmpdir, filepath


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_read_cartesian_vasp5(dtype: torch.dtype) -> None:
    """Two-species VASP5-format POSCAR with Cartesian coordinates."""
    dd: DD = {"device": DEVICE, "dtype": dtype}

    content = (
        "Ti  O \n"
        " 1.0000000000000000\n"
        "     4.59373    0.00000    0.00000\n"
        "     0.00000    4.59373    0.00000\n"
        "     0.00000    0.00000    2.95812\n"
        "   Ti  O \n"
        "   1   1\n"
        "Cartesian\n"
        "  0.000000000  0.000000000  0.000000000\n"
        "  2.296865000  2.296865000  1.479060000\n"
    )
    tmpdir, filepath = _write(content)
    with tmpdir:
        structure = read.read_poscar(filepath, **dd)
        numbers, positions = structure.numbers, structure.positions
        lattice = structure.lattice
        assert lattice is not None

    ref_numbers = torch.tensor([22, 8], device=DEVICE)
    ref_lattice = (
        torch.tensor(
            [
                [4.59373, 0.0, 0.0],
                [0.0, 4.59373, 0.0],
                [0.0, 0.0, 2.95812],
            ],
            **dd,
        )
        * length.AA2AU
    )
    ref_positions = (
        torch.tensor(
            [
                [0.000000000, 0.000000000, 0.000000000],
                [2.296865000, 2.296865000, 1.479060000],
            ],
            **dd,
        )
        * length.AA2AU
    )

    assert (ref_numbers == numbers).all()
    assert pytest.approx(ref_lattice.cpu()) == lattice.cpu()
    assert pytest.approx(ref_positions.cpu()) == positions.cpu()


def test_read_direct_vasp5() -> None:
    """Fractional/Direct coordinates are converted via the lattice vectors."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    content = (
        "cubic diamond\n"
        "  3.7\n"
        "    0.5 0.5 0.0\n"
        "    0.0 0.5 0.5\n"
        "    0.5 0.0 0.5\n"
        "   C\n"
        "   2\n"
        "Direct\n"
        "  0.0 0.0 0.0\n"
        "  0.25 0.25 0.25\n"
    )
    tmpdir, filepath = _write(content)
    with tmpdir:
        structure = read.read_poscar(filepath, **dd)
        numbers, positions = structure.numbers, structure.positions
        lattice = structure.lattice
        assert lattice is not None

    ref_numbers = torch.tensor([6, 6], device=DEVICE)
    ref_lattice = (
        torch.tensor(
            [
                [0.5, 0.5, 0.0],
                [0.0, 0.5, 0.5],
                [0.5, 0.0, 0.5],
            ],
            **dd,
        )
        * 3.7
        * length.AA2AU
    )
    frac = torch.tensor([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]], **dd)
    ref_positions = frac @ ref_lattice

    assert (ref_numbers == numbers).all()
    assert pytest.approx(ref_lattice.cpu()) == lattice.cpu()
    assert pytest.approx(ref_positions.cpu()) == positions.cpu()


def test_read_pre_vasp5_format() -> None:
    """Pre-VASP5 POSCAR: element symbols are given on the comment line."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    content = (
        "Ti  O \n"
        " 1.0000000000000000\n"
        "     4.59373    0.00000    0.00000\n"
        "     0.00000    4.59373    0.00000\n"
        "     0.00000    0.00000    2.95812\n"
        "   1   1\n"
        "Cartesian\n"
        "  0.000000000  0.000000000  0.000000000\n"
        "  2.296865000  2.296865000  1.479060000\n"
    )
    tmpdir, filepath = _write(content)
    with tmpdir:
        structure = read.read_poscar(filepath, **dd)
        numbers = structure.numbers

    ref_numbers = torch.tensor([22, 8], device=DEVICE)
    assert (ref_numbers == numbers).all()


def test_read_selective_dynamics() -> None:
    """A ``Selective dynamics`` line is skipped to reach the coordinate type."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    content = (
        "Anatase\n"
        " 1.0\n"
        "     3.785    0.000    0.000\n"
        "     0.000    3.785    0.000\n"
        "     0.000    0.000    9.514\n"
        "   Ti  O \n"
        "   1   1\n"
        "Selective\n"
        "Cartesian\n"
        "  0.0000000000000000  0.0000000000000000  0.0000000000000000\n"
        "  1.8925000000000000  1.8925000000000000  4.7570000000000000\n"
    )
    tmpdir, filepath = _write(content)
    with tmpdir:
        structure = read.read_poscar(filepath, **dd)
        numbers, positions = structure.numbers, structure.positions

    ref_numbers = torch.tensor([22, 8], device=DEVICE)
    ref_positions = (
        torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.8925, 1.8925, 4.757],
            ],
            **dd,
        )
        * length.AA2AU
    )

    assert (ref_numbers == numbers).all()
    assert pytest.approx(ref_positions.cpu()) == positions.cpu()


def test_read_scaling_factor() -> None:
    """A scaling factor other than 1.0 scales both lattice and coordinates."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    content = (
        "scaled\n"
        "  2.0\n"
        "    1.0 0.0 0.0\n"
        "    0.0 1.0 0.0\n"
        "    0.0 0.0 1.0\n"
        "   C\n"
        "   1\n"
        "Cartesian\n"
        "  0.5 0.5 0.5\n"
    )
    tmpdir, filepath = _write(content)
    with tmpdir:
        structure = read.read_poscar(filepath, **dd)
        positions, lattice = structure.positions, structure.lattice
        assert lattice is not None

    ref_lattice = torch.eye(3, **dd) * 2.0 * length.AA2AU
    ref_positions = torch.tensor([[0.5, 0.5, 0.5]], **dd) * 2.0 * length.AA2AU

    assert pytest.approx(ref_lattice.cpu()) == lattice.cpu()
    assert pytest.approx(ref_positions.cpu()) == positions.cpu()


def test_read_single_char_symbol() -> None:
    """Single-character element symbols (e.g. sulfur, ``S``) parse correctly."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    content = (
        "POSCAR\n"
        "3.0\n"
        "1.0  0.0  0.0\n"
        "0.0  1.0  0.0\n"
        "0.0  0.0  1.0\n"
        "S\n"
        "1\n"
        "direct\n"
        "0.0  0.0  0.0\n"
    )
    tmpdir, filepath = _write(content)
    with tmpdir:
        # The single atom sits at the origin, which collides with the
        # deflate padding check's default padding value.
        with pytest.warns(StructureWarning):
            structure = read.read_poscar(filepath, **dd)
        numbers, positions = structure.numbers, structure.positions

    assert numbers.shape == (1,)
    assert (numbers == torch.tensor([16], device=DEVICE)).all()
    assert positions.shape == (1, 3)


def test_read_symbol_with_trailing_digit() -> None:
    """Pre-VASP5 comment-line symbols suffixed with digits (mctc-lib's
    ``test_valid4_poscar``, e.g. site/isotope labels ``C2``/``F2``) should
    still resolve to their element."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    content = (
        "C2 F2\n"
        "1.0\n"
        "  1.291         2.23608        +0.0000000000\n"
        " -1.291         2.23608        +0.0000000000\n"
        " +0.0000000000  +0.0000000000   5.75\n"
        " 2 2\n"
        "cartesian\n"
        "   0.00000000  0.00000000  1.37627335\n"
        "   0.00000000  2.98144198  1.86702665\n"
        "   0.00000000  0.00000000  0.00394701\n"
        "   0.00000000  2.98144198  3.23935299\n"
    )
    tmpdir, filepath = _write(content)
    with tmpdir:
        structure = read.read_poscar(filepath, **dd)
        numbers, positions = structure.numbers, structure.positions

    assert numbers.shape == (4,)
    assert (numbers == torch.tensor([6, 6, 9, 9], device=DEVICE)).all()
    assert positions.shape == (4, 3)


################################################################################


def test_read_fail_notfound() -> None:
    with pytest.raises(FileNotFoundError):
        read.read_poscar("not found")


def test_read_fail_empty() -> None:
    tmpdir, filepath = _write("")
    with tmpdir:
        with pytest.raises(EmptyFileError):
            read.read_poscar(filepath)


@pytest.mark.parametrize(
    "content",
    [
        # truncated: missing lattice vectors and everything after
        ("Ti  O \n 1.0000000000000000\n     4.59373    0.00000    0.00000\n"),
        # only a single (blank) line: fails before the scaling factor is
        # even reached (mctc-lib's test_invalid1_poscar)
        ("\n"),
        # malformed lattice vector entry
        (
            "Ti  O \n"
            " 1.0\n"
            "     *******    0.00000    0.00000\n"
            "     0.00000    4.59373    0.00000\n"
            "     0.00000    0.00000    2.95812\n"
            "   Ti  O \n"
            "   1   1\n"
            "Cartesian\n"
            "  0.0  0.0  0.0\n"
            "  2.3  2.3  1.5\n"
        ),
        # mismatched number of symbols and counts (VASP5: explicit symbol
        # line)
        (
            "Ti  O \n"
            " 1.0\n"
            "     4.59373    0.00000    0.00000\n"
            "     0.00000    4.59373    0.00000\n"
            "     0.00000    0.00000    2.95812\n"
            "   Ti  O \n"
            "   1   1   1\n"
            "Cartesian\n"
            "  0.0  0.0  0.0\n"
            "  2.3  2.3  1.5\n"
        ),
        # mismatched number of symbols and counts (pre-VASP5: symbols
        # taken from the comment line; mctc-lib's test_invalid4_poscar)
        (
            "Ti  O \n"
            " 1.0000000000000000\n"
            "     4.59373    0.00000    0.00000\n"
            "     0.00000    4.59373    0.00000\n"
            "     0.00000    0.00000    2.95812\n"
            "   2   2   2\n"
            "Cartesian\n"
            "  0.000000000  0.000000000  0.000000000\n"
            "  2.296865000  2.296865000  1.479060000\n"
            "  1.402465769  1.402465769  0.000000000\n"
            "  3.191264231  3.191264231  0.000000000\n"
            "  3.699330769  0.894399231  1.479060000\n"
            "  0.894399231  3.699330769  1.479060000\n"
        ),
        # lattice vector line with too few values (all otherwise valid)
        (
            "Ti  O \n"
            " 1.0\n"
            "     4.59373    0.00000\n"
            "     0.00000    4.59373    0.00000\n"
            "     0.00000    0.00000    2.95812\n"
            "   Ti  O \n"
            "   1   1\n"
            "Cartesian\n"
            "  0.0  0.0  0.0\n"
            "  2.3  2.3  1.5\n"
        ),
        # non-numeric atom counts (same length as the symbol line, so the
        # mismatched-counts check above does not fire first)
        (
            "Ti  O \n"
            " 1.0\n"
            "     4.59373    0.00000    0.00000\n"
            "     0.00000    4.59373    0.00000\n"
            "     0.00000    0.00000    2.95812\n"
            "   Ti  O \n"
            "   a   b\n"
            "Cartesian\n"
            "  0.0  0.0  0.0\n"
            "  2.3  2.3  1.5\n"
        ),
        # unknown element symbol
        (
            "Titan  Oxygen\n"
            " 1.0\n"
            "     4.59373    0.00000    0.00000\n"
            "     0.00000    4.59373    0.00000\n"
            "     0.00000    0.00000    2.95812\n"
            "   1   1\n"
            "Cartesian\n"
            "  0.0  0.0  0.0\n"
            "  2.3  2.3  1.5\n"
        ),
        # non-numeric scaling factor (extra comment line shifts everything)
        (
            "# Rutile\n"
            "Ti  O \n"
            " 1.0\n"
            "     4.59373    0.00000    0.00000\n"
            "     0.00000    4.59373    0.00000\n"
            "     0.00000    0.00000    2.95812\n"
            "   Ti  O \n"
            "   1   1\n"
            "Cartesian\n"
            "  0.0  0.0  0.0\n"
            "  2.3  2.3  1.5\n"
        ),
        # Selective dynamics line immediately followed by coordinate data,
        # with no Cartesian/Direct line in between (mctc-lib's
        # test_invalid2_poscar)
        (
            "Ti  O \n"
            " 1.0000000000000000\n"
            "     4.59373    0.00000    0.00000\n"
            "     0.00000    4.59373    0.00000\n"
            "     0.00000    0.00000    2.95812\n"
            "   2   4\n"
            "Selective\n"
            "  0.000000000  0.000000000  0.000000000\n"
            "  2.296865000  2.296865000  1.479060000\n"
            "  1.402465769  1.402465769  0.000000000\n"
            "  3.191264231  3.191264231  0.000000000\n"
            "  3.699330769  0.894399231  1.479060000\n"
            "  0.894399231  3.699330769  1.479060000\n"
        ),
    ],
    ids=[
        "truncated",
        "blank-line-only",
        "malformed-lattice",
        "lattice-too-few-values",
        "non-numeric-atom-counts",
        "mismatched-counts",
        "mismatched-counts-pre-vasp5",
        "unknown-element",
        "shifted",
        "selective-no-coord-type",
    ],
)
def test_read_fail_format(content: str) -> None:
    tmpdir, filepath = _write(content)
    with tmpdir:
        with pytest.raises(FormatErrorVASP):
            read.read_poscar(filepath)


def test_read_is_periodic_along_all_axes() -> None:
    """A POSCAR always describes a bulk cell, so, as in mctc-lib's
    ``new_structure``, every lattice axis is periodic."""
    p = Path(__file__).parent.resolve() / "files" / "POSCAR"
    structure = read.read_poscar(p)

    assert structure.periodic is not None
    assert structure.periodic.tolist() == [True, True, True]
