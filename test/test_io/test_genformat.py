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
Test the DFTB+ genFormat (.gen) file reader, mirroring mctc-lib's
``mctc_io_read_genformat`` test cases (fixture content taken directly
from mctc-lib's own ``test_read_genformat.f90``). Helical (``H``) mode is
not supported (see the reader's module docstring) and is checked
separately as its own rejection case.
"""

import tempfile
from pathlib import Path

import pytest
import torch

from tad_mctc.exceptions import FormatErrorGenFormat
from tad_mctc.io import read
from tad_mctc.typing import DD
from tad_mctc.units import length


def _write(content: str) -> tuple[tempfile.TemporaryDirectory[str], Path]:
    tmpdir = tempfile.TemporaryDirectory()
    filepath = Path(tmpdir.name) / "mol.gen"
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    return tmpdir, filepath


def test_read_cluster() -> None:
    """mctc-lib's ``valid1``: cluster (``C``) mode, non-periodic, with a
    ``#``-comment line interleaved."""
    dd: DD = {"device": None, "dtype": torch.double}

    content = (
        "3 C\n"
        "O H\n"
        "# B3LYP geometry\n"
        "1 1 0.00000 0.00000 0.11974\n"
        "2 2 0.00000 0.76158 -0.47898\n"
        "2 2 0.00000 -0.76158 -0.47898\n"
    )
    tmpdir, filepath = _write(content)
    with tmpdir:
        result = read.read_genformat(filepath, **dd)

    assert len(result) == 2
    numbers, positions = result

    ref_numbers = torch.tensor([8, 1, 1])
    ref_positions = (
        torch.tensor(
            [
                [0.00000, 0.00000, 0.11974],
                [0.00000, 0.76158, -0.47898],
                [0.00000, -0.76158, -0.47898],
            ],
            **dd,
        )
        * length.AA2AU
    )

    assert (ref_numbers == numbers).all()
    assert pytest.approx(ref_positions.cpu()) == positions.cpu()


def test_read_cluster_multi_species() -> None:
    """mctc-lib's ``valid2``: cluster (``C``) mode, 4 species (including a
    2-letter symbol) with a non-monotonic species-index column (species 1
    reappears after species 4), scientific-notation coordinates."""
    dd: DD = {"device": None, "dtype": torch.double}

    content = (
        "9 C\n"
        "C Br H O\n"
        "     1   1  -8.9147060000E-02  -6.6786080000E-02  -1.0432907000E-01\n"
        "     2   2   1.7639746700E+00   2.6771621000E-01   4.2178865000E-01\n"
        "     3   3  -2.6325805000E-01  -1.1300550700E+00  -1.3052621000E-01\n"
        "     4   3  -7.4963702000E-01   3.9302570000E-01   6.1238499000E-01\n"
        "     5   3  -2.6130022000E-01   3.5462634000E-01  -1.0812232600E+00\n"
        "     6   4   4.7684499800E+00   7.6734388000E-01   1.2078966200E+00\n"
        "     7   1   5.5165496700E+00   2.5437564000E-01   4.3331738000E-01\n"
        "     8   3   6.6378745000E+00   3.1585526000E-01   5.3760272000E-01\n"
        "     9   3   5.1708208600E+00  -3.3263252000E-01  -4.6451965000E-01\n"
    )
    tmpdir, filepath = _write(content)
    with tmpdir:
        result = read.read_genformat(filepath, **dd)

    assert len(result) == 2
    numbers, positions = result

    ref_numbers = torch.tensor([6, 35, 1, 1, 1, 8, 6, 1, 1])
    ref_positions = (
        torch.tensor(
            [
                [-8.9147060000e-02, -6.6786080000e-02, -1.0432907000e-01],
                [1.7639746700e00, 2.6771621000e-01, 4.2178865000e-01],
                [-2.6325805000e-01, -1.1300550700e00, -1.3052621000e-01],
                [-7.4963702000e-01, 3.9302570000e-01, 6.1238499000e-01],
                [-2.6130022000e-01, 3.5462634000e-01, -1.0812232600e00],
                [4.7684499800e00, 7.6734388000e-01, 1.2078966200e00],
                [5.5165496700e00, 2.5437564000e-01, 4.3331738000e-01],
                [6.6378745000e00, 3.1585526000e-01, 5.3760272000e-01],
                [5.1708208600e00, -3.3263252000e-01, -4.6451965000e-01],
            ],
            **dd,
        )
        * length.AA2AU
    )

    assert numbers.unique().numel() == 4
    assert (ref_numbers == numbers).all()
    assert pytest.approx(ref_positions.cpu()) == positions.cpu()


def test_read_supercell() -> None:
    """mctc-lib's ``valid3``: supercell (``S``) mode, cartesian
    coordinates, zero origin + 3 lattice vectors, 8-atom diamond-cubic
    carbon cell (single species, exercising a larger same-species atom
    count than the other supercell/fractional fixtures)."""
    dd: DD = {"device": None, "dtype": torch.double}

    content = (
        "8 S\n"
        "C\n"
        "1 1 0.0 0.0 0.0\n"
        "2 1 -0.0 1.78339 1.7834\n"
        "3 1 1.78339 1.7834 0.0\n"
        "4 1 1.78339 -0.0 1.7834\n"
        "5 1 2.67509 0.8917 2.67509\n"
        "6 1 0.8917 0.8917 0.8917\n"
        "7 1 0.8917 2.67509 2.67509\n"
        "8 1 2.67509 2.67509 0.8917\n"
        "0.0 0.0 0.0\n"
        "3.567 0.0 0.0\n"
        "0.0 3.567 0.0\n"
        "0.0 0.0 3.567\n"
    )
    tmpdir, filepath = _write(content)
    with tmpdir:
        numbers, positions, lattice, periodic = read.read_genformat(  # type: ignore[misc]
            filepath, **dd
        )

    ref_lattice = torch.eye(3, **dd) * 3.567 * length.AA2AU
    ref_positions = (
        torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [-0.0, 1.78339, 1.7834],
                [1.78339, 1.7834, 0.0],
                [1.78339, -0.0, 1.7834],
                [2.67509, 0.8917, 2.67509],
                [0.8917, 0.8917, 0.8917],
                [0.8917, 2.67509, 2.67509],
                [2.67509, 2.67509, 0.8917],
            ],
            **dd,
        )
        * length.AA2AU
    )

    assert (numbers == torch.tensor([6] * 8)).all()
    assert (periodic == torch.tensor([True, True, True])).all()
    assert pytest.approx(ref_lattice.cpu()) == lattice.cpu()
    assert pytest.approx(ref_positions.cpu()) == positions.cpu()


def test_read_supercell_nonzero_origin() -> None:
    """Regression test for a former bug (fixed here, reported upstream to
    mctc-lib) where the origin line was subtracted without first being
    converted from Angstrom to bohr, unlike the coordinates and lattice
    around it."""
    dd: DD = {"device": None, "dtype": torch.double}

    content = (
        "1 S\n"
        "C\n"
        "1 1 0.0 0.0 0.0\n"
        "1.0 0.0 0.0\n"
        "3.567 0.0 0.0\n"
        "0.0 3.567 0.0\n"
        "0.0 0.0 3.567\n"
    )
    tmpdir, filepath = _write(content)
    with tmpdir:
        _, positions, _, _ = read.read_genformat(filepath, **dd)  # type: ignore[misc]

    # position (0,0,0)*AA2AU minus the (also AA2AU-converted) origin (1.0, 0, 0)
    ref_positions = torch.tensor([[-length.AA2AU, 0.0, 0.0]], **dd)
    assert pytest.approx(ref_positions.cpu()) == positions.cpu()


def test_read_fractional() -> None:
    """mctc-lib's ``valid4``: fractional (``F``) mode -- coordinates go
    through the lattice, unlike supercell mode."""
    dd: DD = {"device": None, "dtype": torch.double}

    content = (
        "2 F\n"
        "Ga As\n"
        "1 1 0.00 0.00 0.00\n"
        "2 2 0.25 0.25 0.25\n"
        "0.0 0.0 0.0\n"
        "2.713546 2.713546 0.0\n"
        "0.0 2.713546 2.713546\n"
        "2.713546 0.0 2.713546\n"
    )
    tmpdir, filepath = _write(content)
    with tmpdir:
        numbers, positions, lattice, periodic = read.read_genformat(  # type: ignore[misc]
            filepath, **dd
        )

    ref_lattice = (
        torch.tensor(
            [
                [2.713546, 2.713546, 0.0],
                [0.0, 2.713546, 2.713546],
                [2.713546, 0.0, 2.713546],
            ],
            **dd,
        )
        * length.AA2AU
    )
    frac = torch.tensor([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]], **dd)
    ref_positions = frac @ ref_lattice

    assert (numbers == torch.tensor([31, 33])).all()
    assert (periodic == torch.tensor([True, True, True])).all()
    assert pytest.approx(ref_lattice.cpu()) == lattice.cpu()
    assert pytest.approx(ref_positions.cpu()) == positions.cpu()


def test_read_fail_notfound() -> None:
    with pytest.raises(FileNotFoundError):
        read.read_genformat("not found")


def test_read_fail_helical_unsupported() -> None:
    """mctc-lib's ``H`` (helical/screw-axis) mode stores a screw-axis
    descriptor, not a real lattice vector, and is rejected outright."""
    content = (
        "2 H\n"
        "C\n"
        "1 1 0.0 0.0 1.4271041431\n"
        "2 1 0.0 0.0 0.0\n"
        "-2.703556133 -2.906666140 -0.3618948259\n"
        "2.140932670 18.0 10\n"
    )
    tmpdir, filepath = _write(content)
    with tmpdir:
        with pytest.raises(FormatErrorGenFormat):
            read.read_genformat(filepath)


@pytest.mark.parametrize(
    "content",
    [
        # negative atom count
        (
            "-2 F\nGa As\n1 1 0.00 0.00 0.00\n2 2 0.25 0.25 0.25\n"
            "0.0 0.0 0.0\n2.713546 2.713546 0.0\n"
            "0.0 2.713546 2.713546\n2.713546 0.0 2.713546\n"
        ),
        # unknown mode identifier
        (
            "2 X\nGa As\n1 1 0.00 0.00 0.00\n2 2 0.25 0.25 0.25\n"
            "0.0 0.0 0.0\n2.713546 2.713546 0.0\n"
            "0.0 2.713546 2.713546\n2.713546 0.0 2.713546\n"
        ),
        # unknown element symbol (junk prefix eats the real letter after
        # mctc-lib's own 4-character token truncation)
        (
            "2 F\nGa ***As\n1 1 0.00 0.00 0.00\n2 2 0.25 0.25 0.25\n"
            "0.0 0.0 0.0\n2.713546 2.713546 0.0\n"
            "0.0 2.713546 2.713546\n2.713546 0.0 2.713546\n"
        ),
        # no atom coordinate lines at all
        "2 F\nGa As\n",
        # missing origin/lattice information entirely
        "2 F\nGa As\n1 1 0.00 0.00 0.00\n2 2 0.25 0.25 0.25\n",
        # malformed lattice vector values
        (
            "2 F\nGa As\n1 1 0.00 0.00 0.00\n2 2 0.25 0.25 0.25\n"
            "0.0 0.0 0.0\n***** ***** 0.0\n0.0 ***** *****\n***** 0.0 *****\n"
        ),
        # origin line with too few values
        (
            "2 S\nC\n1 1 0.0 0.0 0.0\n2 1 1.5 0.0 0.0\n"
            "0.0 0.0\n2.0 0.0 0.0\n0.0 100.0 0.0\n0.0 0.0 100.0\n"
        ),
    ],
    ids=[
        "negative-atom-count",
        "unknown-mode",
        "unknown-element",
        "no-atoms",
        "missing-lattice-info",
        "malformed-lattice-vector",
        "short-origin-line",
    ],
)
def test_read_fail_format(content: str) -> None:
    tmpdir, filepath = _write(content)
    with tmpdir:
        with pytest.raises(FormatErrorGenFormat):
            read.read_genformat(filepath)
