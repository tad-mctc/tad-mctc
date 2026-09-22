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
Test the FHI-aims ``geometry.in`` file reader, mirroring mctc-lib's
``mctc_io_read_aims`` test cases.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from tad_mctc.exceptions import FormatErrorAIMS
from tad_mctc.io import read
from tad_mctc.ncoord import cn_d3
from tad_mctc.typing import DD
from tad_mctc.units import length


def _write(content: str) -> tuple[tempfile.TemporaryDirectory[str], Path]:
    tmpdir = tempfile.TemporaryDirectory()
    filepath = Path(tmpdir.name) / "geometry.in"
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    return tmpdir, filepath


################################################################################
# non-periodic
################################################################################


def test_read_cartesian_no_lattice() -> None:
    """mctc-lib's ``valid1``: plain cartesian atoms, no periodicity."""
    dd: DD = {"device": None, "dtype": torch.double}

    content = (
        "atom   -0.0090   -0.0157   -0.0000 C\n"
        "atom   -0.7131    1.2038   -0.0000 C\n"
        "atom   -0.5203   -0.9011   -0.0000 H\n"
    )
    tmpdir, filepath = _write(content)
    with tmpdir:
        result = read.read_aims(filepath, **dd)

    assert result.lattice is None
    numbers, positions = result.numbers, result.positions

    ref_numbers = torch.tensor([6, 6, 1])
    ref_positions = (
        torch.tensor(
            [
                [-0.0090, -0.0157, -0.0000],
                [-0.7131, 1.2038, -0.0000],
                [-0.5203, -0.9011, -0.0000],
            ],
            **dd,
        )
        * length.AA2AU
    )

    assert (ref_numbers == numbers).all()
    assert pytest.approx(ref_positions.cpu()) == positions.cpu()


def test_read_comments_and_blank_lines_ignored() -> None:
    """mctc-lib's ``valid3``: ``#`` comments and blank lines are skipped."""
    content = (
        "# 24\n"
        "\n"
        "atom  1.07317  0.04885 -0.07573  c\n"
        "atom  2.51365  0.01256 -0.07580  n\n"
    )
    tmpdir, filepath = _write(content)
    with tmpdir:
        structure = read.read_aims(filepath)
        numbers = structure.numbers

    assert (numbers == torch.tensor([6, 7])).all()


################################################################################
# periodic (lattice_vector)
################################################################################


def test_read_full_3d_lattice_cartesian() -> None:
    """mctc-lib's ``valid4``: three ``lattice_vector`` lines, cartesian atoms."""
    dd: DD = {"device": None, "dtype": torch.double}

    content = (
        "lattice_vector     4.59373    0.00000    0.00000\n"
        "lattice_vector     0.00000    4.59373    0.00000\n"
        "lattice_vector     0.00000    0.00000    2.95812\n"
        "atom  0.000000000  0.000000000  0.000000000  Ti\n"
        "atom  1.402465769  1.402465769  0.000000000  O\n"
    )
    tmpdir, filepath = _write(content)
    with tmpdir:
        structure = read.read_aims(filepath, **dd)
        numbers, positions = structure.numbers, structure.positions
        lattice, periodic = structure.lattice, structure.periodic
        assert lattice is not None

    ref_numbers = torch.tensor([22, 8])
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
        torch.tensor([[0.0, 0.0, 0.0], [1.402465769, 1.402465769, 0.0]], **dd)
        * length.AA2AU
    )

    assert (ref_numbers == numbers).all()
    assert (periodic == torch.tensor([True, True, True])).all()
    assert pytest.approx(ref_lattice.cpu()) == lattice.cpu()
    assert pytest.approx(ref_positions.cpu()) == positions.cpu()


def test_read_mixed_cartesian_and_fractional() -> None:
    """mctc-lib's ``valid5``: cubic diamond, one ``atom`` + one ``atom_frac``."""
    dd: DD = {"device": None, "dtype": torch.double}

    content = (
        "# cubic diamond\n"
        "atom  0.0 0.0 0.0 C\n"
        "atom_frac  0.25 0.25 0.25 C\n"
        "lattice_vector    1.85 1.85 0.0\n"
        "lattice_vector    0.0  1.85 1.85\n"
        "lattice_vector    1.85 0.0  1.85\n"
    )
    tmpdir, filepath = _write(content)
    with tmpdir:
        structure = read.read_aims(filepath, **dd)
        numbers, positions = structure.numbers, structure.positions
        lattice, periodic = structure.lattice, structure.periodic

    # Hand-computed (not via the lattice-multiplication the implementation
    # itself performs): for cubic diamond, the second atom sits at
    # 0.25 * (a1 + a2 + a3) = 0.25 * (3.70, 3.70, 3.70) Angstrom.
    second_atom_ref = torch.tensor([0.925, 0.925, 0.925], **dd) * length.AA2AU

    assert (numbers == torch.tensor([6, 6])).all()
    assert (periodic == torch.tensor([True, True, True])).all()
    assert pytest.approx(torch.zeros(3, **dd).cpu()) == positions[0].cpu()
    assert pytest.approx(second_atom_ref.cpu()) == positions[1].cpu()


def test_read_fractional_equals_cartesian() -> None:
    """mctc-lib's ``valid6``: an ``atom``-only and an ``atom_frac``-only
    description of the same rock-salt MgO cell must agree exactly -- an
    independent cross-check that doesn't rely on hand-computed numbers.

    Only two ``lattice_vector`` lines are given (2D periodicity), so this
    also exercises the branch where the fractional z-component (beyond
    ``periodic_dims``) is treated as a literal Cartesian value rather than
    being transformed by the lattice -- dropping any of the atoms below
    whose z differs from 0 would let that branch go unexercised."""
    dd: DD = {"device": None, "dtype": torch.double}

    lattice_lines = (
        "lattice_vector  2.97762730792410  -2.97762730792410   0.00000000000000\n"
        "lattice_vector  2.97762730792410   2.97762730792410   0.00000000000000\n"
    )
    cartesian_content = (
        "atom            0.00000000000000   0.00000000000000   0.00000000000000 Mg\n"
        "atom            1.48881365396205  -1.48881365396205   0.00000000000000 O\n"
        "atom            1.48881365396205   1.48881365396205   0.00000000000000 O\n"
        "atom            0.00000000000000   0.00000000000000   2.10550046127949 O\n"
        "atom            2.97762730792410   0.00000000000000   0.00000000000000 Mg\n"
        "atom            1.48881365396205  -1.48881365396205   2.10550046127949 Mg\n"
        "atom            1.48881365396205   1.48881365396205   2.10550046127949 Mg\n"
        "atom            2.97762730792410   0.00000000000000   2.10550046127949 O\n"
        + lattice_lines
    )
    fractional_content = (
        "atom_frac       0.00000000000000   0.00000000000000   0.00000000000000 Mg\n"
        "atom_frac       0.50000000000000   0.00000000000000   0.00000000000000 O\n"
        "atom_frac       0.00000000000000   0.50000000000000   0.00000000000000 O\n"
        "atom_frac       0.00000000000000   0.00000000000000   2.10550046127949 O\n"
        "atom_frac       0.50000000000000   0.50000000000000   0.00000000000000 Mg\n"
        "atom_frac       0.50000000000000   0.00000000000000   2.10550046127949 Mg\n"
        "atom_frac       0.00000000000000   0.50000000000000   2.10550046127949 Mg\n"
        "atom_frac       0.50000000000000   0.50000000000000   2.10550046127949 O\n"
        + lattice_lines
    )

    tmpdir1, filepath1 = _write(cartesian_content)
    tmpdir2, filepath2 = _write(fractional_content)
    with tmpdir1, tmpdir2:
        structure = read.read_aims(filepath1, **dd)
        positions1 = structure.positions
        structure = read.read_aims(filepath2, **dd)
        positions2 = structure.positions

    assert pytest.approx(positions1.cpu()) == positions2.cpu()


def test_read_partial_periodicity() -> None:
    """mctc-lib's ``valid7``: a single ``lattice_vector`` line means only
    one axis is periodic."""
    content = (
        "atom             -1.05835465887935   1.85522662363901   0.00000000000000 B\n"
        "atom             -1.05835465887935   1.57910813351869   1.38575958673374 N\n"
        "lattice_vector    4.23341864610095   0.00000000000000   0.00000000000000\n"
    )
    tmpdir, filepath = _write(content)
    with tmpdir:
        structure = read.read_aims(filepath)
        numbers, positions = structure.numbers, structure.positions
        lattice, periodic = structure.lattice, structure.periodic
        assert lattice is not None

    assert (numbers == torch.tensor([5, 7])).all()
    assert (periodic == torch.tensor([True, False, False])).all()
    assert lattice.shape == (3, 3)


def test_read_partial_periodicity_placeholder_axes() -> None:
    """The non-periodic axes of a wire get 1 bohr unit vectors instead of
    mctc-lib's zero rows, so the cell stays invertible and the periodic CN
    can be evaluated."""
    content = (
        "atom             -1.05835465887935   1.85522662363901   0.00000000000000 B\n"
        "atom             -1.05835465887935   1.57910813351869   1.38575958673374 N\n"
        "lattice_vector    4.23341864610095   0.00000000000000   0.00000000000000\n"
    )
    tmpdir, filepath = _write(content)
    with tmpdir:
        structure = read.read_aims(filepath, dtype=torch.double)
        lattice = structure.lattice
        assert lattice is not None

    assert (lattice[1:] == torch.eye(3, dtype=torch.double)[1:]).all()

    cn = cn_d3(structure)
    assert torch.isfinite(cn).all()


################################################################################
# element-symbol quirks
################################################################################


def test_read_symbol_quirks() -> None:
    """mctc-lib's ``valid2``: isotope-prefixed (``18O``), asterisk-suffixed
    (``C*``), and deuterium (``D``) symbols all resolve like mctc-lib's
    ``to_number`` (first two letters only; ``D``/``T`` map to hydrogen)."""
    content = (
        "atom  1.0 0.0 0.0 C\n"
        "atom  2.0 0.0 0.0 C*\n"
        "atom  3.0 0.0 0.0 18O\n"
        "atom  4.0 0.0 0.0 D\n"
    )
    tmpdir, filepath = _write(content)
    with tmpdir:
        structure = read.read_aims(filepath)
        numbers = structure.numbers

    assert (numbers == torch.tensor([6, 6, 8, 1])).all()


################################################################################
# error handling
################################################################################


def test_read_fail_notfound() -> None:
    with pytest.raises(FileNotFoundError):
        read.read_aims("not found")


@pytest.mark.parametrize(
    "content",
    [
        # unknown element symbol
        "atom 0.0 0.0 0.0 hh\n",
        # unexpected keyword
        "lattice    1.0 0.0 0.0\n",
        # malformed lattice vector value
        (
            "lattice_vector 1.0 0.0 0.0\n"
            "lattice_vector 0.0 abcdefg 0.0\n"
            "lattice_vector 0.0 0.0 1.0\n"
            "atom 0.0 0.0 0.0 C\n"
        ),
        # more than three lattice vectors
        (
            "lattice_vector 1.0 0.0 0.0\n"
            "lattice_vector 0.0 1.0 0.0\n"
            "lattice_vector 0.0 0.0 1.0\n"
            "lattice_vector 1.0 1.0 1.0\n"
            "atom 0.0 0.0 0.0 C\n"
        ),
        # no atoms found (comments only)
        "# nothing\n# to\n# see\n# here\n",
        # malformed atom coordinate
        "atom 0.0 abcdefg 0.0 C\n",
        # "X" is tad-mctc's own dummy/padding-atom symbol (pse.S2Z["X"] ==
        # 0), not a real element -- must still be rejected, not silently
        # resolved to atomic number 0
        "atom 0.0 0.0 0.0 X\n",
    ],
    ids=[
        "unknown-element",
        "unexpected-keyword",
        "malformed-lattice-vector",
        "too-many-lattice-vectors",
        "no-atoms",
        "malformed-coordinate",
        "dummy-symbol",
    ],
)
def test_read_fail_format(content: str) -> None:
    tmpdir, filepath = _write(content)
    with tmpdir:
        with pytest.raises(FormatErrorAIMS):
            read.read_aims(filepath)
