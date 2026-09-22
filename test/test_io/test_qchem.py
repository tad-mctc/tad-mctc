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
Test the Q-Chem ``$molecule`` block reader, mirroring mctc-lib's
``mctc_io_read_qchem`` test cases -- including its Z-matrix (internal
coordinate) support.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from tad_mctc.exceptions import FormatErrorQChem
from tad_mctc.io import read
from tad_mctc.typing import DD
from tad_mctc.units import length


def _write(content: str) -> tuple[tempfile.TemporaryDirectory[str], Path]:
    tmpdir = tempfile.TemporaryDirectory()
    filepath = Path(tmpdir.name) / "mol.qchem"
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    return tmpdir, filepath


################################################################################
# cartesian
################################################################################


def test_read_cartesian_symbols() -> None:
    """mctc-lib's ``valid2``: element symbols, lowercase tags, leading
    blank line before ``$molecule``."""
    dd: DD = {"device": None, "dtype": torch.double}

    content = (
        "\n"
        "$molecule\n"
        "   0 1\n"
        "   O   0.000000   0.000000  -0.212195\n"
        "   H   1.370265   0.000000   0.848778\n"
        "   H  -1.370265   0.000000   0.848778\n"
        "$end\n"
    )
    tmpdir, filepath = _write(content)
    with tmpdir:
        result = read.read_qchem(filepath, **dd)

    assert result.lattice is None
    numbers, positions = result.numbers, result.positions

    ref_numbers = torch.tensor([8, 1, 1])
    ref_positions = (
        torch.tensor(
            [
                [0.000000, 0.000000, -0.212195],
                [1.370265, 0.000000, 0.848778],
                [-1.370265, 0.000000, 0.848778],
            ],
            **dd,
        )
        * length.AA2AU
    )

    assert (ref_numbers == numbers).all()
    assert pytest.approx(ref_positions.cpu()) == positions.cpu()


def test_read_cartesian_blank_line_between_atoms_skipped() -> None:
    """A blank line between atom records (not just before ``$molecule``)
    is skipped rather than ending the block early."""
    dd: DD = {"device": None, "dtype": torch.double}

    content = (
        "$molecule\n"
        "0 1\n"
        "O   0.000000   0.000000  -0.212195\n"
        "\n"
        "H   1.370265   0.000000   0.848778\n"
        "$end\n"
    )
    tmpdir, filepath = _write(content)
    with tmpdir:
        structure = read.read_qchem(filepath, **dd)
        numbers, positions = structure.numbers, structure.positions

    assert (numbers == torch.tensor([8, 1])).all()
    assert positions.shape == (2, 3)


def test_read_cartesian_atomic_numbers() -> None:
    """mctc-lib's ``valid1``: raw atomic numbers instead of element
    symbols, uppercase ``$MOLECULE``/``$END`` tags."""
    content = (
        "$MOLECULE\n"
        "   0 1\n"
        "   8   0.000000   0.000000  -0.212195\n"
        "   1   1.370265   0.000000   0.848778\n"
        "   1  -1.370265   0.000000   0.848778\n"
        "$END\n"
    )
    tmpdir, filepath = _write(content)
    with tmpdir:
        structure = read.read_qchem(filepath)
        numbers = structure.numbers

    assert (numbers == torch.tensor([8, 1, 1])).all()


def test_read_cartesian_lowercase_symbols_large() -> None:
    """mctc-lib's ``valid3``: a larger (24-atom, 4-species) molecule with
    all-lowercase element symbols and a lowercase ``$MOLECULE`` tag but
    uppercase ``$END`` terminator."""
    content = (
        "$MOLECULE\n"
        "0 1\n"
        "c  1.07317  0.04885 -0.07573\n"
        "n  2.51365  0.01256 -0.07580\n"
        "c  3.35199  1.09592 -0.07533\n"
        "n  4.61898  0.73028 -0.07549\n"
        "c  4.57907 -0.63144 -0.07531\n"
        "c  3.30131 -1.10256 -0.07524\n"
        "c  2.98068 -2.48687 -0.07377\n"
        "o  1.82530 -2.90038 -0.07577\n"
        "n  4.11440 -3.30433 -0.06936\n"
        "c  5.45174 -2.85618 -0.07235\n"
        "o  6.38934 -3.65965 -0.07232\n"
        "n  5.66240 -1.47682 -0.07487\n"
        "c  7.00947 -0.93648 -0.07524\n"
        "c  3.92063 -4.74093 -0.06158\n"
        "h  0.73398  1.08786 -0.07503\n"
        "h  0.71239 -0.45698  0.82335\n"
        "h  0.71240 -0.45580 -0.97549\n"
        "h  2.99301  2.11762 -0.07478\n"
        "h  7.76531 -1.72634 -0.07591\n"
        "h  7.14864 -0.32182  0.81969\n"
        "h  7.14802 -0.32076 -0.96953\n"
        "h  2.86501 -5.02316 -0.05833\n"
        "h  4.40233 -5.15920  0.82837\n"
        "h  4.40017 -5.16929 -0.94780\n"
        "$END\n"
    )
    tmpdir, filepath = _write(content)
    with tmpdir:
        structure = read.read_qchem(filepath)
        numbers, positions = structure.numbers, structure.positions

    assert numbers.shape[0] == 24
    assert positions.shape == (24, 3)

    species = torch.unique(numbers)
    assert species.shape[0] == 4
    assert set(species.tolist()) == {1, 6, 7, 8}


################################################################################
# Z-matrix (internal coordinates)
################################################################################


def test_read_zmatrix() -> None:
    """mctc-lib's ``valid4``: a Z-matrix mixing all three placement modes
    (distance only, distance+angle, distance+angle+dihedral)."""
    dd: DD = {"device": None, "dtype": torch.double}

    content = (
        "$molecule\n"
        "0 1\n"
        "P\n"
        "H  1 1.407461\n"
        "H  1 1.407521 2 100.786448\n"
        "H  1 1.407521 2 100.786448 3 103.310033 0\n"
        "O  1 1.487800 2 117.174945 3 -128.344983 0\n"
        "$end\n"
    )
    tmpdir, filepath = _write(content)
    with tmpdir:
        structure = read.read_qchem(filepath, **dd)
        numbers, positions = structure.numbers, structure.positions

    assert (numbers == torch.tensor([15, 1, 1, 1, 8])).all()
    assert positions.shape == (5, 3)

    # each Z-matrix atom's distance to its first reference atom (P, at the
    # origin) must equal the value given in its line
    p_pos = positions[0]
    ref_distances_aa = [1.407461, 1.407521, 1.407521, 1.487800]
    for atom_idx, dist_aa in enumerate(ref_distances_aa, start=1):
        got = float(torch.linalg.norm(positions[atom_idx] - p_pos).cpu())
        assert got == pytest.approx(dist_aa * length.AA2AU, abs=1e-10)


def test_read_zmatrix_non_first_reference() -> None:
    """A Z-matrix reference need not be atom 1 -- atom 3 here references
    atom 2, which mctc-lib's own bound check (``ij(1) >= iat``, checked
    against the *not-yet-incremented* running atom count) must still
    accept since it is the immediately preceding atom."""
    dd: DD = {"device": None, "dtype": torch.double}

    content = "$molecule\n0 1\nO\nH 1 0.96\nH 2 0.96 1 104.5\n$end\n"
    tmpdir, filepath = _write(content)
    with tmpdir:
        structure = read.read_qchem(filepath, **dd)
        numbers, positions = structure.numbers, structure.positions

    assert (numbers == torch.tensor([8, 1, 1])).all()

    dist_atom2 = float(torch.linalg.norm(positions[1] - positions[0]).cpu())
    assert dist_atom2 == pytest.approx(0.96 * length.AA2AU, abs=1e-8)

    # atom 3 references atom 2 (not atom 1) for its distance
    dist_atom3 = float(torch.linalg.norm(positions[2] - positions[1]).cpu())
    assert dist_atom3 == pytest.approx(0.96 * length.AA2AU, abs=1e-8)


################################################################################
# error handling
################################################################################


def test_read_fail_notfound() -> None:
    with pytest.raises(FileNotFoundError):
        read.read_qchem("not found")


@pytest.mark.parametrize(
    "content",
    [
        # unknown element symbol that is also not a valid raw atomic number
        ("$molecule\n0 1\nhh 0.0 0.0 0.0\nH 1.0 0.0 0.0\n$end\n"),
        # mctc-lib's invalid1: unknown symbol after mode is already locked
        # in by six valid Cartesian atoms (not on the mode-deciding line)
        (
            "$molecule\n"
            "   0 1\n"
            "C   -0.0090   -0.0157   -0.0000\n"
            "C   -0.7131    1.2038   -0.0000\n"
            "C    1.3990   -0.0157   -0.0000\n"
            "C   -0.0090    2.4232   -0.0000\n"
            "C    2.1031    1.2038   -0.0000\n"
            "C    1.3990    2.4232    0.0000\n"
            "hh  -0.5203   -0.9011   -0.0000\n"
            "hh  -1.7355    1.2038    0.0000\n"
            "hh   1.9103   -0.9011    0.0000\n"
            "H   -0.5203    3.3087    0.0000\n"
            "H    3.1255    1.2038    0.0000\n"
            "H    1.9103    3.3087   -0.0000\n"
            "$end\n"
        ),
        # malformed charge/multiplicity line
        "$molecule\n24\nO 0.0 0.0 0.0\n$end\n",
        # missing $end terminator (unexpected end of input)
        "$molecule\n0 1\nO 0.0 0.0 0.0\nH 1.0 0.0 0.0\n",
        # no $molecule block at all
        "$mol\n0 1\nO 0.0 0.0 0.0\n$end\n",
        # malformed coordinate value
        "$molecule\n0 1\nO ****** 0.0 0.0\n$end\n",
        # cartesian coordinate line with fewer than 3 values
        "$molecule\n0 1\nO 0.0 0.0\n$end\n",
        # no atoms at all: $end immediately after the charge/multiplicity line
        "$molecule\n0 1\n$end\n",
        # non-numeric z-matrix reference index
        "$molecule\n0 1\nP\nH  x 1.407461\n$end\n",
        # dihedral-level z-matrix entry referencing a not-yet-defined atom
        (
            "$molecule\n"
            "0 1\n"
            "P\n"
            "H  1 1.407461\n"
            "H  1 1.407521 2 100.786448\n"
            "H  1 1.407521 2 100.786448 99 103.310033\n"
            "$end\n"
        ),
        # truncated z-matrix entry: reference index without a distance
        "$molecule\n0 1\nP\nH  1\n$end\n",
        # mctc-lib's invalid6: z-matrix atom with no reference data at all
        # (not even a reference index)
        (
            "$molecule\n"
            "0 1\n"
            "P\n"
            "H\n"
            "H  1 1.407521 2 100.786448\n"
            "H  1 1.407521 2 100.786448 3 103.310033 0\n"
            "O  1 1.487800 2 117.174945 3 -128.344983 0\n"
            "$end\n"
        ),
        # z-matrix atom referencing itself/a not-yet-defined atom
        "$molecule\n0 1\nP\nH  2 1.407461\n$end\n",
        # symbol that is neither a known element nor a valid atomic number
        "$molecule\n0 1\n* 0.0 0.0 0.0\n$end\n",
        # mctc-lib's invalid9: junk z-matrix symbol with no reference data
        (
            "$molecule\n"
            "0 1\n"
            "P\n"
            "*\n"
            "H  1 1.407521 2 100.786448\n"
            "H  1 1.407521 2 100.786448 3 103.310033 0\n"
            "O  1 1.487800 2 117.174945 3 -128.344983 0\n"
            "$end\n"
        ),
    ],
    ids=[
        "unknown-element",
        "unknown-element-after-mode-locked",
        "bad-charge-multiplicity",
        "missing-end",
        "no-molecule-block",
        "malformed-coordinate",
        "cartesian-too-few-values",
        "no-atoms",
        "non-numeric-zmatrix-refindex",
        "zmatrix-dihedral-reference-out-of-range",
        "truncated-zmatrix-entry",
        "truncated-zmatrix-entry-no-refindex",
        "zmatrix-forward-reference",
        "junk-symbol",
        "junk-zmatrix-symbol-no-refdata",
    ],
)
def test_read_fail_format(content: str) -> None:
    tmpdir, filepath = _write(content)
    with tmpdir:
        with pytest.raises(FormatErrorQChem):
            read.read_qchem(filepath)
