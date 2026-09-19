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
Test the MDL Molfile (V2000/V3000) and SDF reader, mirroring mctc-lib's
``mctc_io_read_ctfile`` test cases (fixture content taken directly from
mctc-lib's own ``test_read_ctfile.f90``).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from tad_mctc.exceptions import FormatErrorCTFile
from tad_mctc.io import read
from tad_mctc.typing import DD
from tad_mctc.units import length


def _write(
    content: str, suffix: str
) -> tuple[tempfile.TemporaryDirectory[str], Path]:
    tmpdir = tempfile.TemporaryDirectory()
    filepath = Path(tmpdir.name) / f"mol.{suffix}"
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    return tmpdir, filepath


_V2000_BENZENE = """\

  Mrv1823 10191918163D

 12 12  0  0  0  0            999 V2000
   -0.0090   -0.0157   -0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.7131    1.2038   -0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.3990   -0.0157   -0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.0090    2.4232   -0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    2.1031    1.2038   -0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.3990    2.4232    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.5203   -0.9011   -0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.7355    1.2038    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.9103   -0.9011    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.5203    3.3087    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    3.1255    1.2038    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.9103    3.3087   -0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
  2  1  4  0  0  0  0
  3  1  4  0  0  0  0
  4  2  4  0  0  0  0
  5  3  4  0  0  0  0
  6  4  4  0  0  0  0
  6  5  4  0  0  0  0
  1  7  1  0  0  0  0
  2  8  1  0  0  0  0
  3  9  1  0  0  0  0
  4 10  1  0  0  0  0
  5 11  1  0  0  0  0
  6 12  1  0  0  0  0
M  END
"""

_V3000_COMPOUND = """\
Compound 1017
     RDKit          3D

  0  0  0  0  0  0  0  0  0  0999 V3000
M  V30 BEGIN CTAB
M  V30 COUNTS 28 29 0 0 0
M  V30 BEGIN ATOM
M  V30 1 O -0.139802 -1.830034 0.709675 0
M  V30 2 C -0.316756 -0.419558 0.542247 0 CFG=2
M  V30 3 C -1.799960 -0.109911 0.488996 0
M  V30 4 C 0.386061 0.062191 -0.739099 0
M  V30 5 C -2.697868 -0.951923 -0.184715 0
M  V30 6 C -2.294606 1.060665 1.085325 0
M  V30 7 C 1.894644 0.013015 -0.632932 0
M  V30 8 C -4.055760 -0.636028 -0.254512 0
M  V30 9 C -3.653160 1.375870 1.014385 0
M  V30 10 N 2.442088 -0.971958 0.125897 0
M  V30 11 C 2.672560 0.914435 -1.356061 0
M  V30 12 C -4.532743 0.527936 0.344354 0
M  V30 13 C 3.791786 -1.035730 0.183623 0
M  V30 14 C 4.059439 0.822550 -1.276192 0
M  V30 15 C 4.633339 -0.168766 -0.490541 0
M  V30 16 H 0.832285 -1.959785 0.673224 0
M  V30 17 H 0.117435 0.060361 1.428520 0
M  V30 18 H 0.082070 1.094331 -0.955596 0
M  V30 19 H 0.094415 -0.560551 -1.594197 0
M  V30 20 H -2.337706 -1.867971 -0.649441 0
M  V30 21 H -1.623344 1.733954 1.613637 0
M  V30 22 H -4.740022 -1.302611 -0.772469 0
M  V30 23 H -4.025651 2.281920 1.484883 0
M  V30 24 H 2.212518 1.677511 -1.976085 0
M  V30 25 H -5.590493 0.771200 0.292658 0
M  V30 26 H 4.192367 -1.831268 0.805865 0
M  V30 27 H 4.686889 1.517092 -1.827730 0
M  V30 28 H 5.709975 -0.266939 -0.409811 0
M  V30 END ATOM
M  V30 BEGIN BOND
M  V30 1 1 2 1
M  V30 2 1 2 3
M  V30 3 1 2 4
M  V30 4 2 3 5
M  V30 5 1 3 6
M  V30 6 1 4 7
M  V30 7 1 5 8
M  V30 8 2 6 9
M  V30 9 2 7 10
M  V30 10 1 7 11
M  V30 11 2 8 12
M  V30 12 1 10 13
M  V30 13 2 11 14
M  V30 14 2 13 15
M  V30 15 1 9 12
M  V30 16 1 14 15
M  V30 17 1 1 16
M  V30 18 1 2 17 CFG=1
M  V30 19 1 4 18
M  V30 20 1 4 19
M  V30 21 1 5 20
M  V30 22 1 6 21
M  V30 23 1 8 22
M  V30 24 1 9 23
M  V30 25 1 11 24
M  V30 26 1 12 25
M  V30 27 1 13 26
M  V30 28 1 14 27
M  V30 29 1 15 28
M  V30 END BOND
M  V30 BEGIN COLLECTION
M  V30 MDLV30/STERAC1 ATOMS=(1 2)
M  V30 END COLLECTION
M  V30 END CTAB
M  END
"""


def test_read_v2000_molfile() -> None:
    """mctc-lib's ``valid1-mol``: a simple V2000 benzene, all-carbon-ring
    aromatic bonds plus single C-H bonds."""
    dd: DD = {"device": None, "dtype": torch.double}

    tmpdir, filepath = _write(_V2000_BENZENE, "mol")
    with tmpdir:
        result = read.read_molfile(filepath, **dd)

    numbers, positions, lattice, periodic, bonds, bond_orders = result
    assert lattice is None
    assert periodic is None
    assert bonds is not None
    assert bond_orders is not None

    assert numbers.shape == (12,)
    assert len(torch.unique(numbers)) == 2
    assert bonds.shape == (12, 2)
    # first bond record "2 1 4" -> 0-indexed (1, 0), aromatic (order 4)
    assert (bonds[0] == torch.tensor([1, 0])).all()
    assert bond_orders[0] == pytest.approx(4.0)

    ref_first = torch.tensor([-0.0090, -0.0157, -0.0000], **dd) * length.AA2AU
    assert pytest.approx(ref_first.cpu()) == positions[0].cpu()


def test_read_v2000_symbol_quirks() -> None:
    """Isotope-prefixed (``18O``), asterisk-suffixed (``C*``), and
    deuterium (``D``) symbols resolve the same way as in aims/qchem."""
    content = (
        "\n"
        "  xtb     09072013503D\n"
        " xtb: 6.3.2 (b5103a3)\n"
        "  3  2  0     0  0            999 V2000\n"
        "    1.0732    0.0488   -0.0757 C*  0  0  0  0  0  0  0  0  0  0  0  0\n"
        "    1.8253   -2.9004   -0.0758 18O 0  0  0  0  0  0  0  0  0  0  0  0\n"
        "    0.7340    1.0879   -0.0750 D   0  0  0  0  0  0  0  0  0  0  0  0\n"
        "  1  2  1  0  0  0  0\n"
        "  1  3  1  0  0  0  0\n"
        "M  END\n"
    )
    tmpdir, filepath = _write(content, "mol")
    with tmpdir:
        numbers, _, _, _, _, _ = read.read_molfile(filepath)

    assert (numbers == torch.tensor([6, 8, 1])).all()


def test_read_v2000_with_charge_property_ignored() -> None:
    """An ``M  CHG`` property line (per-atom formal charge override) must
    not break parsing -- charge is validated-and-dropped like every other
    embedded charge/multiplicity field in this package."""
    content = (
        "\n"
        "  Mrv1823\n"
        "\n"
        "  2  1  0  0  0  0            999 V2000\n"
        "    0.0000    0.0000    0.0000 N   0  0  0  0  0  0  0  0  0  0  0  0\n"
        "    1.0000    0.0000    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0\n"
        "  1  2  1  0  0  0  0\n"
        "M  CHG  1   1   1\n"
        "M  END\n"
    )
    tmpdir, filepath = _write(content, "mol")
    with tmpdir:
        numbers, positions, _, _, bonds, bond_orders = read.read_molfile(
            filepath
        )

    assert (numbers == torch.tensor([7, 1])).all()
    assert bonds is not None and bond_orders is not None
    assert bonds.shape == (1, 2)
    assert positions.shape == (2, 3)


def test_read_v2000_short_property_columns() -> None:
    """mctc-lib's ``valid5-mol``: the same benzene connection table as
    ``valid1-mol``, but every atom/bond record's trailing property
    columns (isotope/charge/... and stereo/topology/...) are cut short
    instead of padded out to all twelve/seven fields. The fixed-column
    parser only reads coordinates/symbol and the first three bond
    fields, so short trailing columns must not raise."""
    content = (
        "\n"
        "  Mrv1823 10191918163D          \n"
        "\n"
        " 12 12  0  0  0  0            999 V2000\n"
        "   -0.0090   -0.0157   -0.0000 C   0\n"
        "   -0.7131    1.2038   -0.0000 C   0\n"
        "    1.3990   -0.0157   -0.0000 C   0\n"
        "   -0.0090    2.4232   -0.0000 C   0\n"
        "    2.1031    1.2038   -0.0000 C   0\n"
        "    1.3990    2.4232    0.0000 C   0\n"
        "   -0.5203   -0.9011   -0.0000 H   0\n"
        "   -1.7355    1.2038    0.0000 H   0\n"
        "    1.9103   -0.9011    0.0000 H   0\n"
        "   -0.5203    3.3087    0.0000 H   0\n"
        "    3.1255    1.2038    0.0000 H   0\n"
        "    1.9103    3.3087   -0.0000 H   0\n"
        "  2  1  4  0\n"
        "  3  1  4  0\n"
        "  4  2  4  0\n"
        "  5  3  4  0\n"
        "  6  4  4  0\n"
        "  6  5  4  0\n"
        "  1  7  1  0\n"
        "  2  8  1  0\n"
        "  3  9  1  0\n"
        "  4 10  1  0\n"
        "  5 11  1  0\n"
        "  6 12  1  0\n"
        "M  END\n"
    )
    tmpdir, filepath = _write(content, "mol")
    with tmpdir:
        numbers, positions, _, _, bonds, bond_orders = read.read_molfile(
            filepath
        )

    assert bonds is not None and bond_orders is not None
    assert numbers.shape == (12,)
    assert len(torch.unique(numbers)) == 2
    assert bonds.shape == (12, 2)
    assert positions.shape == (12, 3)
    assert (bonds[0] == torch.tensor([1, 0])).all()
    assert bond_orders[0] == pytest.approx(4.0)


def test_read_v2000_fail_missing_header_line() -> None:
    """mctc-lib's ``invalid1-mol``: one header line is missing, so every
    later line shifts up by one -- the real counts line is consumed as
    the (discarded) comment line, and the first atom record is misread
    as the counts line, which fails to parse as two integers."""
    content = (
        " OpenBabel10191918023D\n"
        "\n"
        " 12 12  0  0  0  0  0  0  0  0999 V2000\n"
        "   -0.0090   -0.0157    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n"
    )
    tmpdir, filepath = _write(content, "mol")
    with tmpdir:
        with pytest.raises(
            FormatErrorCTFile, match="Cannot read header of molfile"
        ):
            read.read_molfile(filepath)


def test_read_v2000_fail_undersupplied_atom_lines() -> None:
    """mctc-lib's ``invalid2-mol``: the header declares 12 atoms but only
    6 atom records are present before ``M  END`` -- the 7th read
    misparses the properties terminator as a coordinate line."""
    content = (
        "\n"
        "          10191918023D\n"
        "\n"
        " 12 12  0  0  0  0  0  0  0  0999 V2000\n"
        "   -0.0090   -0.0157    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n"
        "   -0.7131    1.2038    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n"
        "    1.3990   -0.0157    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n"
        "   -0.0090    2.4232    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n"
        "    2.1031    1.2038    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n"
        "    1.3990    2.4232    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n"
        "M  END\n"
    )
    tmpdir, filepath = _write(content, "mol")
    with tmpdir:
        with pytest.raises(
            FormatErrorCTFile,
            match="Cannot read coordinates from connection table",
        ):
            read.read_molfile(filepath)


def test_read_v3000_molfile() -> None:
    """mctc-lib's ``valid3-mol``: V3000 with per-atom/-bond ``CFG=`` stereo
    properties and a ``COLLECTION`` block, both ignored."""
    tmpdir, filepath = _write(_V3000_COMPOUND, "mol")
    with tmpdir:
        numbers, positions, lattice, periodic, bonds, bond_orders = (
            read.read_molfile(filepath)
        )

    assert lattice is None
    assert periodic is None
    assert numbers.shape == (28,)
    assert len(torch.unique(numbers)) == 4
    assert bonds is not None and bond_orders is not None
    assert bonds.shape == (29, 2)
    assert positions.shape == (28, 3)
    # first bond record "1 1 2 1" -> index=1, type=1, atoms (2, 1)
    # (mctc-lib's own field order is index/type/atom1/atom2) -> 0-indexed
    # (1, 0)
    assert (bonds[0] == torch.tensor([1, 0])).all()
    assert bond_orders[0] == pytest.approx(1.0)


# Shared atom/bond blocks for mctc-lib's "Compound 11" V3000 fixture family
# (``invalid6``-``invalid9-mol``): 17 atoms, 16 bonds. Each variant below
# only differs in the COUNTS line, an extra nested block, or whether the
# CTAB is properly closed -- factored here so each test's diff against the
# Fortran source is a one-line change, not a ~40-line fixture.
_V3000_C11_ATOMS = (
    "M  V30 1 O -2.821131 -0.276238 -0.753131 0\n"
    "M  V30 2 C -2.076407 0.000289 0.175864 0\n"
    "M  V30 3 O -2.469860 0.872693 1.126516 0\n"
    "M  V30 4 C -0.648307 -0.508439 0.384207 0\n"
    "M  V30 5 N -0.553725 -1.908221 -0.092137 0\n"
    "M  V30 6 C 0.306640 0.448659 -0.352110 0\n"
    "M  V30 7 C 1.764852 0.167437 -0.097096 0\n"
    "M  V30 8 C 2.575104 0.984442 0.587951 0\n"
    "M  V30 9 H -3.391314 1.091514 0.873612 0\n"
    "M  V30 10 H -0.438887 -0.513473 1.460318 0\n"
    "M  V30 11 H 0.421206 -2.197466 -0.111774 0\n"
    "M  V30 12 H -0.893195 -1.946576 -1.055443 0\n"
    "M  V30 13 H 0.073377 1.483235 -0.066299 0\n"
    "M  V30 14 H 0.129860 0.396798 -1.434760 0\n"
    "M  V30 15 H 2.179323 -0.745345 -0.519251 0\n"
    "M  V30 16 H 2.219589 1.914253 1.021797 0\n"
    "M  V30 17 H 3.622875 0.736438 0.730780 0\n"
)
_V3000_C11_BONDS = (
    "M  V30 1 2 1 2\n"
    "M  V30 2 1 2 3\n"
    "M  V30 3 1 4 2\n"
    "M  V30 4 1 4 5\n"
    "M  V30 5 1 4 6\n"
    "M  V30 6 1 6 7\n"
    "M  V30 7 2 7 8 CFG=2\n"
    "M  V30 8 1 3 9\n"
    "M  V30 9 1 4 10 CFG=1\n"
    "M  V30 10 1 5 11\n"
    "M  V30 11 1 5 12\n"
    "M  V30 12 1 6 13\n"
    "M  V30 13 1 6 14\n"
    "M  V30 14 1 7 15\n"
    "M  V30 15 1 8 16\n"
    "M  V30 16 1 8 17\n"
)


def _v3000_compound11(
    counts: str, extra_block: str = "", end_ctab: bool = True
) -> str:
    content = (
        "Compound 11\n"
        "     RDKit          3D\n"
        "\n"
        "  0  0  0  0  0  0  0  0  0  0999 V3000\n"
        "M  V30 BEGIN CTAB\n"
        f"M  V30 COUNTS {counts}\n"
        "M  V30 BEGIN ATOM\n"
        + _V3000_C11_ATOMS
        + "M  V30 END ATOM\n"
        + "M  V30 BEGIN BOND\n"
        + _V3000_C11_BONDS
        + "M  V30 END BOND\n"
        + extra_block
    )
    if end_ctab:
        content += "M  V30 END CTAB\n"
    content += "M  END\n"
    return content


def test_read_v3000_fail_atom_count_mismatch() -> None:
    """mctc-lib's ``invalid6-mol``: COUNTS declares 16 atoms but the ATOM
    block actually holds 17 records -- after consuming the declared 16,
    the 17th atom record is checked against ``END ATOM`` and fails."""
    content = _v3000_compound11("16 16 0 0 0")
    tmpdir, filepath = _write(content, "mol")
    with tmpdir:
        with pytest.raises(
            FormatErrorCTFile, match="ATOM block is not terminated"
        ):
            read.read_molfile(filepath)


def test_read_v3000_fail_bond_count_mismatch() -> None:
    """mctc-lib's ``invalid8-mol``: COUNTS declares 15 bonds but the BOND
    block actually holds 16 records -- analogous to the atom-count
    mismatch above, but for the BOND block."""
    content = _v3000_compound11("17 15 0 0 0")
    tmpdir, filepath = _write(content, "mol")
    with tmpdir:
        with pytest.raises(
            FormatErrorCTFile, match="BOND block is not terminated"
        ):
            read.read_molfile(filepath)


def test_read_v3000_fail_unknown_block() -> None:
    """mctc-lib's ``invalid7-mol``: an unrecognized nested block name
    (``BEGIN INVALID`` / ``END INVALID``) inside an otherwise well-formed
    CTAB must be rejected rather than silently skipped."""
    content = _v3000_compound11(
        "17 16 0 0 0",
        extra_block="M  V30 BEGIN INVALID\nM  V30 END INVALID\n",
    )
    tmpdir, filepath = _write(content, "mol")
    with tmpdir:
        with pytest.raises(
            FormatErrorCTFile, match="Unknown connection table entry"
        ):
            read.read_molfile(filepath)


def test_read_v3000_fail_missing_end_ctab() -> None:
    """mctc-lib's ``invalid9-mol``: the CTAB is never closed with
    ``M  V30 END CTAB`` before ``M  END`` -- the connection-table scanner
    runs off the end of the file looking for the next block."""
    content = _v3000_compound11("17 16 0 0 0", end_ctab=False)
    tmpdir, filepath = _write(content, "mol")
    with tmpdir:
        with pytest.raises(
            FormatErrorCTFile, match="while reading the connection table"
        ):
            read.read_molfile(filepath)


def test_read_v3000_fail_garbage_in_counts_extra_field() -> None:
    """mctc-lib's ``invalid7-sdf``: the COUNTS line's 3rd field (an unused
    sgroup/3d-obj/chiral-flag placeholder) is garbage (``"a"``) rather than
    an integer. mctc-lib reads and validates all 5 COUNTS fields as
    integers even though 3 of them are otherwise dropped, so this must
    fail the same way a garbage atom/bond count would."""
    content = _v3000_compound11("17 16 a 0 0")
    tmpdir, filepath = _write(content, "mol")
    with tmpdir:
        with pytest.raises(
            FormatErrorCTFile,
            match="Cannot read connection table counts",
        ):
            read.read_molfile(filepath)


def test_read_v3000_fail_missing_counts_header() -> None:
    """The CTAB's first entry must be a ``COUNTS`` record -- anything
    else, even otherwise well-formed V3000 syntax, is rejected."""
    content = (
        "Compound\n"
        "     RDKit          3D\n"
        "\n"
        "  0  0  0  0  0  0  0  0  0  0999 V3000\n"
        "M  V30 BEGIN CTAB\n"
        "M  V30 BEGIN ATOM\n"
        "M  V30 END ATOM\n"
        "M  V30 END CTAB\n"
        "M  END\n"
    )
    tmpdir, filepath = _write(content, "mol")
    with tmpdir:
        with pytest.raises(FormatErrorCTFile, match="COUNTS header not found"):
            read.read_molfile(filepath)


def test_read_v3000_skips_unrecognized_ctab_entry() -> None:
    """A top-level CTAB entry that is neither a ``BEGIN`` block nor the
    closing ``END CTAB`` (e.g. an optional ``LINKNODE`` line) is skipped
    -- only an unrecognized *block* name is an error (see the
    unknown-block test above)."""
    content = _v3000_compound11(
        "17 16 0 0 0", extra_block="M  V30 LINKNODE 1 2 3 4 5 6\n"
    )
    tmpdir, filepath = _write(content, "mol")
    with tmpdir:
        numbers, *_ = read.read_molfile(filepath)

    assert numbers.shape == (17,)


def test_read_v3000_fail_malformed_atom_record() -> None:
    """A truncated V3000 ATOM record (missing coordinate/mapping fields)
    is a format error, not an uncaught ``IndexError``/``ValueError``."""
    content = (
        "Compound\n"
        "     RDKit          3D\n"
        "\n"
        "  0  0  0  0  0  0  0  0  0  0999 V3000\n"
        "M  V30 BEGIN CTAB\n"
        "M  V30 COUNTS 1 0 0 0 0\n"
        "M  V30 BEGIN ATOM\n"
        "M  V30 1 C 0.0 0.0\n"
        "M  V30 END ATOM\n"
        "M  V30 END CTAB\n"
        "M  END\n"
    )
    tmpdir, filepath = _write(content, "mol")
    with tmpdir:
        with pytest.raises(FormatErrorCTFile, match="Cannot read coordinates"):
            read.read_molfile(filepath)


def test_read_v3000_fail_malformed_bond_record() -> None:
    """A truncated V3000 BOND record (missing an atom index) is a format
    error, not an uncaught ``IndexError``/``ValueError``."""
    content = (
        "Compound\n"
        "     RDKit          3D\n"
        "\n"
        "  0  0  0  0  0  0  0  0  0  0999 V3000\n"
        "M  V30 BEGIN CTAB\n"
        "M  V30 COUNTS 2 1 0 0 0\n"
        "M  V30 BEGIN ATOM\n"
        "M  V30 1 C 0.0 0.0 0.0 0\n"
        "M  V30 2 C 1.0 0.0 0.0 0\n"
        "M  V30 END ATOM\n"
        "M  V30 BEGIN BOND\n"
        "M  V30 1 1 1\n"
        "M  V30 END BOND\n"
        "M  V30 END CTAB\n"
        "M  END\n"
    )
    tmpdir, filepath = _write(content, "mol")
    with tmpdir:
        with pytest.raises(
            FormatErrorCTFile, match="Cannot read bond information"
        ):
            read.read_molfile(filepath)


def test_read_sdf_wrapper() -> None:
    """mctc-lib's ``read_sdf``: the same connection table, plus a trailing
    key-value data block terminated by ``$$$$``."""
    content = (
        _V2000_BENZENE.rstrip("\n")
        + "\n"
        + ">  <smiles>  (1)\nc1ccccc1\n\n$$$$\n"
    )
    tmpdir, filepath = _write(content, "sdf")
    with tmpdir:
        numbers, positions, _, _, bonds, _ = read.read_sdf(filepath)

    assert numbers.shape == (12,)
    assert bonds is not None
    assert bonds.shape == (12, 2)


def test_read_sdf_fail_missing_terminator() -> None:
    """mctc-lib's ``read_sdf`` requires the ``$$$$`` terminator; a data
    block that runs to EOF without one is an error, not a silent success."""
    content = (
        _V2000_BENZENE.rstrip("\n") + "\n" + ">  <smiles>  (1)\nc1ccccc1\n"
    )
    tmpdir, filepath = _write(content, "sdf")
    with tmpdir:
        with pytest.raises(FormatErrorCTFile):
            read.read_sdf(filepath)


def test_read_sdf_fail_bond_count_mismatch() -> None:
    """mctc-lib's ``invalid2-sdf``: the counts line declares 18 bonds but
    only 12 bond records are present -- reading past them misparses
    ``M  END`` as a bond record."""
    content = (
        _V2000_BENZENE.replace(
            " 12 12  0  0  0  0            999 V2000",
            " 12 18  0  0  0  0            999 V2000",
        ).rstrip("\n")
        + "\n$$$$\n"
    )
    tmpdir, filepath = _write(content, "sdf")
    with tmpdir:
        with pytest.raises(
            FormatErrorCTFile,
            match="Cannot read topology from connection table",
        ):
            read.read_sdf(filepath)


def test_read_sdf_fail_missing_molfile_end() -> None:
    """mctc-lib's ``invalid3-sdf``: the connection table's own ``M  END``
    terminator is missing -- the bond block runs straight into the SDF
    data block, which is then scanned forever (through the eventual
    ``$$$$``) looking for a properties terminator that never comes."""
    content = (
        _V2000_BENZENE.replace("M  END\n", "").rstrip("\n")
        + "\n"
        + "> <smiles>  (1)\nc1ccccc1\n\n$$$$\n"
    )
    tmpdir, filepath = _write(content, "sdf")
    with tmpdir:
        with pytest.raises(
            FormatErrorCTFile, match="while reading the properties block"
        ):
            read.read_sdf(filepath)


def test_read_sdf_v3000() -> None:
    """Real coverage gap (mctc-lib never exercises this combination
    directly): ``read_sdf`` wrapping a V3000 record, mirroring
    ``valid3-sdf``'s fixture content but through the SDF entry point
    instead of ``read_molfile`` -- the SDF ``$$$$`` scan must work
    whichever molfile version produced the connection table."""
    content = (
        _V3000_COMPOUND.rstrip("\n")
        + "\n"
        + ">  <smiles>  (1)\nNC(CC=C)C(=O)O\n\n$$$$\n"
    )
    tmpdir, filepath = _write(content, "sdf")
    with tmpdir:
        result_sdf = read.read_sdf(filepath)

    tmpdir2, filepath2 = _write(_V3000_COMPOUND, "mol")
    with tmpdir2:
        result_mol = read.read_molfile(filepath2)

    for got, expected in zip(result_sdf, result_mol):
        if got is None or expected is None:
            assert got is expected
        else:
            assert torch.equal(got, expected)


def test_read_fail_notfound() -> None:
    with pytest.raises(FileNotFoundError):
        read.read_molfile("not found")


@pytest.mark.parametrize(
    "content",
    [
        # invalid format version (neither V2000 nor V3000)
        (
            "\n\n\n"
            " 12 12  0  0  0  0            999 V1000\n"
            "   -0.0090   -0.0157   -0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n"
        ),
        # non-positive atom count (V2000)
        ("\n\n\n" "  0 12  0  0  0  0            999 V2000\n" "M  END\n"),
        # unknown element symbol (V2000)
        (
            "\n\n\n"
            "  1  0  0  0  0  0            999 V2000\n"
            "    0.0000    0.0000    0.0000 Xx  0  0  0  0  0  0  0  0  0  0  0  0\n"
            "M  END\n"
        ),
        # malformed coordinate (V2000)
        (
            "\n\n\n"
            "  1  0  0  0  0  0            999 V2000\n"
            "    abcdefgh    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n"
            "M  END\n"
        ),
        # dummy/wildcard atom symbol "*" (V3000, unsupported)
        (
            "\n\n\n"
            "  0  0  0  0  0  0  0  0  0  0999 V3000\n"
            "M  V30 BEGIN CTAB\n"
            "M  V30 COUNTS 1 0 0 0 0\n"
            "M  V30 BEGIN ATOM\n"
            "M  V30 1 * 0.0 0.0 0.0 0\n"
            "M  V30 END ATOM\n"
            "M  V30 END CTAB\n"
            "M  END\n"
        ),
        # zero atom count (V3000)
        (
            "\n\n\n"
            "  0  0  0  0  0  0  0  0  0  0999 V3000\n"
            "M  V30 BEGIN CTAB\n"
            "M  V30 COUNTS 0 0 0 0 0\n"
            "M  V30 END CTAB\n"
            "M  END\n"
        ),
        # missing "BEGIN CTAB" (V3000)
        ("\n\n\n" "  0  0  0  0  0  0  0  0  0  0999 V3000\n" "M  END\n"),
        # mapped atom (aamap > 0) is not supported (V3000)
        (
            "\n\n\n"
            "  0  0  0  0  0  0  0  0  0  0999 V3000\n"
            "M  V30 BEGIN CTAB\n"
            "M  V30 COUNTS 1 0 0 0 0\n"
            "M  V30 BEGIN ATOM\n"
            "M  V30 1 C 0.0 0.0 0.0 1\n"
            "M  V30 END ATOM\n"
            "M  V30 END CTAB\n"
            "M  END\n"
        ),
    ],
    ids=[
        "invalid-version",
        "non-positive-atom-count",
        "unknown-element",
        "malformed-coordinate",
        "v3000-dummy-atom",
        "v3000-zero-atoms",
        "v3000-missing-ctab",
        "v3000-mapped-atom",
    ],
)
def test_read_fail_format(content: str) -> None:
    tmpdir, filepath = _write(content, "mol")
    with tmpdir:
        with pytest.raises(FormatErrorCTFile):
            read.read_molfile(filepath)
