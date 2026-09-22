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
Test the XYZ file reader and writer.
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from tad_mctc.batch import pack
from tad_mctc.data.structures import get_structure
from tad_mctc.exceptions import EmptyFileError, FormatErrorXYZ
from tad_mctc.io import read, write
from tad_mctc.io.read.xyz import _parse_atom_block
from tad_mctc.io.structure import Structure
from tad_mctc.typing import DD
from tad_mctc.units import length

from ..conftest import DEVICE
from ..utils import load_pair, load_sample

_SAMPLE_SOURCES: list[tuple[str, str]] = [("heavy28", "h2o")]


def _read(text: str) -> Structure:
    return read.xyz.read_xyz_fileobj(io.StringIO(text))


def test_read_fail() -> None:
    with pytest.raises(FileNotFoundError):
        read.read_xyz("not found")

    p = Path(__file__).parent.resolve() / "fail" / "nat-no-digit.xyz"
    with pytest.raises(FormatErrorXYZ):
        read.read_xyz(p)


def test_write_fail() -> None:
    sample = get_structure("heavy28", "h2o")
    numbers = sample.numbers
    positions = sample.positions

    with pytest.raises(FileExistsError):
        with tempfile.TemporaryDirectory() as tmpdirname:
            # create file
            filepath = Path(tmpdirname) / "already_exists"
            with filepath.open("w", encoding="utf-8") as f:
                f.write("")

            # try writing to just created file
            write.write_xyz(filepath, numbers, positions)


def test_write_batch_fail() -> None:
    sample = get_structure("heavy28", "h2o")

    with tempfile.TemporaryDirectory() as tmpdirname:
        filepath = Path(tmpdirname) / "dummy.xyz"

        # numbers batched, positions not
        numbers = pack((sample.numbers, sample.numbers))
        positions = sample.positions
        with pytest.raises(ValueError):
            write.write_xyz(filepath, numbers, positions, overwrite=True)

        # positions batched, numbers not
        numbers = sample.numbers
        positions = pack((sample.positions, sample.positions))
        with pytest.raises(ValueError):
            write.write_xyz(filepath, numbers, positions, overwrite=True)

        # too many dimensions
        numbers = torch.rand(2, 3, 4)
        positions = torch.rand(2, 3, 4, 3)
        with pytest.raises(ValueError):
            write.write_xyz(filepath, numbers, positions, overwrite=True)


def test_write_comment_fail() -> None:
    sample = get_structure("heavy28", "h2o")
    numbers = sample.numbers
    positions = sample.positions

    comment = "Comment \n with \n linebreaks \n"

    with pytest.raises(ValueError):
        with tempfile.TemporaryDirectory() as tmpdirname:
            fp = Path(tmpdirname) / "already_exists"
            write.write_xyz(fp, numbers, positions, comment=comment)


##############################################################################


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("collection,record", _SAMPLE_SOURCES)
@pytest.mark.parametrize("batch_agnostic", [True, False])
def test_write_and_read(
    dtype: torch.dtype, collection: str, record: str, batch_agnostic: bool
) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    numbers, positions = load_sample(collection, record, dd)

    # Create a temporary directory to save the file
    with tempfile.TemporaryDirectory() as tmpdirname:
        filepath = Path(tmpdirname) / f"{collection}_{record}.xyz"

        # Write to XYZ file
        write.write_xyz(filepath, numbers, positions)

        # Read from XYZ file
        structure = read.read_xyz(filepath, batch_agnostic=batch_agnostic, **dd)
        read_numbers, read_positions = structure.numbers, structure.positions

    if batch_agnostic is True:
        numbers = numbers.unsqueeze(0)
        positions = positions.unsqueeze(0)

    # Check if the read data matches the written data
    assert read_numbers.dtype == numbers.dtype
    assert read_numbers.shape == numbers.shape
    assert (read_numbers == numbers).all()
    assert pytest.approx(positions.cpu()) == read_positions.cpu()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("collection1,record1", _SAMPLE_SOURCES)
@pytest.mark.parametrize("collection2,record2", _SAMPLE_SOURCES)
def test_write_and_read_batch(
    dtype: torch.dtype,
    collection1: str,
    record1: str,
    collection2: str,
    record2: str,
) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    numbers1, positions1 = load_sample(collection1, record1, dd)
    numbers2, positions2 = load_sample(collection2, record2, dd)

    numbers_batch = pack((numbers1, numbers2))
    positions_batch = pack((positions1, positions2))

    # Create a temporary directory to save the file
    with tempfile.TemporaryDirectory() as tmpdirname:
        filepath = Path(tmpdirname) / "combined.xyz"

        # Write first structure to XYZ file
        write.write_xyz(filepath, numbers1, positions1, mode="w")

        # Append second structure to the same XYZ file
        write.write_xyz(filepath, numbers2, positions2, mode="a")

        # Read from XYZ file
        structure = read.read_xyz(filepath, **dd)
        read_numbers, read_positions = structure.numbers, structure.positions

    # Check if the read data matches the written data
    assert (read_numbers == numbers_batch).all()
    assert pytest.approx(positions_batch.cpu()) == read_positions.cpu()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("collection1,record1", _SAMPLE_SOURCES)
@pytest.mark.parametrize("collection2,record2", _SAMPLE_SOURCES)
def test_write_batch_and_read_batch(
    dtype: torch.dtype,
    collection1: str,
    record1: str,
    collection2: str,
    record2: str,
) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    numbers, positions = load_pair(
        collection1, record1, collection2, record2, dd
    )

    # Create a temporary directory to save the file
    with tempfile.TemporaryDirectory() as tmpdirname:
        filepath = Path(tmpdirname) / "combined.xyz"

        # Write structure to XYZ file
        write.write_xyz(filepath, numbers, positions, mode="w")

        # Read from XYZ file
        structure = read.read_xyz(filepath, **dd)
        read_numbers, read_positions = structure.numbers, structure.positions

    # Check if the read data matches the written data
    assert read_numbers.dtype == numbers.dtype
    assert read_numbers.shape == numbers.shape
    assert (read_numbers == numbers).all()
    assert pytest.approx(positions.cpu()) == read_positions.cpu()


def test_parse_atom_block_zero_atoms() -> None:
    """A zero-atom frame must short-circuit before `numpy.loadtxt`, which
    cannot parse an empty block."""
    numbers, coords = _parse_atom_block(io.StringIO(""), 0)

    assert numbers.shape == (0,)
    assert coords.shape == (0, 3)
    assert numbers.dtype == np.int64
    assert coords.dtype == np.float64


##############################################################################
# Ported from mctc-lib's test/test_read_xyz.f90 (non-extended-XYZ cases).
#
# mctc-lib's `structure_type` additionally exposes `comment` and `nid`
# (number of unique, string-identical species labels, e.g. "C" and "C*"
# count as two species even though both map to carbon). Neither concept
# exists in this reader's `(numbers, positions)` return value, so those
# specific assertions are not ported; only `nat` and the parsed content are
# checked here.
#
# The Extended XYZ cases (`valid6/7-extxyz`, `invalid1-12-extxyz`) are
# ported separately below, once Extended XYZ support was actually added.
##############################################################################


def test_valid1_xyz() -> None:
    """mctc-lib's ``test_valid1_xyz``: standard formatting with a non-empty
    comment line (dropped, since this reader does not expose the comment)."""
    text = (
        "9\n"
        "WATER27, (H2O)3\n"
        "O     1.1847029    1.1150792   -0.0344641 \n"
        "H     0.4939088    0.9563767    0.6340089 \n"
        "H     2.0242676    1.0811246    0.4301417 \n"
        "O    -1.1469443    0.0697649    1.1470196 \n"
        "H    -1.2798308   -0.5232169    1.8902833 \n"
        "H    -1.0641398   -0.4956693    0.3569250 \n"
        "O    -0.1633508   -1.0289346   -1.2401808 \n"
        "H     0.4914771   -0.3248733   -1.0784838 \n"
        "H    -0.5400907   -0.8496512   -2.1052499 \n"
    )

    structure = _read(text)

    numbers, positions = structure.numbers, structure.positions

    ref_numbers = torch.tensor([8, 1, 1, 8, 1, 1, 8, 1, 1])
    ref_positions = (
        torch.tensor(
            [
                [1.1847029, 1.1150792, -0.0344641],
                [0.4939088, 0.9563767, 0.6340089],
                [2.0242676, 1.0811246, 0.4301417],
                [-1.1469443, 0.0697649, 1.1470196],
                [-1.2798308, -0.5232169, 1.8902833],
                [-1.0641398, -0.4956693, 0.3569250],
                [-0.1633508, -1.0289346, -1.2401808],
                [0.4914771, -0.3248733, -1.0784838],
                [-0.5400907, -0.8496512, -2.1052499],
            ]
        )
        * length.AA2AU
    )

    assert numbers.shape == (9,)
    assert (numbers == ref_numbers).all()
    assert pytest.approx(ref_positions.cpu()) == positions.cpu()


def test_valid2_xyz_exotic_symbols() -> None:
    """mctc-lib's ``test_valid2_xyz``: decorated element symbols (isotope
    mass-number prefix, wildcard suffix, deuterium) that mctc-lib's
    ``symbol_to_number`` normalizes before lookup."""
    text = (
        "24\n"
        "\n"
        "C          1.07317        0.04885       -0.07573\n"
        "N          2.51365        0.01256       -0.07580\n"
        "C*         3.35199        1.09592       -0.07533\n"
        "N          4.61898        0.73028       -0.07549\n"
        "C*         4.57907       -0.63144       -0.07531\n"
        "C          3.30131       -1.10256       -0.07524\n"
        "C          2.98068       -2.48687       -0.07377\n"
        "18O        1.82530       -2.90038       -0.07577\n"
        "N          4.11440       -3.30433       -0.06936\n"
        "C*         5.45174       -2.85618       -0.07235\n"
        "O          6.38934       -3.65965       -0.07232\n"
        "N          5.66240       -1.47682       -0.07487\n"
        "C          7.00947       -0.93648       -0.07524\n"
        "C          3.92063       -4.74093       -0.06158\n"
        "D          0.73398        1.08786       -0.07503\n"
        "D          0.71239       -0.45698        0.82335\n"
        "D          0.71240       -0.45580       -0.97549\n"
        "H          2.99301        2.11762       -0.07478\n"
        "H          7.76531       -1.72634       -0.07591\n"
        "H          7.14864       -0.32182        0.81969\n"
        "H          7.14802       -0.32076       -0.96953\n"
        "H          2.86501       -5.02316       -0.05833\n"
        "H          4.40233       -5.15920        0.82837\n"
        "H          4.40017       -5.16929       -0.94780\n"
    )

    numbers = _read(text).numbers

    ref_numbers = torch.tensor(
        [
            6, 7, 6, 7, 6, 6, 6, 8, 7, 6, 8, 7, 6, 6,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        ]
    )  # fmt: skip
    assert numbers.shape == (24,)
    assert (numbers == ref_numbers).all()


def test_valid3_xyz_lowercase_and_extra_column() -> None:
    """mctc-lib's ``test_valid3_xyz``: lowercase element symbols, a single
    ``#`` comment line, and a trailing (partial charge) column that must be
    ignored rather than causing a parse error."""
    text = (
        "24\n"
        "#\n"
        "c  1.07317  0.04885 -0.07573 -0.05445590\n"
        "n  2.51365  0.01256 -0.07580 -0.00457526\n"
        "c  3.35199  1.09592 -0.07533  0.08391889\n"
        "n  4.61898  0.73028 -0.07549 -0.27870751\n"
        "c  4.57907 -0.63144 -0.07531  0.11914924\n"
        "c  3.30131 -1.10256 -0.07524 -0.02621044\n"
        "c  2.98068 -2.48687 -0.07377  0.26115960\n"
        "o  1.82530 -2.90038 -0.07577 -0.44071824\n"
        "n  4.11440 -3.30433 -0.06936 -0.10804747\n"
        "c  5.45174 -2.85618 -0.07235  0.30411699\n"
        "o  6.38934 -3.65965 -0.07232 -0.44083760\n"
        "n  5.66240 -1.47682 -0.07487 -0.07457706\n"
        "c  7.00947 -0.93648 -0.07524 -0.04790859\n"
        "c  3.92063 -4.74093 -0.06158 -0.03738239\n"
        "h  0.73398  1.08786 -0.07503  0.06457802\n"
        "h  0.71239 -0.45698  0.82335  0.08293905\n"
        "h  0.71240 -0.45580 -0.97549  0.08296802\n"
        "h  2.99301  2.11762 -0.07478  0.05698136\n"
        "h  7.76531 -1.72634 -0.07591  0.09025556\n"
        "h  7.14864 -0.32182  0.81969  0.07152988\n"
        "h  7.14802 -0.32076 -0.96953  0.07159003\n"
        "h  2.86501 -5.02316 -0.05833  0.08590674\n"
        "h  4.40233 -5.15920  0.82837  0.06906357\n"
        "h  4.40017 -5.16929 -0.94780  0.06926350\n"
    )

    numbers = _read(text).numbers

    ref_numbers = torch.tensor(
        [
            6, 7, 6, 7, 6, 6, 6, 8, 7, 6, 8, 7, 6, 6,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        ]
    )  # fmt: skip
    assert numbers.shape == (24,)
    assert (numbers == ref_numbers).all()


def test_valid4_xyz_trajectory() -> None:
    """mctc-lib's ``test_valid4_xyz``: three literal, hand-written water
    frames read back-to-back from the same stream (unlike
    ``test_write_and_read_batch``, which round-trips through ``write_xyz``
    and so cannot catch a symmetric symbol<->number bug)."""
    text = (
        "3\n"
        "WATER27, H2O\n"
        "O     1.1847029    1.1150792   -0.0344641 \n"
        "H     0.4939088    0.9563767    0.6340089 \n"
        "H     2.0242676    1.0811246    0.4301417 \n"
        "3\n"
        "WATER27, H2O\n"
        "O    -1.1469443    0.0697649    1.1470196 \n"
        "H    -1.2798308   -0.5232169    1.8902833 \n"
        "H    -1.0641398   -0.4956693    0.3569250 \n"
        "3\n"
        "WATER27, H2O\n"
        "O    -0.1633508   -1.0289346   -1.2401808 \n"
        "H     0.4914771   -0.3248733   -1.0784838 \n"
        "H    -0.5400907   -0.8496512   -2.1052499 \n"
    )

    structure = _read(text)

    numbers, positions = structure.numbers, structure.positions

    ref_numbers = torch.tensor([8, 1, 1])
    ref_positions = (
        torch.tensor(
            [
                [
                    [1.1847029, 1.1150792, -0.0344641],
                    [0.4939088, 0.9563767, 0.6340089],
                    [2.0242676, 1.0811246, 0.4301417],
                ],
                [
                    [-1.1469443, 0.0697649, 1.1470196],
                    [-1.2798308, -0.5232169, 1.8902833],
                    [-1.0641398, -0.4956693, 0.3569250],
                ],
                [
                    [-0.1633508, -1.0289346, -1.2401808],
                    [0.4914771, -0.3248733, -1.0784838],
                    [-0.5400907, -0.8496512, -2.1052499],
                ],
            ]
        )
        * length.AA2AU
    )

    assert numbers.shape == (3, 3)
    assert (numbers == ref_numbers).all()
    assert pytest.approx(ref_positions.cpu()) == positions.cpu()


def test_valid5_xyz_numeric_symbol() -> None:
    """mctc-lib's ``test_valid5_xyz``: atomic numbers used in place of
    element symbols in the first column."""
    text = (
        "3\n"
        "WATER27, H2O\n"
        "8     1.1847029    1.1150792   -0.0344641 \n"
        "1     0.4939088    0.9563767    0.6340089 \n"
        "1     2.0242676    1.0811246    0.4301417 \n"
    )

    numbers = _read(text).numbers

    ref_numbers = torch.tensor([8, 1, 1])
    assert numbers.shape == (3,)
    assert (numbers == ref_numbers).all()


def test_invalid1_xyz_atom_count_mismatch() -> None:
    """mctc-lib's ``test_invalid1_xyz``: declared atom count (9) exceeds the
    number of atom lines actually present (6)."""
    text = (
        "9\n"
        "WATER27, (H2O)3\n"
        "O     1.1847029    1.1150792   -0.0344641 \n"
        "H     0.4939088    0.9563767    0.6340089 \n"
        "H     2.0242676    1.0811246    0.4301417 \n"
        "O    -1.1469443    0.0697649    1.1470196 \n"
        "H    -1.2798308   -0.5232169    1.8902833 \n"
        "H    -1.0641398   -0.4956693    0.3569250 \n"
    )

    with pytest.raises(FormatErrorXYZ):
        read.xyz.read_xyz_fileobj(io.StringIO(text))


def test_invalid2_xyz_blank_first_line() -> None:
    """mctc-lib's ``test_invalid2_xyz``: the very first (natoms) line is
    blank. Unlike a *trailing* blank line (a legitimate end-of-trajectory
    marker for this reader), a blank line before any frame was read means no
    usable data was found at all."""
    text = (
        "\n"
        "WATER27, (H2O)3\n"
        "O     1.1847029    1.1150792   -0.0344641 \n"
        "H     0.4939088    0.9563767    0.6340089 \n"
        "H     2.0242676    1.0811246    0.4301417 \n"
        "O    -1.1469443    0.0697649    1.1470196 \n"
        "H    -1.2798308   -0.5232169    1.8902833 \n"
        "H    -1.0641398   -0.4956693    0.3569250 \n"
        "O    -0.1633508   -1.0289346   -1.2401808 \n"
        "H     0.4914771   -0.3248733   -1.0784838 \n"
        "H    -0.5400907   -0.8496512   -2.1052499 \n"
    )

    with pytest.raises(EmptyFileError):
        read.xyz.read_xyz_fileobj(io.StringIO(text))


def test_invalid3_xyz_only_natoms_line() -> None:
    """mctc-lib's ``test_invalid3_xyz``: the file ends right after the
    natoms line, with no comment or atom lines at all."""
    with pytest.raises(FormatErrorXYZ):
        read.xyz.read_xyz_fileobj(io.StringIO("120\n"))


def test_invalid4_xyz_negative_natoms() -> None:
    """mctc-lib's ``test_invalid4_xyz``: a negative atom count."""
    text = (
        "-3\n"
        "H2O\n"
        "O     1.1847029    1.1150792   -0.0344641 \n"
        "H     0.4939088    0.9563767    0.6340089 \n"
        "H     2.0242676    1.0811246    0.4301417 \n"
    )

    with pytest.raises(FormatErrorXYZ):
        read.xyz.read_xyz_fileobj(io.StringIO(text))


def test_invalid5_xyz_unknown_symbol() -> None:
    """mctc-lib's ``test_invalid5_xyz``: element symbols corrupted with a
    ``****`` prefix that cannot be mapped to an atomic number."""
    text = (
        "3\n"
        "H2O\n"
        "****o   1.1847029    1.1150792   -0.0344641 \n"
        "****h   0.4939088    0.9563767    0.6340089 \n"
        "****h   2.0242676    1.0811246    0.4301417 \n"
    )

    with pytest.raises(FormatErrorXYZ):
        read.xyz.read_xyz_fileobj(io.StringIO(text))


def test_invalid6_xyz_reversed_columns() -> None:
    """mctc-lib's ``test_invalid6_xyz``: coordinates and element symbol
    swapped (``x y z symbol`` instead of ``symbol x y z``)."""
    text = (
        "3\n"
        "H2O\n"
        "1.1847029    1.1150792   -0.0344641      O\n"
        "0.4939088    0.9563767    0.6340089      H\n"
        "2.0242676    1.0811246    0.4301417      H\n"
    )

    # Not wrapped in `FormatErrorXYZ`: this fails inside `numpy.loadtxt`
    # trying to parse "O"/"H" as the first column's float, since the first
    # four columns are always assumed to be `symbol x y z`.
    with pytest.raises(ValueError):
        read.xyz.read_xyz_fileobj(io.StringIO(text))


def test_invalid7_xyz_non_numeric_natoms() -> None:
    """mctc-lib's ``test_invalid7_xyz``: the natoms line is not a number."""
    text = (
        "nine\n"
        "WATER27, (H2O)3\n"
        "O     1.1847029    1.1150792   -0.0344641 \n"
        "H     0.4939088    0.9563767    0.6340089 \n"
        "H     2.0242676    1.0811246    0.4301417 \n"
    )

    with pytest.raises(FormatErrorXYZ):
        read.xyz.read_xyz_fileobj(io.StringIO(text))


##############################################################################
# Ported from mctc-lib's test/test_read_xyz.f90 (Extended XYZ cases).
##############################################################################


def _extxyz_single(header: str, atom: str) -> io.StringIO:
    """Wrap one atom line and its Extended XYZ header into a 1-atom file,
    mirroring mctc-lib's ``check_invalid_extxyz`` helper."""
    return io.StringIO(f"1\n{header}\n{atom}\n")


def test_valid6_extxyz_forces_column_before_species() -> None:
    """mctc-lib's ``valid6-extxyz``: a ``forces`` property occupies the
    first 3 columns before ``species``/``pos``, exercising that an
    unrecognized property is correctly skipped over rather than
    misparsed; ``pbc`` overrides the all-``True`` periodicity that
    ``Lattice`` alone would otherwise imply."""
    text = (
        "2\n"
        'Lattice="5.0 0.0 0.0 0.0 6.0 0.0 0.0 0.0 7.0" '
        "Properties=forces:R:3:species:S:1:pos:R:3 energy=-1.2 "
        'pbc="T F T" comment="periodic test"\n'
        "0.1 0.2 0.3 H 1.0 2.0 3.0\n"
        "0.4 0.5 0.6 O 4.0 5.0 6.0\n"
    )

    structure = _read(text)

    numbers, positions = structure.numbers, structure.positions

    lattice, periodic = structure.lattice, structure.periodic
    assert lattice is not None and periodic is not None

    ref_lattice = torch.diag(torch.tensor([5.0, 6.0, 7.0])) * length.AA2AU
    ref_positions = (
        torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]) * length.AA2AU
    )

    assert numbers.shape == (2,)
    assert (numbers == torch.tensor([1, 8])).all()
    assert periodic.dtype == torch.bool
    assert (periodic == torch.tensor([True, False, True])).all()
    assert pytest.approx(ref_lattice.cpu()) == lattice.cpu()
    assert pytest.approx(ref_positions.cpu()) == positions.cpu()


def test_valid7_extxyz_z_column_bracket_pbc_all_false() -> None:
    """mctc-lib's ``valid7-extxyz``: no ``species`` column, atomic numbers
    given via a ``Z`` property instead; ``pbc=[F,F,F]`` (bracketed, no
    quotes) means the file is not periodic, so the structure has no
    lattice (mirrors mctc-lib's own ``count(struc%periodic) == 0``)."""
    text = (
        "2\n"
        "Properties=pos:R:3:Z:I:1 pbc=[F,F,F]\n"
        "1.0 2.0 3.0 1\n"
        "4.0 5.0 6.0 8\n"
    )

    structure = _read(text)

    numbers, positions = structure.numbers, structure.positions

    ref_positions = (
        torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]) * length.AA2AU
    )
    assert numbers.shape == (2,)
    assert (numbers == torch.tensor([1, 8])).all()
    assert pytest.approx(ref_positions.cpu()) == positions.cpu()


def test_invalid1_extxyz_bad_lattice_value_count() -> None:
    """mctc-lib's ``invalid1-extxyz``: ``Lattice`` has 2 values, neither
    3 nor 9."""
    with pytest.raises(FormatErrorXYZ):
        read.xyz.read_xyz_fileobj(
            _extxyz_single(
                'Properties=species:S:1:pos:R:3 Lattice="1.0 0.0"',
                "H 0.0 0.0 0.0",
            )
        )


def test_extxyz_periodic_pbc_without_lattice_raises() -> None:
    """A periodic axis without its lattice vector cannot be evaluated, so
    this is rejected (mctc-lib keeps a zero cell instead)."""
    with pytest.raises(FormatErrorXYZ, match="no 'Lattice'"):
        read.xyz.read_xyz_fileobj(
            _extxyz_single(
                'Properties=species:S:1:pos:R:3 pbc="T T F"',
                "H 0.5 0.5 0.5",
            )
        )


def test_extxyz_non_periodic_pbc_without_lattice_is_a_molecule() -> None:
    """An all-false ``pbc`` without a ``Lattice``, as ASE writes for a
    molecule, reads as a plain molecule."""
    structure = read.xyz.read_xyz_fileobj(
        _extxyz_single(
            'Properties=species:S:1:pos:R:3 pbc="F F F"',
            "H 0.5 0.5 0.5",
        )
    )

    assert structure.lattice is None and structure.periodic is None


def test_invalid2_extxyz_bad_pbc_value_count() -> None:
    """mctc-lib's ``invalid2-extxyz``: ``pbc`` has 2 values, not 3."""
    with pytest.raises(FormatErrorXYZ):
        read.xyz.read_xyz_fileobj(
            _extxyz_single(
                'Properties=species:S:1:pos:R:3 pbc="T F"',
                "H 0.0 0.0 0.0",
            )
        )


def test_invalid3_extxyz_unterminated_comment_quote() -> None:
    """mctc-lib's ``invalid3-extxyz``: an unterminated quote for a
    ``comment`` key appearing after an otherwise well-formed
    ``Properties`` -- the header is malformed overall (mctc-lib re-scans
    the whole line for each key it looks up), not just its unused comment
    value."""
    with pytest.raises(FormatErrorXYZ):
        read.xyz.read_xyz_fileobj(
            _extxyz_single(
                'Properties=species:S:1:pos:R:3 comment="unterminated',
                "H 0.0 0.0 0.0",
            )
        )


def test_invalid4_extxyz_properties_wrong_field_count() -> None:
    """mctc-lib's ``invalid4-extxyz``: ``Properties`` entries must come in
    groups of 3 (``name:kind:count``); ``pos:R`` is missing its count."""
    with pytest.raises(FormatErrorXYZ):
        read.xyz.read_xyz_fileobj(
            _extxyz_single("Properties=species:S:1:pos:R", "H 0.0 0.0 0.0")
        )


def test_invalid5_extxyz_species_wrong_kind() -> None:
    """mctc-lib's ``invalid5-extxyz``: ``species`` must be kind ``S``."""
    with pytest.raises(FormatErrorXYZ):
        read.xyz.read_xyz_fileobj(
            _extxyz_single("Properties=species:R:1:pos:R:3", "H 0.0 0.0 0.0")
        )


def test_invalid_extxyz_properties_count_not_integer() -> None:
    """A ``Properties`` entry's count field must parse as an integer."""
    with pytest.raises(FormatErrorXYZ):
        read.xyz.read_xyz_fileobj(
            _extxyz_single("Properties=species:S:x:pos:R:3", "H 0.0 0.0 0.0")
        )


def test_invalid_extxyz_properties_count_zero() -> None:
    """A ``Properties`` entry's count must be at least 1."""
    with pytest.raises(FormatErrorXYZ):
        read.xyz.read_xyz_fileobj(
            _extxyz_single("Properties=species:S:0:pos:R:3", "H 0.0 0.0 0.0")
        )


def test_invalid_extxyz_z_wrong_kind() -> None:
    """``Z`` must be kind ``I`` (integer), mirroring ``species`` needing
    kind ``S`` (see the wrong-kind test above)."""
    with pytest.raises(FormatErrorXYZ):
        read.xyz.read_xyz_fileobj(
            _extxyz_single("Properties=Z:R:1:pos:R:3", "8 0.0 0.0 0.0")
        )


def test_invalid6_extxyz_pos_wrong_count() -> None:
    """mctc-lib's ``invalid6-extxyz``: ``pos`` must have count 3."""
    with pytest.raises(FormatErrorXYZ):
        read.xyz.read_xyz_fileobj(
            _extxyz_single("Properties=species:S:1:pos:R:2", "H 0.0 0.0")
        )


def test_invalid7_extxyz_missing_species_and_z() -> None:
    """mctc-lib's ``invalid7-extxyz``: neither ``species`` nor ``Z`` is
    given, so there is no way to resolve an element."""
    with pytest.raises(FormatErrorXYZ):
        read.xyz.read_xyz_fileobj(
            _extxyz_single("Properties=pos:R:3", "0.0 0.0 0.0")
        )


def test_invalid8_extxyz_missing_pos() -> None:
    """mctc-lib's ``invalid8-extxyz``: no ``pos`` entry at all."""
    with pytest.raises(FormatErrorXYZ):
        read.xyz.read_xyz_fileobj(_extxyz_single("Properties=species:S:1", "H"))


def test_invalid9_extxyz_atom_line_too_few_columns() -> None:
    """mctc-lib's ``invalid9-extxyz``: the atom line has fewer tokens than
    ``Properties`` declares."""
    with pytest.raises(FormatErrorXYZ):
        read.xyz.read_xyz_fileobj(
            _extxyz_single("Properties=species:S:1:pos:R:3", "H 0.0 0.0")
        )


def test_invalid10_extxyz_atom_line_too_many_columns() -> None:
    """mctc-lib's ``invalid10-extxyz``: the atom line has one extra token
    beyond what ``Properties`` declares."""
    with pytest.raises(FormatErrorXYZ):
        read.xyz.read_xyz_fileobj(
            _extxyz_single(
                "Properties=species:S:1:pos:R:3", "H 0.0 0.0 0.0 extra"
            )
        )


def test_invalid11_extxyz_non_numeric_position() -> None:
    """mctc-lib's ``invalid11-extxyz``: a ``pos`` column holds a
    non-numeric token."""
    with pytest.raises(FormatErrorXYZ):
        read.xyz.read_xyz_fileobj(
            _extxyz_single(
                "Properties=species:S:1:pos:R:3", "H invalid 0.0 0.0"
            )
        )


def test_invalid12_extxyz_non_positive_atomic_number() -> None:
    """mctc-lib's ``invalid12-extxyz``: a ``Z`` column of 0 cannot be
    mapped to an element (no ``species`` column to fall back on)."""
    with pytest.raises(FormatErrorXYZ):
        read.xyz.read_xyz_fileobj(
            _extxyz_single("Properties=Z:I:1:pos:R:3", "0 0.0 0.0 0.0")
        )


def test_invalid_extxyz_lattice_non_numeric_value() -> None:
    """A ``Lattice`` value must be 3 or 9 real numbers, not junk tokens."""
    with pytest.raises(FormatErrorXYZ):
        read.xyz.read_xyz_fileobj(
            _extxyz_single(
                'Properties=species:S:1:pos:R:3 Lattice="a b c d e f g h i"',
                "H 0.0 0.0 0.0",
            )
        )


def test_extxyz_lattice_three_values_is_diagonal() -> None:
    """``Lattice`` with exactly 3 values fills only the diagonal of an
    otherwise-zero lattice matrix, mirroring mctc-lib's ``parse_lattice``."""
    structure = _read(
        '1\nProperties=species:S:1:pos:R:3 Lattice="1.0 2.0 3.0"\n'
        "H 0.0 0.0 0.0\n"
    )
    numbers, positions = structure.numbers, structure.positions
    lattice, periodic = structure.lattice, structure.periodic
    assert lattice is not None

    assert (numbers == torch.tensor([1])).all()
    assert positions.shape == (1, 3)
    assert periodic is not None and periodic.all()
    ref_lattice = torch.diag(torch.tensor([1.0, 2.0, 3.0])) * length.AA2AU
    assert pytest.approx(ref_lattice.cpu()) == lattice.cpu()


def test_invalid_extxyz_pbc_invalid_token() -> None:
    """A ``pbc`` token must start with ``t``/``T`` or ``f``/``F``."""
    with pytest.raises(FormatErrorXYZ):
        read.xyz.read_xyz_fileobj(
            _extxyz_single(
                'Properties=species:S:1:pos:R:3 pbc="T Q T"',
                "H 0.0 0.0 0.0",
            )
        )


def test_invalid_extxyz_z_column_non_integer() -> None:
    """A ``Z`` column's value must parse as an integer."""
    with pytest.raises(FormatErrorXYZ):
        read.xyz.read_xyz_fileobj(
            _extxyz_single("Properties=Z:I:1:pos:R:3", "abc 0.0 0.0 0.0")
        )


def test_invalid_extxyz_species_unknown_symbol() -> None:
    """A ``species`` column that does not map to any known element."""
    with pytest.raises(FormatErrorXYZ):
        read.xyz.read_xyz_fileobj(
            _extxyz_single("Properties=species:S:1:pos:R:3", "Xx 0.0 0.0 0.0")
        )


def test_invalid_extxyz_truncated_atom_block() -> None:
    """Fewer atom lines than the header's atom count declares, for the
    Extended XYZ path (see ``test_invalid5_xyz_atom_count_mismatch`` for
    the plain-XYZ analogue)."""
    text = "2\nProperties=species:S:1:pos:R:3\nH 0.0 0.0 0.0\n"
    with pytest.raises(FormatErrorXYZ):
        read.xyz.read_xyz_fileobj(io.StringIO(text))


def test_extxyz_header_tolerates_extra_whitespace_and_escapes() -> None:
    """The header line scanner tolerates extra whitespace after ``=``,
    backslash-escaped characters inside quoted/bracketed values, and
    same-type nested brackets (all in unused keys here), rather than
    misparsing any of them as the end of the header."""
    text = (
        "1\n"
        "Properties=species:S:1:pos:R:3 "
        'comment=  "a\\b" escaped=[a "b\\]c" d] nested=[[a] b]\n'
        "H 0.0 0.0 0.0\n"
    )
    structure = _read(text)
    numbers, positions = structure.numbers, structure.positions

    assert (numbers == torch.tensor([1])).all()
    assert positions.shape == (1, 3)


def test_invalid_extxyz_header_empty_key() -> None:
    """A stray ``=value`` fragment with no key name before it ends the
    header scan early, same as an unterminated quote/bracket does."""
    with pytest.raises(FormatErrorXYZ):
        read.xyz.read_xyz_fileobj(
            _extxyz_single(
                "Properties=species:S:1:pos:R:3 =foo", "H 0.0 0.0 0.0"
            )
        )


def test_invalid_extxyz_header_unterminated_bracket() -> None:
    """A bracketed value that never closes ends the header scan early,
    same as an unterminated quote does (see
    ``test_invalid3_extxyz_unterminated_comment_quote``)."""
    with pytest.raises(FormatErrorXYZ):
        read.xyz.read_xyz_fileobj(
            _extxyz_single(
                "Properties=species:S:1:pos:R:3 pbc=[T F T",
                "H 0.0 0.0 0.0",
            )
        )


def test_extxyz_lattice_without_properties_is_ignored() -> None:
    """Discriminating case not in mctc-lib's own suite (per the advisor
    review that scoped this feature): Extended XYZ mode is switched on by
    ``Properties`` alone -- a comment line carrying ``Lattice=`` but no
    ``Properties`` key is just an ordinary (plain) comment, and the
    lattice is never even looked at."""
    text = (
        "1\n"
        'Lattice="5.0 0.0 0.0 0.0 5.0 0.0 0.0 0.0 5.0"\n'
        "H 0.0 0.0 0.0\n"
    )

    structure = _read(text)

    numbers, positions = structure.numbers, structure.positions

    assert numbers.shape == (1,)
    assert (numbers == torch.tensor([1])).all()
    assert positions.shape == (1, 3)


def test_extxyz_multiframe_periodic_unsupported() -> None:
    """Discriminating case not in mctc-lib's own suite: mctc-lib's
    ``read_xyz`` only ever handles a single frame, so there is no
    established convention for a per-frame-batched lattice/periodic
    shape. A multi-frame file where a frame declares periodicity must
    raise rather than silently drop or mis-batch it."""
    text = (
        "1\n"
        'Lattice="5.0 0.0 0.0 0.0 5.0 0.0 0.0 0.0 5.0" '
        "Properties=species:S:1:pos:R:3\n"
        "H 0.0 0.0 0.0\n"
        "1\n"
        'Lattice="5.0 0.0 0.0 0.0 5.0 0.0 0.0 0.0 5.0" '
        "Properties=species:S:1:pos:R:3\n"
        "H 0.0 0.0 0.0\n"
    )

    with pytest.raises(FormatErrorXYZ):
        read.xyz.read_xyz_fileobj(io.StringIO(text))
