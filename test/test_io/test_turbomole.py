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

import tempfile
from pathlib import Path

import pytest
import torch

from tad_mctc.exceptions import EmptyFileError, FormatErrorTM
from tad_mctc.io import read, write
from tad_mctc.typing import DD, PathLike

from ..conftest import DEVICE
from ..utils import load_sample, resolve_structure

_SAMPLE_SOURCES: list[tuple[str, str]] = [
    ("mb16_43", "LiH"),
    ("heavy28", "h2o"),
]


def test_read_fail() -> None:
    with pytest.raises(FileNotFoundError):
        read.read_turbomole("not found")


def test_read_fail_empty() -> None:
    # Create a temporary directory to save the file
    with tempfile.TemporaryDirectory() as tmpdirname:
        filepath = Path(tmpdirname) / "coord"
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("")

        with pytest.raises(FormatErrorTM):
            read.read_turbomole(filepath)


def test_read_fail_almost_empty() -> None:
    # Create a temporary directory to save the file
    with tempfile.TemporaryDirectory() as tmpdirname:
        filepath = Path(tmpdirname) / "coord"
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("$coord")

        with pytest.raises(EmptyFileError):
            read.read_turbomole(filepath)


def test_read_fail_format() -> None:
    # Create a temporary directory to save the file
    filepath = Path(__file__).parent.resolve() / "fail" / "coord"
    with pytest.raises(FormatErrorTM):
        read.read_turbomole(filepath)


def test_write_fail() -> None:
    sample = resolve_structure("heavy28", "h2o")
    numbers = sample.numbers
    positions = sample.positions

    with tempfile.TemporaryDirectory() as tmpdirname:
        # create file
        filepath = Path(tmpdirname) / "already_exists"
        with filepath.open("w", encoding="utf-8") as f:
            f.write("")

        # try writing to just created file
        with pytest.raises(FileExistsError):
            write.write_turbomole(filepath, numbers, positions)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("collection,record", _SAMPLE_SOURCES)
@pytest.mark.parametrize("extra", [True, False])
def test_write_and_read(
    dtype: torch.dtype, collection: str, record: str, extra: bool
) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    numbers, positions = load_sample(collection, record, dd)

    # Create a temporary directory to save the file
    with tempfile.TemporaryDirectory() as tmpdirname:
        filepath = Path(tmpdirname) / f"{collection}_{record}.coord"

        # Write to XYZ file
        write.write_turbomole(filepath, numbers, positions)

        # write something to beginning of file to test finding the coord section
        if extra is True:
            prepend_to_file(filepath, "something")

        # Read from XYZ file
        read_numbers, read_positions = read.read_turbomole(  # type: ignore[misc]
            filepath, **dd
        )

    # Check if the read data matches the written data
    assert read_numbers.dtype == numbers.dtype
    assert read_numbers.shape == numbers.shape
    assert (read_numbers == numbers).all()
    assert pytest.approx(positions.cpu()) == read_positions.cpu()


def prepend_to_file(file_path: PathLike, text_to_prepend: str) -> None:
    """Prepends text to the beginning of a file.

    Parameters:
    file_path (str): Path to the file.
    text_to_prepend (str): Text to insert at the beginning of the file.
    """
    # Read the existing content of the file
    with open(file_path) as file:
        original_content = file.read()

    # Write the new text, followed by the original content
    with open(file_path, "w") as file:
        file.write(text_to_prepend + "\n" + original_content)


################################################################################
# ported from mctc-lib's test_read_turbomole.f90 (non-periodic cases)
################################################################################


def test_read_valid_angs_unit() -> None:
    """mctc-lib's own ``valid1-coord`` fixture: ``$coord angs`` unit suffix
    on a plain (non-periodic) molecule."""
    content = (
        "$coord angs\n"
        " 1.1847029  1.1150792 -0.0344641 O\n"
        " 0.4939088  0.9563767  0.6340089 H\n"
        " 2.0242676  1.0811246  0.4301417 H\n"
        "-1.1469443  0.0697649  1.1470196 O\n"
        "-1.2798308 -0.5232169  1.8902833 H\n"
        "-1.0641398 -0.4956693  0.3569250 H\n"
        "-0.1633508 -1.0289346 -1.2401808 O\n"
        " 0.4914771 -0.3248733 -1.0784838 H\n"
        "-0.5400907 -0.8496512 -2.1052499 H\n"
        "$end\n"
    )
    with tempfile.TemporaryDirectory() as tmpdirname:
        filepath = Path(tmpdirname) / "coord"
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        numbers, positions = read.read_turbomole(filepath)  # type: ignore[misc]

    assert positions.shape[0] == 9
    assert len(torch.unique(numbers)) == 2


def test_read_no_coord_section() -> None:
    """mctc-lib's own ``invalid1-coord`` fixture: a file containing only
    ``$end``, i.e. no ``$coord`` data group at all (distinct from an empty
    file or a bare ``$coord`` tag with no atoms)."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        filepath = Path(tmpdirname) / "coord"
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("$end\n")

        with pytest.raises(FormatErrorTM):
            read.read_turbomole(filepath)


def test_read_fail_bad_element_symbol() -> None:
    """mctc-lib's own ``invalid4-coord`` fixture: an unparsable element
    symbol."""
    content = (
        "$coord angs\n"
        "-1.1469443  0.0697649  1.1470196 --->o\n"
        "-1.2798308 -0.5232169  1.8902833 H\n"
        "-1.0641398 -0.4956693  0.3569250 H\n"
        "$end\n"
    )
    with tempfile.TemporaryDirectory() as tmpdirname:
        filepath = Path(tmpdirname) / "coord"
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        with pytest.raises(FormatErrorTM):
            read.read_turbomole(filepath)


def test_read_fail_bad_coordinate_value() -> None:
    """mctc-lib's own ``invalid8-coord`` fixture: a non-numeric coordinate
    value."""
    content = (
        "$coord angs\n"
        " 1.1847029  1.1150792 -0.0344641 O\n"
        " 0.4939088  0.9563767  0.6340089 H\n"
        " 2.0242676  1.0811246  0.4301417 H\n"
        "-1.1469443  abcd.efgh  1.1470196 O\n"
        "-1.2798308 -0.5232169  1.8902833 H\n"
        "-1.0641398 -0.4956693  0.3569250 H\n"
        "-0.1633508 -1.0289346 -1.2401808 O\n"
        " 0.4914771 -0.3248733 -1.0784838 H\n"
        "-0.5400907 -0.8496512 -2.1052499 H\n"
        "$end\n"
    )
    with tempfile.TemporaryDirectory() as tmpdirname:
        filepath = Path(tmpdirname) / "coord"
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        with pytest.raises(FormatErrorTM):
            read.read_turbomole(filepath)


def test_read_scientific_notation() -> None:
    """mctc-lib's own ``valid9-coord`` fixture: coordinates given in
    scientific notation.

    mctc-lib's version of this test also checks the total charge parsed
    from ``$eht charge=1``, but tad-mctc's ``read_turbomole`` does not
    parse ``$eht`` at all -- charge/uhf are read separately via
    ``read_chrg``/``read_uhf`` (see ``tad_mctc.io.read.dotfiles``) -- so
    only the coordinate parsing is exercised here.
    """
    content = (
        "$coord\n"
        "    4.82824919102333E-02    5.71831000079710E-02    1.73514614763116E-01      C\n"
        "    4.82824919102333E-02    5.71831000079710E-02    2.78568246476372E+00      N\n"
        "    2.46093310136750E+00    5.71831000079710E-02    3.59954953387915E+00      C\n"
        "    3.99138416000780E+00   -2.21116805417472E-01    1.58364683739854E+00      N\n"
        "    2.54075511539052E+00   -1.18599185608072E-01   -5.86344093538442E-01      C\n"
        "   -2.06104824371096E+00    8.28021114689117E-01    4.40357113204146E+00      C\n"
        "    6.72173545596011E+00    2.10496546922931E-01    1.72565972456309E+00      C\n"
        "    3.05878562448454E+00    7.09403031823937E-02    5.55721088395376E+00      H\n"
        "    3.36822820962351E+00   -2.07680855613880E-01   -2.46191575873710E+00      H\n"
        "   -1.68465267663933E+00    1.48551338123814E-01   -9.21486948343917E-01      H\n"
        "   -3.83682349412373E+00    3.78984491295393E-01    3.43261116458953E+00      H\n"
        "   -1.96215889726624E+00   -2.17412943024358E-01    6.19219651728748E+00      H\n"
        "   -1.85966017471395E+00    2.87036107386343E+00    4.74746341688781E+00      H\n"
        "    7.49947096948557E+00   -8.77758695396645E-01    3.31081834253025E+00      H\n"
        "    7.58490546886959E+00   -4.29156708916399E-01   -4.73754235690626E-02      H\n"
        "    7.00829346274163E+00    2.24769645216395E+00    2.03795579552532E+00      H\n"
        "$eht charge=1\n"
        "$end\n"
    )
    with tempfile.TemporaryDirectory() as tmpdirname:
        filepath = Path(tmpdirname) / "coord"
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        numbers, positions = read.read_turbomole(filepath)  # type: ignore[misc]

    assert positions.shape[0] == 16
    assert len(torch.unique(numbers)) == 3


################################################################################


@pytest.mark.parametrize("file", ["energy1", "energy2", "energy3", "energy4"])
def test_read_turbomole_energy_fail(file: str) -> None:
    p = Path(__file__).parent.resolve() / "fail" / file
    with pytest.raises(FormatErrorTM):
        read.read_turbomole_energy(p)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("file", ["energy"])
def test_read_turbomole_energy(dtype: torch.dtype, file: str) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    p = Path(__file__).parent.resolve() / "output" / file
    e = read.read_turbomole_energy(p, **dd)
    assert isinstance(e, torch.Tensor)

    ref = torch.tensor(-291.856093690170, **dd)
    assert pytest.approx(ref.cpu()) == e.cpu()
