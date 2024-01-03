# This file is part of tad-mctc.
#
# SPDX-Identifier: LGPL-3.0
# Copyright (C) 2023 Marvin Friede
#
# tad-mctc is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# tad_mctc is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with tad-mctc. If not, see <https://www.gnu.org/licenses/>.
"""
Test the XYZ file reader and writer.
"""
import tempfile
from pathlib import Path

import pytest
import torch

from tad_mctc.data.molecules import mols as samples
from tad_mctc.exceptions import EmptyFileError, FormatErrorTM
from tad_mctc.io import read, write
from tad_mctc.typing import DD, PathLike

from ..conftest import DEVICE

sample_list = ["LiH", "H2O"]


def test_read_fail() -> None:
    with pytest.raises(FileNotFoundError):
        read.read_turbomole_from_path("not found")


def test_read_fail_empty() -> None:
    # Create a temporary directory to save the file
    with tempfile.TemporaryDirectory() as tmpdirname:
        filepath = Path(tmpdirname) / "coord"
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("")

        with pytest.raises(FormatErrorTM):
            read.read_turbomole_from_path(filepath)


def test_read_fail_almost_empty() -> None:
    # Create a temporary directory to save the file
    with tempfile.TemporaryDirectory() as tmpdirname:
        filepath = Path(tmpdirname) / "coord"
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("$coord")

        with pytest.raises(EmptyFileError):
            read.read_turbomole_from_path(filepath)


def test_read_fail_format() -> None:
    # Create a temporary directory to save the file
    filepath = Path(__file__).parent.resolve() / "fail" / "coord"
    with pytest.raises(FormatErrorTM):
        read.read_turbomole_from_path(filepath)


def test_write_fail() -> None:
    sample = samples["H2O"]
    numbers = sample["numbers"]
    positions = sample["positions"]

    with tempfile.TemporaryDirectory() as tmpdirname:
        # create file
        filepath = Path(tmpdirname) / "already_exists"
        with filepath.open("w", encoding="utf-8") as f:
            f.write("")

        # try writing to just created file
        with pytest.raises(FileExistsError):
            write.write_turbomole_to_path(filepath, numbers, positions)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", sample_list)
@pytest.mark.parametrize("extra", [True, False])
def test_write_and_read(dtype: torch.dtype, name: str, extra: bool) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)

    # Create a temporary directory to save the file
    with tempfile.TemporaryDirectory() as tmpdirname:
        filepath = Path(tmpdirname) / f"{name}.coord"

        # Write to XYZ file
        write.write_turbomole_to_path(filepath, numbers, positions)

        # write something to beginning of file to test finding the coord section
        if extra is True:
            prepend_to_file(filepath, "something")

        # Read from XYZ file
        read_numbers, read_positions = read.read_turbomole_from_path(filepath, **dd)

    # Check if the read data matches the written data
    assert (read_numbers == numbers).all()
    assert pytest.approx(positions) == read_positions


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


@pytest.mark.parametrize("file", ["energy1", "energy2", "energy3", "energy4"])
def test_read_turbomole_energy_fail(file: str) -> None:
    p = Path(__file__).parent.resolve() / "fail" / file
    with pytest.raises(FormatErrorTM):
        read.read_turbomole_energy_from_path(p)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("file", ["energy"])
def test_read_turbomole_energy(dtype: torch.dtype, file: str) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    p = Path(__file__).parent.resolve() / "output" / file
    e = read.read_turbomole_energy_from_path(p, **dd)

    ref = torch.tensor(-291.856093690170, **dd)
    assert pytest.approx(ref) == e
