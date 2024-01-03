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

from tad_mctc.batch import pack
from tad_mctc.data.molecules import mols as samples
from tad_mctc.io import read, write
from tad_mctc.typing import DD

from ..conftest import DEVICE

sample_list = ["H2O"]


def test_read_fail() -> None:
    with pytest.raises(FileNotFoundError):
        read.read_xyz_from_path("not found")


def test_write_fail() -> None:
    sample = samples["H2O"]
    numbers = sample["numbers"]
    positions = sample["positions"]

    with pytest.raises(FileExistsError):
        with tempfile.TemporaryDirectory() as tmpdirname:
            # create file
            filepath = Path(tmpdirname) / "already_exists"
            with filepath.open("w", encoding="utf-8") as f:
                f.write("")

            # try writing to just created file
            write.write_xyz_to_path(filepath, numbers, positions)


def test_write_comment_fail() -> None:
    sample = samples["H2O"]
    numbers = sample["numbers"]
    positions = sample["positions"]

    comment = "Comment \n with \n linebreaks \n"

    with pytest.raises(ValueError):
        with tempfile.TemporaryDirectory() as tmpdirname:
            fp = Path(tmpdirname) / "already_exists"
            write.write_xyz_to_path(fp, numbers, positions, comment=comment)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_write_and_read(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)

    # Create a temporary directory to save the file
    with tempfile.TemporaryDirectory() as tmpdirname:
        filepath = Path(tmpdirname) / f"{name}.xyz"

        # Write to XYZ file
        write.write_xyz_to_path(filepath, numbers, positions)

        # Read from XYZ file
        read_numbers, read_positions = read.read_xyz_from_path(filepath, **dd)

    # Check if the read data matches the written data
    assert (read_numbers == numbers).all()
    assert pytest.approx(positions) == read_positions


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", sample_list)
@pytest.mark.parametrize("name2", sample_list)
def test_write_and_read_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    sample1 = samples[name1]
    numbers1 = sample1["numbers"].to(DEVICE)
    positions1 = sample1["positions"].to(**dd)

    sample2 = samples[name2]
    numbers2 = sample2["numbers"].to(DEVICE)
    positions2 = sample2["positions"].to(**dd)

    numbers_batch = pack((numbers1, numbers2))
    positions_batch = pack((positions1, positions2))

    # Create a temporary directory to save the file
    with tempfile.TemporaryDirectory() as tmpdirname:
        filepath = Path(tmpdirname) / "combined.xyz"

        # Write first structure to XYZ file
        write.write_xyz_to_path(filepath, numbers1, positions1, mode="w")

        # Append second structure to the same XYZ file
        write.write_xyz_to_path(filepath, numbers2, positions2, mode="a")

        # Read from XYZ file
        read_numbers, read_positions = read.read_xyz_from_path(filepath, **dd)

    # Check if the read data matches the written data
    assert (read_numbers == numbers_batch).all()
    assert pytest.approx(positions_batch) == read_positions
