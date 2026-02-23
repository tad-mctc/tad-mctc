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

import tempfile
from pathlib import Path

import pytest
import torch

from tad_mctc.batch import pack
from tad_mctc.data.molecules import mols as samples
from tad_mctc.exceptions import FormatErrorXYZ
from tad_mctc.io import read, write
from tad_mctc.typing import DD

from ..conftest import DEVICE

sample_list = ["H2O"]


def test_read_fail() -> None:
    with pytest.raises(FileNotFoundError):
        read.read_xyz("not found")

    p = Path(__file__).parent.resolve() / "fail" / "nat-no-digit.xyz"
    with pytest.raises(FormatErrorXYZ):
        read.read_xyz(p)


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
            write.write_xyz(filepath, numbers, positions)


def test_write_batch_fail() -> None:
    sample = samples["H2O"]

    with tempfile.TemporaryDirectory() as tmpdirname:
        filepath = Path(tmpdirname) / "dummy.xyz"

        # numbers batched, positions not
        numbers = pack((sample["numbers"], sample["numbers"]))
        positions = sample["positions"]
        with pytest.raises(ValueError):
            write.write_xyz(filepath, numbers, positions, overwrite=True)

        # positions batched, numbers not
        numbers = sample["numbers"]
        positions = pack((sample["positions"], sample["positions"]))
        with pytest.raises(ValueError):
            write.write_xyz(filepath, numbers, positions, overwrite=True)

        # too many dimensions
        numbers = torch.rand(2, 3, 4)
        positions = torch.rand(2, 3, 4, 3)
        with pytest.raises(ValueError):
            write.write_xyz(filepath, numbers, positions, overwrite=True)


def test_write_comment_fail() -> None:
    sample = samples["H2O"]
    numbers = sample["numbers"]
    positions = sample["positions"]

    comment = "Comment \n with \n linebreaks \n"

    with pytest.raises(ValueError):
        with tempfile.TemporaryDirectory() as tmpdirname:
            fp = Path(tmpdirname) / "already_exists"
            write.write_xyz(fp, numbers, positions, comment=comment)


##############################################################################


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", sample_list)
@pytest.mark.parametrize("batch_agnostic", [True, False])
def test_write_and_read(
    dtype: torch.dtype, name: str, batch_agnostic: bool
) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)

    # Create a temporary directory to save the file
    with tempfile.TemporaryDirectory() as tmpdirname:
        filepath = Path(tmpdirname) / f"{name}.xyz"

        # Write to XYZ file
        write.write_xyz(filepath, numbers, positions)

        # Read from XYZ file
        read_numbers, read_positions = read.read_xyz(
            filepath, batch_agnostic=batch_agnostic, **dd
        )

    if batch_agnostic is True:
        numbers = numbers.unsqueeze(0)
        positions = positions.unsqueeze(0)

    # Check if the read data matches the written data
    assert read_numbers.dtype == numbers.dtype
    assert read_numbers.shape == numbers.shape
    assert (read_numbers == numbers).all()
    assert pytest.approx(positions.cpu()) == read_positions.cpu()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", sample_list)
@pytest.mark.parametrize("name2", sample_list)
def test_write_and_read_batch(
    dtype: torch.dtype, name1: str, name2: str
) -> None:
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
        write.write_xyz(filepath, numbers1, positions1, mode="w")

        # Append second structure to the same XYZ file
        write.write_xyz(filepath, numbers2, positions2, mode="a")

        # Read from XYZ file
        read_numbers, read_positions = read.read_xyz(filepath, **dd)

    # Check if the read data matches the written data
    assert (read_numbers == numbers_batch).all()
    assert pytest.approx(positions_batch.cpu()) == read_positions.cpu()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", sample_list)
@pytest.mark.parametrize("name2", sample_list)
def test_write_batch_and_read_batch(
    dtype: torch.dtype, name1: str, name2: str
) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    sample1, sample2 = samples[name1], samples[name2]
    numbers = pack(
        (
            sample1["numbers"].to(DEVICE),
            sample2["numbers"].to(DEVICE),
        )
    )
    positions = pack(
        (
            sample1["positions"].to(**dd),
            sample2["positions"].to(**dd),
        )
    )

    # Create a temporary directory to save the file
    with tempfile.TemporaryDirectory() as tmpdirname:
        filepath = Path(tmpdirname) / "combined.xyz"

        # Write structure to XYZ file
        write.write_xyz(filepath, numbers, positions, mode="w")

        # Read from XYZ file
        read_numbers, read_positions = read.read_xyz(filepath, **dd)

    # Check if the read data matches the written data
    assert read_numbers.dtype == numbers.dtype
    assert read_numbers.shape == numbers.shape
    assert (read_numbers == numbers).all()
    assert pytest.approx(positions.cpu()) == read_positions.cpu()
