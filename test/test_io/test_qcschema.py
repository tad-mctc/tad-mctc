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

from tad_mctc.io import read
from tad_mctc.typing import DD
from tad_mctc.units.length import AA2AU

from ..conftest import DEVICE

sample_list = ["LiH", "H2O"]


def test_read_fail() -> None:
    with pytest.raises(FileNotFoundError):
        read.read_qcschema_from_path("not found")


def test_read_fail_empty() -> None:
    # Create a temporary directory to save the file
    with tempfile.TemporaryDirectory() as tmpdirname:
        filepath = Path(tmpdirname) / "mol.json"
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("")

        with pytest.raises(ValueError):
            read.read_qcschema_from_path(filepath)


@pytest.mark.parametrize("file", ["mol1.json", "mol2.json", "mol3.json"])
def test_json(file: str) -> None:
    p = Path(__file__).parent.resolve() / "fail" / file
    with pytest.raises(KeyError):
        read.read_qcschema_from_path(p)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_read(dtype: torch.dtype) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    numbers = torch.tensor([8, 1, 1], device=DEVICE)
    positions = torch.tensor(
        [
            [+0.00000000000000, +0.00000000000000, -0.74288549752983],
            [-1.43472674945442, +0.00000000000000, +0.37144274876492],
            [+1.43472674945442, +0.00000000000000, +0.37144274876492],
        ],
        **dd
    )

    # Create a temporary directory to save the file
    filepath = Path(__file__).parent / "files" / "mol.json"
    with open(filepath, encoding="utf-8") as fp:
        read_numbers, read_positions = read.read_qcschema(fp, **dd)

    # Check if the read data matches the written data
    assert (numbers == read_numbers).all()
    assert pytest.approx(positions) == read_positions
