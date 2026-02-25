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
Test the JSON/QCSchema file reader.
"""

import tempfile
from pathlib import Path

import pytest
import torch

from tad_mctc.io import read
from tad_mctc.typing import DD

from ..conftest import DEVICE


def test_read_fail() -> None:
    with pytest.raises(FileNotFoundError):
        read.read_xyz_qm9("not found")


def test_read_fail_empty() -> None:
    # Create a temporary directory to save the file
    with tempfile.TemporaryDirectory() as tmpdirname:
        filepath = Path(tmpdirname) / "qm9.xyz"
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("")

        with pytest.raises(ValueError):
            read.read_xyz_qm9(filepath)


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
    filepath = Path(__file__).parent / "files" / "qm9.xyz"
    with open(filepath, encoding="utf-8") as fp:
        read_numbers, read_positions = read.xyz.read_xyz_qm9_fileobj(fp, **dd)

    # Check if the read data matches the written data
    assert read_numbers.dtype == numbers.dtype
    assert read_numbers.shape == numbers.shape
    assert (numbers == read_numbers).all()
    assert pytest.approx(positions.cpu()) == read_positions.cpu()
