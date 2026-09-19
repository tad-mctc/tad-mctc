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
Test the general file writer.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from tad_mctc.io import read, write
from tad_mctc.typing import DD

from ..conftest import DEVICE
from ..utils import load_sample

_SAMPLE_SOURCES: list[tuple[str, str]] = [("heavy28", "h2o")]


@pytest.mark.parametrize("file", ["mol.mol", "mol.ein", "POSCAR"])
def test_fail(file: str) -> None:
    numbers = torch.tensor([8, 1, 1], device=DEVICE)
    positions = torch.zeros(3, 3, device=DEVICE)

    p = Path(__file__).parent.resolve() / "output" / file
    with pytest.raises(NotImplementedError):
        write.write(p, numbers, positions)


def test_fail_already_exists() -> None:
    numbers = torch.tensor([8, 1, 1], device=DEVICE)
    positions = torch.zeros(3, 3, device=DEVICE)

    p = Path(__file__).parent.resolve() / "files" / "mol.xyz"
    with pytest.raises(FileExistsError):
        write.write(p, numbers, positions, overwrite=False)


def test_fail_ftype() -> None:
    numbers = torch.tensor([8, 1, 1], device=DEVICE)
    positions = torch.zeros(3, 3, device=DEVICE)

    p = Path(__file__).parent.resolve() / "output" / "notfound"
    with pytest.raises(ValueError):
        write.write(p, numbers, positions)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("collection,record", _SAMPLE_SOURCES)
@pytest.mark.parametrize("fname", ["mol.xyz", "coord"])
def test_write_and_read(
    dtype: torch.dtype, collection: str, record: str, fname: str
) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    numbers, positions = load_sample(collection, record, dd)

    # Create a temporary directory to save the file
    with tempfile.TemporaryDirectory() as tmpdirname:
        filepath = Path(tmpdirname) / fname

        write.write(filepath, numbers, positions)
        read_numbers, read_positions = read.read(  # type: ignore[misc]
            filepath, **dd
        )

    # Check if the read data matches the written data
    assert read_numbers.dtype == numbers.dtype
    assert read_numbers.shape == numbers.shape
    assert (read_numbers == numbers).all()
    assert pytest.approx(positions.cpu()) == read_positions.cpu()
