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

from tad_mctc.exceptions import EmptyFileError, FormatError
from tad_mctc.io import read
from tad_mctc.typing import DD

from ..conftest import DEVICE


@pytest.mark.parametrize("name", [".CHRG", ".UHF"])
def test_read_fail_empty(name: str) -> None:
    with tempfile.TemporaryDirectory() as tmpdirname:
        filepath = Path(tmpdirname) / name
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("")

        with pytest.raises(EmptyFileError):
            if name == ".CHRG":
                read.read_chrg(filepath)
            else:
                read.read_uhf(filepath)


@pytest.mark.parametrize("name", [".CHRG", ".UHF"])
def test_read_fail_empty_not_int(name: str) -> None:
    with tempfile.TemporaryDirectory() as tmpdirname:
        filepath = Path(tmpdirname) / name
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("not an number")

        with pytest.raises(FormatError):
            if name == ".CHRG":
                read.read_chrg(filepath)
            else:
                read.read_spin(filepath)


@pytest.mark.parametrize("dtype", [torch.long, torch.float, torch.double])
@pytest.mark.parametrize("name", [".CHRG", ".UHF"])
def test_read_chrg(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    with tempfile.TemporaryDirectory() as tmpdirname:
        filepath = Path(tmpdirname) / name
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("1")

        # also works by just giving the parent directory
        if name == ".CHRG":
            chrg = read.read_chrg(Path(tmpdirname), **dd)
        else:
            chrg = read.read_spin(Path(tmpdirname), **dd)

        assert chrg.dtype == dtype
        assert pytest.approx(1) == chrg.cpu()


@pytest.mark.parametrize("dtype", [torch.long, torch.float, torch.double])
@pytest.mark.parametrize("name", [".CHRG", ".UHF"])
def test_read_chrg_default(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    with tempfile.TemporaryDirectory() as tmpdirname:
        filepath = Path(tmpdirname) / name

        if name == ".CHRG":
            chrg = read.read_chrg(filepath, **dd)
        else:
            chrg = read.read_spin(filepath, **dd)

        assert chrg.dtype == dtype
        assert pytest.approx(0) == chrg.cpu()


@pytest.mark.parametrize("dtype", [torch.long, torch.float, torch.double])
@pytest.mark.parametrize("name", [".CHRG", ".UHF"])
def test_read_chrg_default_2(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    with tempfile.TemporaryDirectory() as tmpdirname:
        filepath = Path(tmpdirname)

        if name == ".CHRG":
            chrg = read.read_chrg(filepath, **dd)
        else:
            chrg = read.read_spin(filepath, **dd)

        assert chrg.dtype == dtype
        assert pytest.approx(0) == chrg.cpu()


@pytest.mark.parametrize("dtype", [torch.long, torch.float, torch.double])
@pytest.mark.parametrize("name", [".CHRG", ".UHF"])
def test_read_chrg_default_3(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    with tempfile.TemporaryDirectory() as tmpdirname:
        filepath = Path(tmpdirname) / "mol.xyz"
        with open(filepath, mode="w", encoding="utf-8") as f:
            f.write("test")

        # also works by just giving the parent directory
        if name == ".CHRG":
            chrg = read.read_chrg(filepath, **dd)
        else:
            chrg = read.read_spin(filepath, **dd)

        assert chrg.dtype == dtype
        assert pytest.approx(0) == chrg.cpu()
