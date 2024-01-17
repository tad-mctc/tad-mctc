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
                read.read_chrg_from_path(filepath)
            else:
                read.read_uhf_from_path(filepath)


@pytest.mark.parametrize("name", [".CHRG", ".UHF"])
def test_read_fail_empty_not_int(name: str) -> None:
    with tempfile.TemporaryDirectory() as tmpdirname:
        filepath = Path(tmpdirname) / name
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("not an number")

        with pytest.raises(FormatError):
            if name == ".CHRG":
                read.read_chrg_from_path(filepath)
            else:
                read.read_spin_from_path(filepath)


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
            chrg = read.read_chrg_from_path(Path(tmpdirname), **dd)
        else:
            chrg = read.read_spin_from_path(Path(tmpdirname), **dd)

        assert chrg.dtype == dtype
        assert pytest.approx(torch.tensor(1, **dd)) == chrg


@pytest.mark.parametrize("dtype", [torch.long, torch.float, torch.double])
@pytest.mark.parametrize("name", [".CHRG", ".UHF"])
def test_read_chrg_default(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    with tempfile.TemporaryDirectory() as tmpdirname:
        filepath = Path(tmpdirname) / name

        if name == ".CHRG":
            chrg = read.read_chrg_from_path(filepath, **dd)
        else:
            chrg = read.read_spin_from_path(filepath, **dd)

        assert chrg.dtype == dtype
        assert pytest.approx(torch.tensor(0, **dd)) == chrg


@pytest.mark.parametrize("dtype", [torch.long, torch.float, torch.double])
@pytest.mark.parametrize("name", [".CHRG", ".UHF"])
def test_read_chrg_default_2(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    with tempfile.TemporaryDirectory() as tmpdirname:
        filepath = Path(tmpdirname)

        if name == ".CHRG":
            chrg = read.read_chrg_from_path(filepath, **dd)
        else:
            chrg = read.read_spin_from_path(filepath, **dd)

        assert chrg.dtype == dtype
        assert pytest.approx(torch.tensor(0, **dd)) == chrg


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
            chrg = read.read_chrg_from_path(filepath, **dd)
        else:
            chrg = read.read_spin_from_path(filepath, **dd)

        assert chrg.dtype == dtype
        assert pytest.approx(torch.tensor(0, **dd)) == chrg
