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
Test the general file reader.
"""
from pathlib import Path

import pytest
import torch

from tad_mctc.io import read
from tad_mctc.typing import DD

from ..conftest import DEVICE


@pytest.mark.parametrize("file", ["mol.mol", "mol.ein", "POSCAR"])
def test_fail(file: str) -> None:
    p = Path(__file__).parent.resolve() / "files" / file
    with pytest.raises(NotImplementedError):
        read.read_from_path(p)


def test_fail_unknown() -> None:
    p = Path(__file__).parent.resolve() / "files" / "mol.xyz"
    with pytest.raises(ValueError):
        read.read_from_path(p, ftype="something")


def test_fail_notfound() -> None:
    p = Path(__file__).parent.resolve() / "files" / "notfound"
    with pytest.raises(FileNotFoundError):
        read.read_from_path(p)


################################################################################


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("file", ["mol.xyz", "mol.json", "coord", "qm9.xyz"])
def test_types(dtype: torch.dtype, file: str) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    p = Path(__file__).parent.resolve() / "files" / file

    ref_numbers = torch.tensor([8, 1, 1], device=DEVICE)
    ref_positions = torch.tensor(
        [
            [+0.00000000000000, +0.00000000000000, -0.74288549752983],
            [-1.43472674945442, +0.00000000000000, +0.37144274876492],
            [+1.43472674945442, +0.00000000000000, +0.37144274876492],
        ],
        **dd
    )

    ftype = None if "qm9" not in file else "qm9"
    numbers, positions = read.read_from_path(p, ftype=ftype, **dd)

    assert (ref_numbers == numbers).all()
    assert pytest.approx(ref_positions) == positions
