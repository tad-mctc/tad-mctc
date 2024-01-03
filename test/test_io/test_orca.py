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
Test the file reader for ORCA files.
"""
import tempfile
from pathlib import Path

import pytest
import torch

from tad_mctc.exceptions import EmptyFileError, FormatErrorORCA
from tad_mctc.io import read
from tad_mctc.typing import DD

from ..conftest import DEVICE


@pytest.mark.parametrize("file", ["orca.engrad", "orca2.engrad"])
def test_read_engrad_fail(file: str) -> None:
    p = Path(Path(__file__).parent.resolve(), "fail", file)
    with pytest.raises(FormatErrorORCA):
        read.read_orca_engrad_from_path(p)


def test_read_engrad_fail_empty() -> None:
    # Create a temporary directory to save the file
    with tempfile.TemporaryDirectory() as tmpdirname:
        filepath = Path(tmpdirname) / "orca.engrad"
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("")

        with open(filepath, encoding="utf-8") as f:
            with pytest.raises(EmptyFileError):
                read.read_orca_engrad(f)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("file", ["orca.engrad"])
def test_read_engrad(dtype: torch.dtype, file: str) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    p = Path(__file__).parent.resolve() / "output" / file
    e, g = read.read_orca_engrad_from_path(p, **dd)

    ref_e = torch.tensor(-17.271065945172, **dd)
    ref_g = torch.tensor(
        [
            [-3.2920000059277754e-09, 0.0, -0.006989939603954554],
            [-0.0014856194611638784, -0.0, +0.0035081561654806137],
            [+0.0014856194611638784, -0.0, +0.0035081561654806137],
        ],
        **dd
    )

    assert pytest.approx(ref_e) == e
    assert pytest.approx(ref_g) == g
