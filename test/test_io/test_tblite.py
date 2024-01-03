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
Test the check for deflating (padding clash).
"""
from pathlib import Path

import pytest
import torch

from tad_mctc.io import read
from tad_mctc.typing import DD

from ..conftest import DEVICE


@pytest.mark.parametrize("file", ["tblite.json"])
def test_read_tblite_json(file: str) -> None:
    p = Path(__file__).parent.resolve() / "output" / file
    with open(p, encoding="utf-8") as f:
        data = read.tblite._read_tblite_gfn(f)

    assert data["version"] == "0.2.1"
    assert data["energy"] == -0.8814248363751351
    assert data["energies"] == [
        -0.30036182118846183,
        -0.58106301518667340,
    ]
    assert data["gradient"] == [
        +4.8212532360532058e-18,
        -8.5642661603437023e-33,
        -1.2967437449579400e-02,
        -4.8212532360532058e-18,
        +8.5642661603437023e-33,
        +1.2967437449579402e-02,
    ]
    assert data["virial"] == [
        +0.0000000000000000e00,
        +0.0000000000000000e00,
        -7.2702928950083074e-18,
        +0.0000000000000000e00,
        +0.0000000000000000e00,
        +1.2914634508491054e-32,
        -7.2702928950083074e-18,
        +1.2914634508491054e-32,
        +3.9108946881752781e-02,
    ]


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("file", ["tblite.json"])
def test_read_tblite_engrad(dtype: torch.dtype, file: str) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    p = Path(__file__).parent.resolve() / "output" / file
    energies, gradients = read.read_tblite_engrad_from_path(p, **dd)

    ref_e = torch.tensor([-0.30036182118846183, -0.5810630151866734], **dd)
    ref_g = torch.tensor(
        [
            [
                +4.8212532360532058e-18,
                -8.5642661603437023e-33,
                -1.2967437449579400e-02,
            ],
            [
                -4.8212532360532058e-18,
                +8.5642661603437023e-33,
                +1.2967437449579402e-02,
            ],
        ],
        **dd
    )

    assert pytest.approx(ref_e) == energies
    assert pytest.approx(ref_g) == gradients
