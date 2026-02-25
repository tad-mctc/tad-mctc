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
    energies, gradients = read.read_tblite_engrad(p, **dd)

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

    assert pytest.approx(ref_e.cpu()) == energies.cpu()
    assert pytest.approx(ref_g.cpu()) == gradients.cpu()
