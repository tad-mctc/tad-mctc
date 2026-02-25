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
        read.read(p)


def test_fail_unknown() -> None:
    p = Path(__file__).parent.resolve() / "files" / "mol.xyz"
    with pytest.raises(ValueError):
        read.read(p, ftype="something")


def test_fail_notfound() -> None:
    p = Path(__file__).parent.resolve() / "files" / "notfound"
    with pytest.raises(FileNotFoundError):
        read.read(p)


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
    numbers, positions = read.read(p, ftype=ftype, **dd)

    assert (ref_numbers == numbers).all()
    assert pytest.approx(ref_positions.cpu()) == positions.cpu()
