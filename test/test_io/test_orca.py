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
        read.read_orca_engrad(p)


def test_read_engrad_fail_empty() -> None:
    # Create a temporary directory to save the file
    with tempfile.TemporaryDirectory() as tmpdirname:
        filepath = Path(tmpdirname) / "orca.engrad"
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("")

        with open(filepath, encoding="utf-8") as f:
            with pytest.raises(EmptyFileError):
                read.orca.read_orca_engrad_fileobj(f)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("file", ["orca.engrad"])
def test_read_engrad(dtype: torch.dtype, file: str) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    p = Path(__file__).parent.resolve() / "output" / file
    e, g = read.read_orca_engrad(p, **dd)

    ref_e = torch.tensor(-17.271065945172, **dd)
    ref_g = torch.tensor(
        [
            [-3.2920000059277754e-09, 0.0, -0.006989939603954554],
            [-0.0014856194611638784, -0.0, +0.0035081561654806137],
            [+0.0014856194611638784, -0.0, +0.0035081561654806137],
        ],
        **dd
    )

    assert pytest.approx(ref_e.cpu()) == e.cpu()
    assert pytest.approx(ref_g.cpu()) == g.cpu()
