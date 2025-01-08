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
Test the molecule container.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tad_mctc.molecule import Mol
from tad_mctc.typing import DD

from ...conftest import DEVICE


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_from_path(dtype: torch.dtype) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    here = Path(__file__).resolve().parent
    mol = Mol.from_path(here / "co2.xyz", **dd)

    assert mol.dtype == dtype
    assert (mol.numbers == torch.tensor([6, 8, 8], **dd)).all()
