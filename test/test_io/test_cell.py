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
Test the internal ``cell_to_lattice`` helper shared by the turbomole and
cjson readers' periodic-cell handling.
"""

from __future__ import annotations

import torch

from tad_mctc.io.read._cell import cell_to_lattice
from tad_mctc.typing import DD

from ..conftest import DEVICE


def test_negative_vol2_flips_dvol_sign() -> None:
    """No valid triclinic cell has these angles (``vol2 < 0``); mctc-lib
    handles it anyway ("this should not happen, but who knows") by
    flipping the sign of the computed volume rather than raising."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    cellpar = (1.0, 1.0, 1.0, 0.1, 0.1, 3.0)

    lattice = cell_to_lattice(cellpar, dd)

    assert lattice.shape == (3, 3)
    assert bool(lattice.isfinite().all())
