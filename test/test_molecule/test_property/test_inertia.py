# This file is part of tad-multicharge.
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
Test the calculation of the inertia moment.
"""

from __future__ import annotations

import pytest
import torch

from tad_mctc.data.mass import ATOMIC
from tad_mctc.molecule import inertia_moment, mass_center
from tad_mctc.typing import DD

from ...conftest import DEVICE


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_single(dtype: torch.dtype) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    # H2 along z-axis
    numbers = torch.tensor([1, 1], device=DEVICE)
    masses = ATOMIC.to(**dd)[numbers]
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], **dd)

    # diagonal, with zeros for the z-axis (rotation axis), non-zero for x and y
    ref = torch.tensor(
        [
            [918.71, 0.0, 0.0],
            [0.0, 918.71, 0.0],
            [0.0, 0.0, 0.0],
        ],
        **dd,
    )

    im = inertia_moment(masses, positions)
    assert pytest.approx(ref, abs=1e-1) == im

    # precompute center of mass
    com = mass_center(masses, positions)
    pos = positions - com

    im = inertia_moment(masses, pos, pos_already_com=True)
    assert pytest.approx(ref, abs=1e-1) == im

    # no centering w.r.t. principal axes
    im = inertia_moment(masses, pos, center_pa=False)
    ref = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 918.71],
        ],
        **dd,
    )

    assert pytest.approx(ref, abs=1e-1) == im


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_batch(dtype: torch.dtype) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    # H2 along x- and z-axis
    numbers = torch.tensor(
        [
            [1, 1],
            [1, 1],
        ],
        device=DEVICE,
    )
    positions = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
        ],
        **dd,
    )

    masses = ATOMIC.to(**dd)[numbers]

    ref = torch.tensor(
        [
            [
                [918.71, 0.0, 0.0],
                [0.0, 918.71, 0.0],
                [0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0],
                [0.0, 918.71, 0.0],
                [0.0, 0.0, 918.71],
            ],
        ],
        **dd,
    )

    im = inertia_moment(masses, positions)
    assert pytest.approx(ref, abs=1e-1) == im

    # precompute center of mass
    com = mass_center(masses, positions)
    pos = positions - com.unsqueeze(-2)

    im = inertia_moment(masses, pos, pos_already_com=True)
    assert pytest.approx(ref, abs=1e-1) == im
