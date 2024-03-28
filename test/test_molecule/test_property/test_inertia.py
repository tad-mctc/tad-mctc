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
