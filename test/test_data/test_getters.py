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
Test getters for atomic data.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from tad_mctc.data.getters import (
    get_atomic_masses,
    get_ecore,
    get_hardness,
    get_vdw_pairwise,
    get_zvalence,
)
from tad_mctc.units import GMOL2AU


@pytest.fixture
def atomic_numbers():
    return torch.tensor([1, 2, 3], dtype=torch.long)


@pytest.fixture
def mock_mass_tensor():
    return torch.tensor([1.0, 4.0, 7.0, 10.0], dtype=torch.float)


@pytest.fixture
def mock_zeff_tensor():
    return torch.tensor([1, 2, 3, 4], dtype=torch.long)


@pytest.fixture
def mock_hardness_tensor():
    return torch.tensor(
        [
            0.00000000,
            0.4725929,
            0.92203391,
            0.17452888,
        ],
        dtype=torch.float,
    )


@pytest.fixture
def mock_vdw_pairwise_tensor():
    return torch.tensor(
        [
            [1.0, 1.5, 2.0, 2.5],
            [1.0, 1.5, 2.0, 2.5],
            [1.0, 1.5, 2.0, 2.5],
            [1.0, 1.5, 2.0, 2.5],
        ],
        dtype=torch.float64,
    )


def test_get_atomic_masses(atomic_numbers, mock_mass_tensor):
    # patch the name that `get_atomic_masses()` dereferences
    with patch(
        "tad_mctc.data.mass.ATOMIC",
        side_effect=lambda dtype=torch.float64, device=None: mock_mass_tensor,
    ):
        ref = mock_mass_tensor[atomic_numbers] * GMOL2AU
        assert (
            pytest.approx(ref.cpu()) == get_atomic_masses(atomic_numbers).cpu()
        )

        ref_no_au = mock_mass_tensor[atomic_numbers]
        assert (
            pytest.approx(ref_no_au.cpu())
            == get_atomic_masses(atomic_numbers, atomic_units=False).cpu()
        )


def test_get_zvalence(atomic_numbers, mock_zeff_tensor):
    with patch(
        "tad_mctc.data.zeff.ZVALENCE",
        side_effect=lambda dtype=torch.long, device=None: mock_zeff_tensor,
    ):
        ref = mock_zeff_tensor[atomic_numbers]
        assert pytest.approx(ref.cpu()) == get_zvalence(atomic_numbers).cpu()


def test_get_ecore(atomic_numbers, mock_zeff_tensor):
    with patch(
        "tad_mctc.data.zeff.ECORE",
        side_effect=lambda dtype=torch.long, device=None: mock_zeff_tensor,
    ):
        ref = mock_zeff_tensor[atomic_numbers]
        assert pytest.approx(ref.cpu()) == get_ecore(atomic_numbers).cpu()


def test_get_hardness(atomic_numbers, mock_hardness_tensor):
    with patch(
        "tad_mctc.data.hardness.GAM",
        side_effect=lambda dtype=torch.float64, device=None: mock_hardness_tensor,
    ):
        ref = mock_hardness_tensor[atomic_numbers]
        assert pytest.approx(ref.cpu()) == get_hardness(atomic_numbers).cpu()


def test_get_vdw_pairwise(atomic_numbers, mock_vdw_pairwise_tensor):
    # patch the name that `get_vdw_pairwise()` dereferences
    with patch(
        "tad_mctc.data.radii.VDW_PAIRWISE",
        side_effect=lambda dtype=torch.float64, device=None: mock_vdw_pairwise_tensor,
    ):
        ref = mock_vdw_pairwise_tensor[
            atomic_numbers.unsqueeze(-1), atomic_numbers.unsqueeze(-2)
        ]
        assert (
            pytest.approx(ref.cpu()) == get_vdw_pairwise(atomic_numbers).cpu()
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_load(dtype: torch.dtype) -> None:
    from tad_mctc.data import (
        ATOMIC,
        ECORE,
        GAM,
        VDW_D3,
        VDW_PAIRWISE,
        ZEFF,
        ZVALENCE,
    )

    atomic = ATOMIC(dtype=dtype)
    assert atomic.dtype == dtype

    ecore = ECORE(dtype=dtype)
    assert ecore.dtype == dtype

    gam = GAM(dtype=dtype)
    assert gam.dtype == dtype

    vdw_d3 = VDW_D3(dtype=dtype)
    assert vdw_d3.dtype == dtype

    vdw_pairwise = VDW_PAIRWISE(dtype=dtype)
    assert vdw_pairwise.dtype == dtype

    zeff = ZEFF(dtype=dtype)
    assert zeff.dtype == dtype

    zvalence = ZVALENCE(dtype=dtype)
    assert zvalence.dtype == dtype
