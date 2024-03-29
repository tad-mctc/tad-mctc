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
Test getters for atomic data.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from tad_mctc.data.getters import get_atomic_masses, get_ecore, get_zvalence
from tad_mctc.units import GMOL2AU


@pytest.fixture
def atomic_numbers():
    return torch.tensor([1, 2, 3], dtype=torch.int)


@pytest.fixture
def mock_mass_tensor():
    return torch.tensor([1.0, 4.0, 7.0, 10.0], dtype=torch.float)


@pytest.fixture
def mock_zeff_tensor():
    return torch.tensor([1, 2, 3, 4], dtype=torch.int)


def test_get_atomic_masses(atomic_numbers, mock_mass_tensor):
    with patch("tad_mctc.data.mass.ATOMIC", new=mock_mass_tensor):
        # Test in atomic units
        masses = get_atomic_masses(atomic_numbers)
        ref = mock_mass_tensor[atomic_numbers] * GMOL2AU
        assert pytest.approx(ref.cpu()) == masses.cpu()

        # Test without atomic units
        masses2 = get_atomic_masses(atomic_numbers, atomic_units=False)
        ref2 = mock_mass_tensor[atomic_numbers]
        assert pytest.approx(ref2.cpu()) == masses2.cpu()


def test_get_zvalence(atomic_numbers, mock_zeff_tensor):
    with patch("tad_mctc.data.zeff.ZVALENCE", new=mock_zeff_tensor):
        ref = mock_zeff_tensor[atomic_numbers]
        assert pytest.approx(ref.cpu()) == get_zvalence(atomic_numbers).cpu()


def test_get_ecore(atomic_numbers, mock_zeff_tensor):
    with patch("tad_mctc.data.zeff.ECORE", new=mock_zeff_tensor):
        ref = mock_zeff_tensor[atomic_numbers]
        assert pytest.approx(ref.cpu()) == get_ecore(atomic_numbers).cpu()
