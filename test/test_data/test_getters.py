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
    return torch.tensor([1.0, 4.0, 7.0], dtype=torch.float)


@pytest.fixture
def mock_zeff_tensor():
    return torch.tensor([1, 2, 3], dtype=torch.int)


def test_get_atomic_masses(atomic_numbers, mock_mass_tensor):
    with patch("tad_mctc.mass.ATOMIC", new=mock_mass_tensor):
        # Test in atomic units
        masses = get_atomic_masses(atomic_numbers)
        ref = mock_mass_tensor[atomic_numbers] * GMOL2AU
        assert pytest.approx(ref) == masses

        # Test without atomic units
        masses2 = get_atomic_masses(atomic_numbers, atomic_units=False)
        ref2 = mock_mass_tensor[atomic_numbers]
        assert pytest.approx(ref2) == masses2


def test_get_zvalence(atomic_numbers, mock_zeff_tensor):
    with patch("tad_mctc.zeff.ZVALENCE", new=mock_zeff_tensor):
        ref = mock_zeff_tensor[atomic_numbers]
        assert pytest.approx(ref) == get_zvalence(atomic_numbers)


def test_get_ecore(atomic_numbers, mock_zeff_tensor):
    with patch("tad_mctc.zeff.ECORE", new=mock_zeff_tensor):
        ref = mock_zeff_tensor[atomic_numbers]
        assert pytest.approx(ref) == get_ecore(atomic_numbers)
