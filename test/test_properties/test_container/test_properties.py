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

import pytest
import torch

from tad_mctc.data.getters import get_atomic_masses
from tad_mctc.data.structures import get_structure
from tad_mctc.io.structure import Structure
from tad_mctc.properties.general import center_of_mass, enn
from tad_mctc.typing import DD
from tad_mctc.units import GMOL2AU

from ...conftest import DEVICE

# Historical name -> mstore (collection, record): both are mstore records
# reached only through `get_structure`, see `test_ncoord/samples.py` for
# the same pattern and its rationale.
_SAMPLE_SOURCES = {
    "H2": ("mb16_43", "H2"),
    "H2O": ("heavy28", "h2o"),
}


def _structure(name: str) -> Structure:
    collection, record = _SAMPLE_SOURCES[name]
    return get_structure(collection, record)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_enn(dtype: torch.dtype) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    numbers = _structure("H2").numbers.to(DEVICE)
    positions = _structure("H2").positions.to(**dd)

    assert pytest.approx(1 / (2 * 0.702529)) == enn(numbers, positions).cpu()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_mass_center(dtype: torch.dtype) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    numbers = _structure("H2").numbers.to(DEVICE)
    positions = _structure("H2").positions.to(**dd)

    masses = get_atomic_masses(numbers, **dd)
    com = center_of_mass(masses, positions)

    ref = torch.tensor([0.0, 0.0, 0.0], **dd)
    assert pytest.approx(ref.cpu(), abs=1e-8) == com.cpu()


# H: 1.00797, O: 15.9994  (g/mol from the ATOMIC table)
_MASS_GMOL = {
    "H2": 2 * 1.00797,  # 2.01594
    "H2O": 15.9994 + 2 * 1.00797,  # 18.01534
}


@pytest.mark.parametrize("name", ["H2", "H2O"])
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_mass_gmol(name: str, dtype: torch.dtype) -> None:
    """Total mass returned in g/mol (default atomic_units=False)."""
    dd: DD = {"device": DEVICE, "dtype": dtype}
    tol = torch.finfo(dtype).eps * 100

    numbers = _structure(name).numbers.to(DEVICE)

    mass = get_atomic_masses(numbers, atomic_units=False, **dd).sum()

    ref = _MASS_GMOL[name]
    assert pytest.approx(ref, rel=tol) == mass.item()


@pytest.mark.parametrize("name", ["H2", "H2O"])
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_mass_au(name: str, dtype: torch.dtype) -> None:
    """Total mass returned in atomic units (atomic_units=True)."""
    dd: DD = {"device": DEVICE, "dtype": dtype}
    tol = torch.finfo(dtype).eps * 100

    numbers = _structure(name).numbers.to(DEVICE)

    mass = get_atomic_masses(numbers, atomic_units=True, **dd).sum()

    ref = _MASS_GMOL[name] * GMOL2AU
    assert pytest.approx(ref, rel=tol) == mass.item()
