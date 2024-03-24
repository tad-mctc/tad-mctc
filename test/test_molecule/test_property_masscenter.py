"""
Test the utility functions for molecular properties.
"""

from __future__ import annotations

import pytest
import torch

from tad_mctc.molecule import property
from tad_mctc.typing import DD

from ..conftest import DEVICE


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_mass_center_single(dtype: torch.dtype) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    masses = torch.tensor([1.0, 2.0], **dd)
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], **dd)

    ref = torch.tensor([2.0 / 3, 0.0, 0.0], **dd)
    assert pytest.approx(ref) == property.mass_center(masses, positions)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_mass_center_batch(dtype: torch.dtype) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    masses = torch.tensor([[1.0, 2.0], [2.0, 1.0]], **dd)
    positions = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        ],
        **dd,
    )

    ref = torch.tensor([[2.0 / 3, 0.0, 0.0], [0.0, 1.0 / 3, 0.0]], **dd)
    assert pytest.approx(ref) == property.mass_center(masses, positions)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_mass_center_zero(dtype: torch.dtype) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    masses = torch.tensor([0.0, 2.0], **dd)
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], **dd)

    ref = torch.tensor([1.0, 0.0, 0.0], **dd)
    assert pytest.approx(ref) == property.mass_center(masses, positions)
