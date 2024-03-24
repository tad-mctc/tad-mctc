"""
Test the utility functions for molecular properties.
"""

from __future__ import annotations

import pytest
import torch

from tad_mctc.data.mass import ATOMIC
from tad_mctc.data.molecules import mols as samples
from tad_mctc.molecule import property
from tad_mctc.typing import DD
from tad_mctc.units import AU2RCM

from ..conftest import DEVICE

slist = ["SiH4", "PbH4-BiH3", "C6H5I-CH3SH", "MB16_43_01"]
slist = ["CO2", "CH4"]


@pytest.mark.parametrize("name", slist)
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_rotconst_single(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)
    masses = ATOMIC.to(**dd)[numbers]

    ref = torch.tensor([2.0 / 3, 0.0, 0.0], **dd)
    a = property.rot_consts(masses, positions)
    # assert pytest.approx(ref) == a * AU2RCM
