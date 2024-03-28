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
Test the calculation of the rotational constants.
"""

# from __future__ import annotations

# import pytest
# import torch

# from tad_mctc.data.mass import ATOMIC
# from tad_mctc.data.molecules import mols as samples
# from tad_mctc.molecule import property
# from tad_mctc.typing import DD
# from tad_mctc.units import AU2RCM

# from ...conftest import DEVICE

# slist = ["SiH4", "PbH4-BiH3", "C6H5I-CH3SH", "MB16_43_01"]
# slist = ["CO2", "CH4"]


# @pytest.mark.parametrize("name", slist)
# @pytest.mark.parametrize("dtype", [torch.float, torch.double])
# def test_rotconst_single(dtype: torch.dtype, name: str) -> None:
#     dd: DD = {"device": DEVICE, "dtype": dtype}

#     sample = samples[name]
#     numbers = sample["numbers"].to(DEVICE)
#     positions = sample["positions"].to(**dd)
#     masses = ATOMIC.to(**dd)[numbers]

#     ref = torch.tensor([2.0 / 3, 0.0, 0.0], **dd)
#     a = property.rot_consts(masses, positions)
#     assert pytest.approx(ref) == a * AU2RCM
