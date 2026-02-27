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
Test the calculation of the rotational constants.
"""

# from __future__ import annotations

# import pytest
# import torch

# from tad_mctc.data.mass import ATOMIC_MASS
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
#     masses = ATOMIC_MASS(**dd)[numbers]

#     ref = torch.tensor([2.0 / 3, 0.0, 0.0], **dd)
#     a = property.rot_consts(masses, positions)
#     assert pytest.approx(ref) == a * AU2RCM
