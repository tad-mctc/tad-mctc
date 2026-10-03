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
Test error handling in coordination number calculation.
"""

from __future__ import annotations

import pytest
import torch

from tad_mctc.io.structure import Structure
from tad_mctc.ncoord import CNModel, erf_count, exp_count, gfn2_count
from tad_mctc.typing import DD, CountingFunction

from ..conftest import DEVICE
from ._variants import VARIANTS


@pytest.mark.parametrize("variant_name", list(VARIANTS))
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_fail(variant_name: str, dtype: torch.dtype) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}
    function = VARIANTS[variant_name].call

    numbers = torch.tensor([1, 1], device=DEVICE)
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], **dd)

    # `Structure.__post_init__` (`structure_check`) rejects the shape
    # mismatch before `CNModel` ever sees it.
    with pytest.raises(RuntimeError):
        wrong_positions = positions[:1]
        function(Structure(numbers=numbers, positions=wrong_positions))

    with pytest.raises(RuntimeError):
        wrong_numbers = torch.tensor([1], device=DEVICE)
        function(Structure(numbers=wrong_numbers, positions=positions))


@pytest.mark.parametrize("cfunc", [erf_count, exp_count, gfn2_count])
def test_coordination_number_custom_counting(cfunc: CountingFunction) -> None:
    numbers = torch.tensor([6, 1], dtype=torch.long)
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64
    )

    model = CNModel(count=cfunc, cutoff=5.0)
    res = model(Structure(numbers=numbers, positions=positions))
    assert torch.isfinite(res).all()
