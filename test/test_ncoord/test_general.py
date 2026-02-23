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

from tad_mctc.ncoord import (
    cn_d3,
    cn_d3_gradient,
    cn_d4,
    cn_eeq,
    derf_count,
    dexp_count,
    dgfn2_count,
    erf_count,
    exp_count,
    gfn2_count,
)
from tad_mctc.typing import DD, CNFunction, CNGradFunction, CountingFunction

from ..conftest import DEVICE


@pytest.mark.parametrize("function", [cn_d3, cn_d4, cn_eeq])
@pytest.mark.parametrize("cfunc", [erf_count, exp_count, gfn2_count])
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_fail(
    function: CNFunction, cfunc: CountingFunction, dtype: torch.dtype
) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    numbers = torch.tensor([1, 1], device=DEVICE)
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], **dd)

    # rcov wrong shape
    with pytest.raises(ValueError):
        rcov = torch.tensor([1.0], **dd)
        function(numbers, positions, counting_function=cfunc, rcov=rcov)

    # wrong numbers
    with pytest.raises(ValueError):
        numbers = torch.tensor([1], device=DEVICE)
        function(numbers, positions, counting_function=cfunc)


@pytest.mark.parametrize("function", [cn_d3_gradient])
@pytest.mark.parametrize("cfunc", [derf_count, dexp_count, dgfn2_count])
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_grad_fail(
    function: CNGradFunction, cfunc: CountingFunction, dtype: torch.dtype
) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    numbers = torch.tensor([1, 1], device=DEVICE)
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], **dd)

    # rcov wrong shape
    with pytest.raises(ValueError):
        rcov = torch.tensor([1.0], **dd)
        function(numbers, positions, dcounting_function=cfunc, rcov=rcov)

    # wrong numbers
    with pytest.raises(ValueError):
        numbers = torch.tensor([1], device=DEVICE)
        function(numbers, positions, dcounting_function=cfunc)
