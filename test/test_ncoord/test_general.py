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
    cn_eeq_en,
    cn_gfn2,
    coordination_number,
    derf_count,
    dexp_count,
    dgfn2_count,
    erf_count,
    exp_count,
    gfn2_count,
)
from tad_mctc.typing import DD, CNFunction, CNGradFunction, CountingFunction

from ..conftest import DEVICE


@pytest.mark.parametrize("function", [cn_d3, cn_d4, cn_gfn2, cn_eeq, cn_eeq_en])
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_fail(function: CNFunction, dtype: torch.dtype) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    numbers = torch.tensor([1, 1], device=DEVICE)
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], **dd)

    with pytest.raises(ValueError):
        wrong_positions = positions[:1]
        function(numbers, wrong_positions)

    with pytest.raises(ValueError):
        wrong_numbers = torch.tensor([1], device=DEVICE)
        function(wrong_numbers, positions)


@pytest.mark.parametrize("cfunc", [erf_count, exp_count, gfn2_count])
def test_coordination_number_custom_counting(cfunc: CountingFunction) -> None:
    numbers = torch.tensor([6, 1], dtype=torch.long)
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64
    )
    cutoff = torch.tensor(5.0, dtype=torch.float64)

    res = coordination_number(
        numbers,
        positions,
        counting_function=cfunc,
        cutoff=cutoff,
    )
    assert torch.isfinite(res).all()


@pytest.mark.parametrize("cfunc", [erf_count, exp_count, gfn2_count])
def test_coordination_number_shape_fail(cfunc: CountingFunction) -> None:
    numbers = torch.tensor([1, 1], dtype=torch.long)
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64
    )
    rcov = torch.tensor([1.0], dtype=torch.float64)
    cutoff = torch.tensor(5.0, dtype=torch.float64)

    with pytest.raises(ValueError):
        coordination_number(
            numbers,
            positions,
            counting_function=cfunc,
            rcov=rcov,
            cutoff=cutoff,
        )


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
