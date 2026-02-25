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
Test derivative (w.r.t. positions) of the exponential and error counting
functions used for the coordination number within the EEQ model and D4.
"""

from __future__ import annotations

import pytest
import torch

from tad_mctc.autograd import bjacrev, jacrev
from tad_mctc.batch import pack
from tad_mctc.convert import tensor_to_numpy
from tad_mctc.ncoord import (
    cn_d3,
    cn_d4,
    cn_eeq,
    derf_count,
    dexp_count,
    dgfn2_count,
    erf_count,
    exp_count,
    gfn2_count,
)
from tad_mctc.typing import DD, CNFunction, CountingFunction, Tensor

from ...conftest import DEVICE
from ..samples import samples
from ..utils import numgrad

sample_list = ["SiH4", "PbH4-BiH3", "MB16_43_01"]


@pytest.mark.parametrize("function", [cn_d3, cn_d4, cn_eeq])
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_single(
    function: CNFunction,
    dtype: torch.dtype,
    name: str,
) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}
    tol = torch.finfo(dtype).eps ** 0.5 * 50

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)

    # numerical gradient as ref
    numdr = numgrad(function, numbers, positions)

    def wrapper(pos: Tensor) -> Tensor:
        return function(numbers, pos)

    pos = positions.detach().clone().requires_grad_(True)
    jac: Tensor = jacrev(wrapper)(pos)  # type: ignore
    assert pytest.approx(numdr.cpu(), abs=tol) == tensor_to_numpy(jac)


@pytest.mark.parametrize("function", [cn_d3, cn_d4, cn_eeq])
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", ["SiH4"])
@pytest.mark.parametrize("name2", sample_list)
def test_batch(
    function: CNFunction,
    dtype: torch.dtype,
    name1: str,
    name2: str,
) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}
    tol = torch.finfo(dtype).eps ** 0.5 * 50

    sample1, sample2 = samples[name1], samples[name2]

    numbers = pack(
        (
            sample1["numbers"].to(DEVICE),
            sample2["numbers"].to(DEVICE),
        )
    )
    positions = pack(
        (
            sample1["positions"].to(**dd),
            sample2["positions"].to(**dd),
        )
    )

    # numerical gradient as ref
    numdr = numgrad(function, numbers, positions)

    def wrapper(num: Tensor, pos: Tensor) -> Tensor:
        return function(num, pos)

    pos = positions.detach().clone().requires_grad_(True)
    jac: Tensor = bjacrev(wrapper, argnums=1)(numbers, pos)  # type: ignore
    assert pytest.approx(numdr.cpu(), abs=tol) == tensor_to_numpy(jac)
