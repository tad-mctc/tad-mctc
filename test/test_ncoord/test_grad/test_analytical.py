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

from tad_mctc.batch import pack, real_pairs
from tad_mctc.ncoord import (  # cn_d4,; cn_d4_gradient,; cn_eeq,; cn_eeq_gradient,
    cn_d3,
    cn_d3_gradient,
    derf_count,
    dexp_count,
    dgfn2_count,
    erf_count,
    exp_count,
    gfn2_count,
)
from tad_mctc.typing import DD, CNFunction, CNGradFunction, CountingFunction

from ...conftest import DEVICE
from ..samples import samples
from ..utils import numgrad

sample_list = ["SiH4", "PbH4-BiH3", "MB16_43_01"]


@pytest.mark.parametrize(
    "function",
    [
        (cn_d3, cn_d3_gradient),
    ],
)
@pytest.mark.parametrize(
    "cfunc",
    [
        (exp_count, dexp_count),
        (erf_count, derf_count),
        (gfn2_count, dgfn2_count),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_single(
    function: tuple[CNFunction, CNGradFunction],
    cfunc: tuple[CountingFunction, CountingFunction],
    dtype: torch.dtype,
    name: str,
) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}
    tol = torch.finfo(dtype).eps ** 0.5 * 50

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)
    cutoff = torch.tensor(50, **dd)

    numdr = numgrad(function[0], cfunc[0], numbers, positions)
    dcndr = function[1](
        numbers, positions, dcounting_function=cfunc[1], cutoff=cutoff
    )

    # the same atom gets masked in the PyTorch implementation
    mask = real_pairs(numbers, mask_diagonal=True).unsqueeze(-1)
    numdr = torch.where(mask, numdr, numdr.new_tensor(0.0))

    assert pytest.approx(dcndr.cpu(), abs=tol) == numdr.cpu()


@pytest.mark.parametrize(
    "function",
    [
        (cn_d3, cn_d3_gradient),
    ],
)
@pytest.mark.parametrize(
    "cfunc",
    [
        (exp_count, dexp_count),
        (erf_count, derf_count),
        (gfn2_count, dgfn2_count),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", ["SiH4"])
@pytest.mark.parametrize("name2", sample_list)
def test_batch(
    function: tuple[CNFunction, CNGradFunction],
    cfunc: tuple[CountingFunction, CountingFunction],
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

    numdr = numgrad(function[0], cfunc[0], numbers, positions)
    dcndr = function[1](numbers, positions, dcounting_function=cfunc[1])

    # the same atom gets masked in the PyTorch implementation
    mask = real_pairs(numbers, mask_diagonal=True).unsqueeze(-1)
    numdr = torch.where(mask, numdr, numdr.new_tensor(0.0))

    assert pytest.approx(dcndr.cpu(), abs=tol) == numdr.cpu()
