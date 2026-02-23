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
from tad_mctc.ncoord import cn_d3, cn_d3_gradient, dexp_count, exp_count
from tad_mctc.typing import DD

from ...conftest import DEVICE
from ..samples import samples
from ..utils import numgrad

sample_list = ["SiH4", "MB16_43_01"]


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_single(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}
    tol = torch.finfo(dtype).eps ** 0.5 * 50

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)

    ref = sample["dcn3dr"].to(**dd)
    ref = ref.reshape(numbers.shape[0], numbers.shape[0], 3)

    dcndr = cn_d3_gradient(numbers, positions, dcounting_function=dexp_count)
    numdr = numgrad(cn_d3, exp_count, numbers, positions)

    # the same atom gets masked in the PyTorch implementation
    mask = real_pairs(numbers, mask_diagonal=True).unsqueeze(-1)
    numdr = torch.where(mask, numdr, numdr.new_tensor(0.0))
    ref = torch.where(mask, ref, ref.new_tensor(0.0))

    assert pytest.approx(dcndr.cpu(), abs=tol) == numdr.cpu()
    assert pytest.approx(dcndr.cpu(), abs=tol) == ref.cpu()
    assert pytest.approx(numdr.cpu(), abs=tol) == ref.cpu()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", ["SiH4"])
@pytest.mark.parametrize("name2", sample_list)
def test_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    # slightly higher to avoid 10 / 1536 failing
    tol = torch.finfo(dtype).eps ** 0.5 * 50

    sample1, sample2 = samples[name1], samples[name2]

    numbers = pack(
        (
            sample1["numbers"].to(DEVICE),
            sample2["numbers"].to(DEVICE),
        )
    )
    nat1 = sample1["numbers"].to(DEVICE).shape[-1]
    nat2 = sample2["numbers"].to(DEVICE).shape[-1]

    positions = pack(
        (
            sample1["positions"].to(**dd),
            sample2["positions"].to(**dd),
        )
    )
    ref = pack(
        (
            sample1["dcn3dr"].to(**dd).reshape(nat1, nat1, 3),
            sample2["dcn3dr"].to(**dd).reshape(nat2, nat2, 3),
        ),
    )

    dcndr = cn_d3_gradient(numbers, positions, dcounting_function=dexp_count)
    numdr = numgrad(cn_d3, exp_count, numbers, positions)

    # the same atom gets masked in the PyTorch implementation
    mask = real_pairs(numbers, mask_diagonal=True).unsqueeze(-1)
    numdr = torch.where(mask, numdr, numdr.new_tensor(0.0))
    ref = torch.where(mask, ref, ref.new_tensor(0.0))

    assert pytest.approx(dcndr.cpu(), abs=tol) == numdr.cpu()
    assert pytest.approx(dcndr.cpu(), abs=tol) == ref.cpu()
    assert pytest.approx(numdr.cpu(), abs=tol) == ref.cpu()
