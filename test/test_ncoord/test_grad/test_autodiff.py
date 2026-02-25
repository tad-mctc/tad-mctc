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

from tad_mctc.autograd import dgradcheck, dgradgradcheck
from tad_mctc.batch import pack
from tad_mctc.ncoord import (
    cn_d3,
    cn_d4,
    cn_eeq,
    erf_count,
    exp_count,
    gfn2_count,
)
from tad_mctc.typing import DD, Callable, CNFunction, CountingFunction, Tensor

from ...conftest import DEVICE
from ..samples import samples

tol = 1e-8
sample_list = ["SiH4", "PbH4-BiH3", "MB16_43_01"]


def gradchecker(
    dtype: torch.dtype,
    name: str,
    cnf: CNFunction,
) -> tuple[
    Callable[[Tensor], Tensor],  # autograd function
    Tensor,  # differentiable variables
]:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)

    # variables to be differentiated
    positions.requires_grad_(True)

    def func(pos: Tensor) -> Tensor:
        return cnf(numbers, pos)

    return func, positions


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list)
@pytest.mark.parametrize("cn_function", [cn_d3, cn_d4, cn_eeq])
def test_gradcheck(
    dtype: torch.dtype,
    name: str,
    cn_function: CNFunction,
) -> None:
    """
    Check a single analytical gradient of parameters against numerical
    gradient from `torch.autograd.gradcheck`.
    """
    func, diffvars = gradchecker(dtype, name, cn_function)
    assert dgradcheck(func, diffvars, atol=tol)


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list)
@pytest.mark.parametrize("cn_function", [cn_d3, cn_d4, cn_eeq])
def test_gradgradcheck(
    dtype: torch.dtype, name: str, cn_function: CNFunction
) -> None:
    """
    Check a single analytical gradient of parameters against numerical
    gradient from `torch.autograd.gradgradcheck`.
    """
    func, diffvars = gradchecker(dtype, name, cn_function)
    assert dgradgradcheck(func, diffvars, atol=tol)


def gradchecker_batch(
    dtype: torch.dtype,
    name1: str,
    name2: str,
    cnf: CNFunction,
) -> tuple[
    Callable[[Tensor], Tensor],  # autograd function
    Tensor,  # differentiable variables
]:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    sample1, sample2 = samples[name1], samples[name2]
    numbers = pack(
        [
            sample1["numbers"].to(DEVICE),
            sample2["numbers"].to(DEVICE),
        ]
    )
    positions = pack(
        [
            sample1["positions"].to(**dd),
            sample2["positions"].to(**dd),
        ]
    )

    # variable to be differentiated
    positions = positions.requires_grad_(True)

    def func(pos: Tensor) -> Tensor:
        return cnf(numbers, pos)

    return func, positions


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["SiH4"])
@pytest.mark.parametrize("name2", sample_list)
@pytest.mark.parametrize("cn_function", [cn_d3, cn_d4, cn_eeq])
def test_gradcheck_batch(
    dtype: torch.dtype, name1: str, name2: str, cn_function: CNFunction
) -> None:
    """
    Check a single analytical gradient of parameters against numerical
    gradient from `torch.autograd.gradcheck`.
    """
    func, diffvars = gradchecker_batch(dtype, name1, name2, cn_function)
    assert dgradcheck(func, diffvars, atol=tol)


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["SiH4"])
@pytest.mark.parametrize("name2", sample_list)
@pytest.mark.parametrize("cn_function", [cn_d3, cn_d4, cn_eeq])
def test_gradgradcheck_batch(
    dtype: torch.dtype, name1: str, name2: str, cn_function: CNFunction
) -> None:
    """
    Check a single analytical gradient of parameters against numerical
    gradient from `torch.autograd.gradgradcheck`.
    """
    func, diffvars = gradchecker_batch(dtype, name1, name2, cn_function)
    assert dgradgradcheck(func, diffvars, atol=tol)
