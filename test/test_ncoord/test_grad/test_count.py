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

import numpy as np
import pytest
import torch

from tad_mctc.convert import numpy_to_tensor
from tad_mctc.ncoord import (
    derf_count,
    dexp_count,
    dgfn2_count,
    erf_count,
    exp_count,
    gfn2_count,
)
from tad_mctc.typing import DD, CountingFunction

from ...conftest import DEVICE


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize(
    "function",
    [
        (exp_count, dexp_count),
        (erf_count, derf_count),
        (gfn2_count, dgfn2_count),
    ],
)
def test_single(
    dtype: torch.dtype, function: tuple[CountingFunction, CountingFunction]
) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    tol = torch.finfo(dtype).eps ** 0.5 * 10
    cf, dcf = function
    kcn = 7.5

    a = numpy_to_tensor(np.random.rand(4), **dd)
    b = numpy_to_tensor(np.random.rand(4), **dd)

    a_grad = a.detach().clone().requires_grad_(True)
    count = cf(a_grad, b, kcn)

    grad_auto = torch.autograd.grad(count.sum(-1), a_grad)[0]
    grad_expl = dcf(a, b, kcn)

    assert pytest.approx(grad_auto.cpu(), abs=tol) == grad_expl.cpu()
