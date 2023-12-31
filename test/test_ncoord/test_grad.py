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

from ..conftest import DEVICE


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize(
    "function",
    [
        (exp_count, dexp_count),
        (erf_count, derf_count),
        (gfn2_count, dgfn2_count),
    ],
)
def test_count(
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
