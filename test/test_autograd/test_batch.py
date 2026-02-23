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
Test hessian.
"""

import pytest
import torch

from tad_mctc.autograd import bjacrev, jacrev
from tad_mctc.autograd.compat import vmap_compat
from tad_mctc.typing import DD, Tensor

from ..conftest import DEVICE


def _linear(A: Tensor, x: Tensor) -> Tensor:
    """
    A simple linear function for testing.
    f(x) = Ax, where A is a constant matrix.
    The Jacobian of this function is A.
    """
    return A @ x


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_bjacrev(dtype: torch.dtype) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    A = torch.tensor(
        [
            [[3.0, 2.0], [2.0, 3.0]],
            [[1.0, 2.0], [2.0, 4.0]],
        ],
        **dd,
    )
    x = torch.tensor(
        [
            [1.0, 2.0],
            [1.0, 2.0],
        ],
        **dd,
        requires_grad=True,
    )

    f = bjacrev(_linear, argnums=1)
    jacobian_matrix = f(A, x)

    # Expected Jacobian for the quadratic function is A
    assert pytest.approx(A.cpu()) == jacobian_matrix.cpu()


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_vmap_compat(dtype: torch.dtype) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    A = torch.tensor(
        [
            [[3.0, 2.0], [2.0, 3.0]],
            [[1.0, 2.0], [2.0, 4.0]],
        ],
        **dd,
    )
    x = torch.tensor(
        [
            [1.0, 2.0],
            [1.0, 2.0],
        ],
        **dd,
        requires_grad=True,
    )

    f = vmap_compat(jacrev(_linear, argnums=1))  # type: ignore
    jacobian_matrix = f(A, x)

    # Expected Jacobian for the quadratic function is A
    assert pytest.approx(A.cpu()) == jacobian_matrix.cpu()
