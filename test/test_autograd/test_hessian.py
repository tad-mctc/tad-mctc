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

from tad_mctc._version import __tversion__
from tad_mctc.autograd import hess_fn_rev, hessian
from tad_mctc.typing import DD, Tensor

from ..conftest import DEVICE


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_hessian(dtype: torch.dtype) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}
    # Create a test input
    A = torch.tensor([[3.0, 2.0], [2.0, 3.0]], **dd)
    x = torch.tensor([1.0, 2.0], requires_grad=True, **dd)

    def quadratic(A: Tensor, x: Tensor) -> Tensor:
        """
        A simple quadratic function for testing.
        f(x) = x^T A x, where A is a constant matrix.
        The Hessian of this function is 2A.
        """
        return x @ A @ x

    # Calculate the Hessian using the `hessian` function
    hessian_matrix = hessian(quadratic, (A, x), argnums=1)

    # Expected Hessian for the quadratic function is 2A
    assert pytest.approx(2 * A.cpu()) == hessian_matrix.cpu()


@pytest.mark.skipif(__tversion__ < (2, 0, 0), reason="Requires torch>=2.0.0")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_hessian_rev(dtype: torch.dtype) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}
    # Create a test input
    A = torch.tensor([[3.0, 2.0], [2.0, 3.0]], **dd)
    x = torch.tensor([1.0, 2.0], requires_grad=True, **dd)

    def quadratic(A: Tensor, x: Tensor) -> Tensor:
        """
        A simple quadratic function for testing.
        f(x) = x^T A x, where A is a constant matrix.
        The Hessian of this function is 2A.
        """
        return x @ A @ x

    # Calculate the Hessian using the `hessian` function
    hessian_matrix = hess_fn_rev(quadratic, argnums=1)(A, x)

    # Expected Hessian for the quadratic function is 2A
    assert pytest.approx(2 * A.cpu()) == hessian_matrix.cpu()


def test_hessian_runtime_error() -> None:
    def function(x: Tensor) -> Tensor:
        return x * x

    # Non-tensor input
    x = 5
    with pytest.raises(ValueError):
        hessian(function, (x,), argnums=0)


def test_hessian_missing_gradients() -> None:
    x = torch.tensor([1.0, 2.0], requires_grad=True)

    # Constant function, independent of x
    def constant_function(_: Tensor) -> Tensor:
        return torch.tensor(5.0)

    hessian_matrix = hessian(constant_function, (x,), argnums=0)

    # Expecting a zero matrix of shape [2, 2] as the function is constant
    expected_hessian = torch.zeros((2, 2), dtype=x.dtype, device=x.device)
    assert pytest.approx(expected_hessian.cpu()) == hessian_matrix.cpu()
