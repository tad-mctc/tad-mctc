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
Test non-functorch versions of Jacobian.
"""

from __future__ import annotations

import pytest
import torch

from tad_mctc.autograd.nonfunctorch import jac
from tad_mctc.convert import tensor_to_numpy
from tad_mctc.typing import DD, Tensor

from ..conftest import DEVICE


def _linear(A: Tensor, x: Tensor) -> Tensor:
    """
    A simple linear function for testing.
    f(x) = Ax, where A is a constant matrix.
    The Jacobian of this function is A.
    """
    return A @ x


def _quadratic(A: Tensor, x: Tensor) -> Tensor:
    """
    A simple quadratic function for testing.
    f(x) = x^T A x, where A is a constant matrix.
    The Hessian of this function is 2A.
    """
    return x @ A @ x


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("create_graph", [True, False, None])
def test_jacobian(dtype: torch.dtype, create_graph: bool | None) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    # Create a test input
    A = torch.tensor([[3.0, 2.0], [2.0, 3.0]], **dd)
    x = torch.tensor([1.0, 2.0], requires_grad=True, **dd)

    Ax = _linear(A, x)
    dAdx = jac(Ax, x, create_graph=create_graph)

    # Expected Jacobian for the quadratic function is A
    assert pytest.approx(A.cpu()) == tensor_to_numpy(dAdx)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_zeros(dtype: torch.dtype) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    # Create a test input
    A = torch.tensor([[3.0, 2.0], [2.0, 3.0]], **dd)
    x = torch.tensor([1.0, 2.0], **dd)

    Ax = _linear(A, x)
    dAdx = jac(Ax, x)

    # Expected Jacobian for the quadratic function is A
    assert pytest.approx(torch.zeros_like(dAdx).cpu()) == tensor_to_numpy(dAdx)


# @pytest.mark.parametrize("dtype", [torch.float, torch.double])
# def test_hessian(dtype: torch.dtype) -> None:
#     dd: DD = {"device": DEVICE, "dtype": dtype}
#     # Create a test input
#     A = torch.tensor([[3.0, 2.0], [2.0, 3.0]], **dd)
#     x = torch.tensor([1.0, 2.0], requires_grad=True, **dd)

#     # Calculate the Hessian using the `hessian` function
#     hessian_matrix = hess(_quadratic, (A, x), argnums=1)

#     # Expected Hessian for the quadratic function is 2A
#     assert pytest.approx(2 * A.cpu()) == hessian_matrix.cpu()


# @pytest.mark.parametrize("dtype", [torch.float, torch.double])
# def test_hessian_options(dtype: torch.dtype) -> None:
#     dd: DD = {"device": DEVICE, "dtype": dtype}
#     # Create a test input
#     A = torch.tensor([[3.0, 2.0], [2.0, 3.0]], **dd)
#     x = torch.tensor([1.0, 2.0], requires_grad=True, **dd)

#     # Calculate the Hessian using the `hessian` function
#     hessian_matrix = hess(_quadratic, (A, x), argnums=1, create_graph=True)

#     # Expected Hessian for the quadratic function is 2A
#     assert pytest.approx(2 * A.cpu()) == tensor_to_numpy(hessian_matrix)
