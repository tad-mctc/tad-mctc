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
Test hessian.
"""
import pytest
import torch

from tad_mctc.autograd import hessian
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
    assert pytest.approx(2 * A) == hessian_matrix


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
    assert pytest.approx(expected_hessian) == hessian_matrix
