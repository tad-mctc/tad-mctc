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

from tad_mctc.autograd.compat import jacrev_compat as jacrev
from tad_mctc.convert import tensor_to_numpy
from tad_mctc.typing import DD, Tensor

from ..conftest import DEVICE


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_jacobian(dtype: torch.dtype) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}
    # Create a test input
    A = torch.tensor([[3.0, 2.0], [2.0, 3.0]], **dd)
    x = torch.tensor([1.0, 2.0], requires_grad=True, **dd)

    def linear(A: Tensor, x: Tensor) -> Tensor:
        """
        A simple linear function for testing.
        f(x) = Ax, where A is a constant matrix.
        The Jacobian of this function is A.
        """
        return A @ x

    # Calculate the Hessian using the `jacobian` function
    jacobian_matrix = jacrev(linear, argnums=1)(A, x)

    # Expected Jacobian for the quadratic function is A
    assert pytest.approx(A) == jacobian_matrix


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_argument_propagation(dtype: torch.dtype) -> None:
    dd = {"device": DEVICE, "dtype": dtype}

    def two_arg_func(x: Tensor, y: Tensor) -> Tensor:
        return x * y

    x = torch.tensor([1.0, 2.0], requires_grad=True, **dd)
    y = torch.tensor([3.0, 4.0], requires_grad=True, **dd)

    f_jac = jacrev(two_arg_func, argnums=1)
    jacobian_matrix = f_jac(x, y)
    expected = tensor_to_numpy(torch.diag(x))
    assert pytest.approx(expected) == jacobian_matrix


def test_non_tensor_input_error() -> None:
    def simple_func(x):
        return x**2

    f_jac = jacrev(simple_func)
    with pytest.raises(RuntimeError):
        f_jac("not a tensor")
