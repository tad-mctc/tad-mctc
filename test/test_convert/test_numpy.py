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
Test numpy and PyTorch interconversion.
"""
from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import pytest
import torch

from tad_mctc import convert
from tad_mctc.typing import Tensor, get_default_dtype, DD

from ..conftest import DEVICE


def test_np_to_torch_float32() -> None:
    """Test if the dtype is retained."""
    arr = np.zeros((10, 10), dtype=np.float32)
    tensor = convert.numpy_to_tensor(arr, dtype=torch.float32)

    assert isinstance(tensor, Tensor)
    assert tensor.dtype == torch.float32


def test_np_to_torch_float64() -> None:
    """Test if the dtype is retained."""
    arr = np.zeros((10, 10), dtype=np.float64)
    tensor = convert.numpy_to_tensor(arr, dtype=torch.float64)

    assert isinstance(tensor, Tensor)
    assert tensor.dtype == torch.float64


def test_np_to_torch_default() -> None:
    """Test if the dtype is retained."""
    arr = np.zeros((10, 10), dtype=np.float64)
    tensor = convert.numpy_to_tensor(arr, dtype=get_default_dtype())

    assert isinstance(tensor, Tensor)
    assert tensor.dtype == get_default_dtype()


def test_np_to_torch_default_context() -> None:
    """Test if the dtype is retained."""

    @contextmanager
    def torch_default_dtype(dtype):
        original_dtype = get_default_dtype()
        torch.set_default_dtype(dtype)
        try:
            yield
        finally:
            torch.set_default_dtype(original_dtype)

    with torch_default_dtype(torch.float64):
        arr = np.zeros((10, 10), dtype=np.float64)
        tensor = convert.numpy_to_tensor(arr)

        assert isinstance(tensor, Tensor)
        assert tensor.dtype == get_default_dtype()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_np_to_torch_with_dtype(dtype: torch.dtype) -> None:
    arr = np.zeros((10, 10))
    tensor = convert.numpy_to_tensor(arr, dtype=dtype)

    assert isinstance(tensor, Tensor)
    assert tensor.dtype == dtype


@pytest.mark.cuda
@pytest.mark.parametrize("device_str", ["cpu", "cuda"])
def test_np_to_torch_with_device(device_str: str) -> None:
    device = convert.str_to_device(device_str)

    arr = np.zeros((10, 10))
    tensor = convert.numpy_to_tensor(arr, device=device)

    assert isinstance(tensor, Tensor)
    assert tensor.device == device


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_torch_to_np_with_dtype(dtype: np.dtype[np.float32 | np.float64]) -> None:
    tensor = torch.zeros((10, 10))
    arr = convert.tensor_to_numpy(tensor, dtype=dtype)

    assert isinstance(arr, np.ndarray)
    assert arr.dtype == dtype


@pytest.mark.cuda
@pytest.mark.parametrize("device_str", ["cpu", "cuda"])
def test_torch_to_np_with_device(device_str: str) -> None:
    device = convert.str_to_device(device_str)

    tensor = torch.zeros((10, 10), device=device)
    arr = convert.tensor_to_numpy(tensor)

    assert isinstance(arr, np.ndarray)


def test_torch_to_np_with_transforms_fail() -> None:
    from tad_mctc.autograd import jacrev

    dd: DD = {"device": DEVICE, "dtype": torch.double}

    # Create a tensor with requires_grad=True
    x = torch.tensor([3, 3], **dd, requires_grad=True)
    y = torch.tensor([[3, 2], [1, 4]], dtype=torch.double)

    def simple_function(x: Tensor, y: Tensor) -> Tensor:
        _ = np.array(y.tolist())  # fails!
        return x**2

    jacobian_func = jacrev(simple_function)

    with pytest.raises(RuntimeError):
        jacobian_func(x, y)


@pytest.mark.parametrize("dtype", [torch.float, torch.double, torch.int64])
def test_torch_to_np_with_transforms(dtype: torch.dtype) -> None:
    from tad_mctc.autograd import jacrev

    dd: DD = {"device": DEVICE, "dtype": torch.double}

    x = torch.tensor([3, 3], **dd, requires_grad=True)
    y = torch.tensor([[3, 2], [1, 4]], **dd)

    npdtype = convert.numpy.torch_to_numpy_dtype_dict[dtype]

    def simple_function(x: Tensor, y: Tensor) -> Tensor:
        t = convert.tensor_to_numpy(y, dtype=npdtype)
        assert t.dtype == npdtype

        for yi, ti in zip(y.flatten(), t.flatten()):
            assert yi.item() == ti

        return x**2

    jacobian_func = jacrev(simple_function)
    jacobian_func(x, y)

    # requires multiple unwraps in tensor_to_numpy conversion
    jacobian_func = jacrev(jacrev(simple_function))
    jacobian_func(x.detach().clone().requires_grad_(), y)
