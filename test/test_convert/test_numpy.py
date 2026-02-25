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
Test numpy and PyTorch interconversion.
"""

from __future__ import annotations

import importlib
from contextlib import contextmanager
from unittest.mock import patch

import numpy as np
import pytest
import torch

from tad_mctc import convert
from tad_mctc._version import __tversion__
from tad_mctc.typing import DD, Generator, Tensor, get_default_dtype

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
    def torch_default_dtype(dtype: torch.dtype) -> Generator[None, None, None]:
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


@pytest.mark.parametrize("device_str", ["cpu"])
def test_np_to_torch_with_device(device_str: str) -> None:
    device = convert.str_to_device(device_str)

    arr = np.zeros((10, 10))
    tensor = convert.numpy_to_tensor(arr, device=device)

    assert isinstance(tensor, Tensor)
    assert tensor.device == device


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_torch_to_np_with_dtype(
    dtype: np.dtype[np.float32 | np.float64],
) -> None:
    tensor = torch.zeros((10, 10))
    arr = convert.tensor_to_numpy(tensor, dtype=dtype)

    assert isinstance(arr, np.ndarray)
    assert arr.dtype == dtype


@pytest.mark.parametrize("device_str", ["cpu"])
def test_torch_to_np_with_device(device_str: str) -> None:
    device = convert.str_to_device(device_str)

    tensor = torch.zeros((10, 10), device=device)
    arr = convert.tensor_to_numpy(tensor)

    assert isinstance(arr, np.ndarray)


@pytest.mark.skipif(__tversion__ < (2, 0, 0), reason="Requires torch>=2.0.0")
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


def test_torch_to_np_below_1_13_0():
    import tad_mctc._version

    torch_version = tad_mctc._version.__tversion__

    with patch("tad_mctc._version.__tversion__", new=(1, 9, 0)):
        # reload cached module to ensure that patched version is used
        importlib.reload(convert.numpy)

        tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        result = convert.tensor_to_numpy(tensor)

        assert isinstance(result, np.ndarray), "Result is not an ndarray"

    # reload for actual version
    importlib.reload(convert.numpy)
    assert torch_version == tad_mctc._version.__tversion__


@pytest.mark.skipif(__tversion__ < (1, 13, 0), reason="Requires torch>=1.13.0")
def test_torch_to_np_above_1_13_0():
    import tad_mctc._version

    torch_version = tad_mctc._version.__tversion__

    with patch("tad_mctc._version.__tversion__", new=(2, 0, 0)):
        # reload cached module to ensure that patched version is used
        importlib.reload(convert.numpy)

        with patch(
            "torch._C._functorch.is_gradtrackingtensor", return_value=False
        ) as mock_is_gradtrackingtensor:
            tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
            result = convert.tensor_to_numpy(tensor)

            assert isinstance(result, np.ndarray), "Result is not an ndarray"

            # Check that `mock_is_gradtrackingtensor` was called once
            assert mock_is_gradtrackingtensor.call_count == 1

            # Check that 1st call had correct args
            called_tensor = mock_is_gradtrackingtensor.call_args[0][0]
            assert torch.equal(called_tensor.cpu(), tensor.cpu())

    # reload for actual version
    importlib.reload(convert.numpy)
    assert torch_version == tad_mctc._version.__tversion__
