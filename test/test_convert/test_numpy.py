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

import numpy as np
import pytest
import torch

from tad_mctc import convert
from tad_mctc.typing import Tensor, get_default_dtype


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
