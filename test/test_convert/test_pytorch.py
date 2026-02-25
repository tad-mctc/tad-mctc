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
Test PyTorch conversion tools.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from tad_mctc import convert
from tad_mctc.typing import Any


def test_fail() -> None:
    with pytest.raises(KeyError):
        convert.str_to_device("wrong")


# Test case for an unknown device string
def test_str_to_device_unknown() -> None:
    with pytest.raises(KeyError) as exc_info:
        convert.str_to_device("unknown_device")
    assert "Unknown device 'unknown_device' given." in str(exc_info.value)


# Test case for a CPU device
def test_str_to_device_cpu() -> None:
    device = convert.str_to_device("cpu")
    assert device.type == "cpu"


# Test case for attempting to use CUDA on a machine without CUDA
def test_str_to_device_no_cuda() -> None:
    with patch("torch.cuda.is_available", return_value=False):
        with pytest.raises(KeyError) as exc_info:
            convert.str_to_device("cuda")
        assert "No CUDA devices available." in str(exc_info.value)


# Test case for using CUDA when it's available
def test_str_to_device_with_cuda() -> None:
    with patch("torch.cuda.is_available", return_value=True):
        with patch("torch.cuda.current_device", return_value=0):
            device = convert.str_to_device("cuda")
            assert device.type == "cuda"
            assert device.index == 0


def test_normalize_device() -> None:
    assert convert.normalize_device("cpu").type == "cpu"
    assert convert.normalize_device(torch.device("cpu")).type == "cpu"
    assert convert.normalize_device(None) == torch.tensor(1).device


###############################################################################


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_any_to_tensor_with_tensor(dtype: torch.dtype) -> None:
    ref = torch.tensor([1, 2, 3], dtype=dtype)
    result = convert.any_to_tensor(ref, dtype=dtype)

    assert torch.is_tensor(result)
    assert result.dtype == dtype
    assert pytest.approx(ref.cpu()) == result.cpu()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_any_to_tensor_with_list(dtype: torch.dtype) -> None:
    result = convert.any_to_tensor([1, 2, 3], dtype=dtype)
    ref = torch.tensor([1, 2, 3], dtype=dtype)

    assert torch.is_tensor(result)
    assert result.dtype == dtype
    assert pytest.approx(ref.cpu()) == result.cpu()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_any_to_tensor_with_list_2(dtype: torch.dtype) -> None:
    result = convert.any_to_tensor([1, 2.5, False], dtype=dtype)
    ref = torch.tensor([1, 2.5, 0], dtype=dtype)

    assert torch.is_tensor(result)
    assert result.dtype == dtype
    assert pytest.approx(ref.cpu()) == result.cpu()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_any_to_tensor_with_float(dtype: torch.dtype) -> None:
    result = convert.any_to_tensor(3.14, dtype=dtype)
    assert torch.is_tensor(result)
    assert result.dtype == dtype
    assert pytest.approx(3.14) == result.cpu().item()


@pytest.mark.parametrize("dtype", [torch.int64, torch.double])
def test_any_to_tensor_with_int(dtype: torch.dtype) -> None:
    result = convert.any_to_tensor(42, dtype=dtype)
    assert torch.is_tensor(result)
    assert result.dtype == dtype
    assert pytest.approx(42.0) == result.cpu().item()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_any_to_tensor_with_bool(dtype: torch.dtype) -> None:
    result = convert.any_to_tensor(True, dtype=dtype)
    assert torch.is_tensor(result)
    assert result.dtype == dtype

    # For bool, it should be True, else 1
    assert result.item() == 1 if dtype != torch.bool else True


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_any_to_tensor_with_string(dtype: torch.dtype) -> None:
    result = convert.any_to_tensor("2.718", dtype=dtype)
    assert torch.is_tensor(result)
    assert result.dtype == dtype
    assert pytest.approx(2.718) == result.cpu().item()


def test_any_to_tensor_with_invalid_string() -> None:
    with pytest.raises(ValueError) as exc_info:
        convert.any_to_tensor("not_a_number")

        err_msg = "Cannot convert string 'not_a_number' to float"
        assert err_msg in str(exc_info.value)


def test_any_to_tensor_with_incompatible_type() -> None:
    with pytest.raises(TypeError) as exc_info:
        convert.any_to_tensor({"key": "value"})

        err_msg = "Tensor-incompatible type"
        assert err_msg in str(exc_info.value)


@pytest.mark.parametrize("invalid", [[1, "2"], [None, 1], [{"key": "value"}]])
def test_any_to_tensor_with_invalid_list(invalid: list[Any]) -> None:
    with pytest.raises(ValueError) as exc_info:
        convert.any_to_tensor(invalid)

        err_msg = "List items must be float, int, or bool types."
        assert err_msg in str(exc_info.value)
