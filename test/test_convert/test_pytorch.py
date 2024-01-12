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
Test PyTorch conversion tools.
"""
from unittest.mock import patch

import pytest

from tad_mctc import convert


def test_fail() -> None:
    with pytest.raises(KeyError):
        convert.str_to_device("wrong")


# Test case for an unknown device string
def test_str_to_device_unknown():
    with pytest.raises(KeyError) as exc_info:
        convert.str_to_device("unknown_device")
    assert "Unknown device 'unknown_device' given." in str(exc_info.value)


# Test case for a CPU device
def test_str_to_device_cpu():
    device = convert.str_to_device("cpu")
    assert device.type == "cpu"


# Test case for attempting to use CUDA on a machine without CUDA
def test_str_to_device_no_cuda():
    with patch("torch.cuda.is_available", return_value=False):
        with pytest.raises(KeyError) as exc_info:
            convert.str_to_device("cuda")
        assert "No CUDA devices available." in str(exc_info.value)


# Test case for using CUDA when it's available
def test_str_to_device_with_cuda():
    with patch("torch.cuda.is_available", return_value=True):
        with patch("torch.cuda.current_device", return_value=0):
            device = convert.str_to_device("cuda")
            assert device.type == "cuda"
            assert device.index == 0
