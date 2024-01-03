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


def test_str_to_device_no_cuda() -> None:
    with patch("torch.cuda.is_available", return_value=False):
        with pytest.raises(KeyError) as exc_info:
            convert.str_to_device("cuda")
        assert "No CUDA devices available." in str(exc_info.value)
