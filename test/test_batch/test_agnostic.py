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
Test the batch-agnostic drop-in functions.
"""
from __future__ import annotations

import torch

from tad_mctc.batch import eye


def test_eye_single():
    shape = (3, 3)
    value = 1.0
    tensor = torch.empty(shape)

    result = eye(tensor, value=value)

    assert result.shape == shape
    assert result.device == tensor.device
    assert result.dtype == tensor.dtype

    # Check the diagonal values and off-diagonal values
    assert torch.all(result.diag() == value)
    assert torch.all(result[result != value] == 0)


def test_eye_batch():
    shape = (5, 3, 3)
    value = 1.0
    tensor = torch.empty(shape)

    result = eye(tensor, value=value)

    assert result.shape == shape
    assert result.device == tensor.device
    assert result.dtype == tensor.dtype

    # Check each identity matrix in the batch
    for i in range(shape[0]):
        assert torch.all(result[i].diag() == value)
        assert torch.all(result[i][result[i] != value] == 0)
