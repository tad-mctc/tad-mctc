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
Test tensor conversion tools.
"""

from __future__ import annotations

import pytest
import torch

from tad_mctc.convert import reshape_fortran, symmetrize
from tad_mctc.typing import DD

from ..conftest import DEVICE


def test_reshape_fortran_1d() -> None:
    x = torch.tensor([1, 2, 3, 4, 5, 6], device=DEVICE)
    reshaped = reshape_fortran(x, (3, 2))

    # check the shape
    assert reshaped.shape == torch.Size((3, 2))

    # check values for correctness
    expected_values = torch.tensor([[1, 4], [2, 5], [3, 6]], device=DEVICE)
    assert torch.equal(reshaped, expected_values)


def test_reshape_fortran_2d() -> None:
    x = torch.tensor([[1, 2], [3, 4], [5, 6]], device=DEVICE)

    new_shape = (2, 3)
    reshaped = reshape_fortran(x, new_shape)

    # check shape
    assert reshaped.shape == torch.Size(new_shape)

    # check values to ensure column-major order
    expected_values = torch.tensor([[1, 5, 4], [3, 2, 6]], device=DEVICE)
    assert torch.equal(reshaped, expected_values)


def test_reshape_fortran_scalar() -> None:
    x = torch.tensor(5, device=DEVICE)
    assert len(x.shape) == 0

    # scalar tensor can only be reshaping to a 1-element shape
    new_shape = (1, 1, 1)
    reshaped = reshape_fortran(x, new_shape)

    # check the shape
    assert reshaped.shape == torch.Size(new_shape)

    # check the values (all equal to the scalar value)
    expected_values = torch.full(new_shape, 5, device=DEVICE)
    assert torch.equal(reshaped, expected_values)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_symmetrize_fail(dtype: torch.dtype) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    with pytest.raises(RuntimeError):
        symmetrize(torch.tensor([1, 2, 3], device=DEVICE))

    with pytest.raises(RuntimeError):
        mat = torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [2.0, 2.0, 2.0],
                [4.0, 3.0, 3.0],
            ],
            device=DEVICE,
        )
        symmetrize(mat)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_symmetrize_success(dtype: torch.dtype) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    mat = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [2.0, 2.0, 2.0],
            [3.0, 2.0, 3.0],
        ],
        **dd,
    )

    sym = symmetrize(mat)
    assert (mat == sym).all()
    assert (sym == sym.mT).all()

    # make unsymmetric
    mat[0, -1] += 1e-5
    assert not (mat == mat.mT).all()

    # now symmetrize
    sym = symmetrize(mat)
    assert (sym == sym.mT).all()
