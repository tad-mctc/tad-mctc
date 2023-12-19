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
Test tensor conversion tools.
"""
from __future__ import annotations

import pytest
import torch

from tad_mctc.convert import reshape_fortran, symmetrize


def test_reshape_fortran_1d():
    x = torch.tensor([1, 2, 3, 4, 5, 6])
    reshaped = reshape_fortran(x, (3, 2))

    # check the shape
    assert reshaped.shape == torch.Size((3, 2))

    # check values for correctness
    expected_values = torch.tensor([[1, 4], [2, 5], [3, 6]])
    assert torch.equal(reshaped, expected_values)


def test_reshape_fortran_2d():
    x = torch.tensor([[1, 2], [3, 4], [5, 6]])

    new_shape = (2, 3)
    reshaped = reshape_fortran(x, new_shape)

    # check shape
    assert reshaped.shape == torch.Size(new_shape)

    # check values to ensure column-major order
    expected_values = torch.tensor([[1, 5, 4], [3, 2, 6]])
    assert torch.equal(reshaped, expected_values)


def test_reshape_fortran_scalar():
    x = torch.tensor(5)
    assert len(x.shape) == 0

    # scalar tensor can only be reshaping to a 1-element shape
    new_shape = (1, 1, 1)
    reshaped = reshape_fortran(x, new_shape)

    # check the shape
    assert reshaped.shape == torch.Size(new_shape)

    # check the values (all equal to the scalar value)
    expected_values = torch.full(new_shape, 5)
    assert torch.equal(reshaped, expected_values)


def test_symmetrize_fail() -> None:
    with pytest.raises(RuntimeError):
        symmetrize(torch.tensor([1, 2, 3]))

    with pytest.raises(RuntimeError):
        mat = torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [2.0, 2.0, 2.0],
                [4.0, 3.0, 3.0],
            ]
        )
        symmetrize(mat)


def test_symmetrize_success() -> None:
    mat = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [2.0, 2.0, 2.0],
            [3.0, 2.0, 3.0],
        ]
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
