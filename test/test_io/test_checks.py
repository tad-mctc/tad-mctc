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
Test the checks for the numbers and positions given to the reader and writer.
"""
import pytest
import torch

from tad_mctc.exceptions import MoleculeError
from tad_mctc.io import checks

natoms = 4
ncart = 3


def test_coldfusion() -> None:
    # distances above threshold
    numbers = torch.tensor([1, 2])
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]])

    # Should pass without error
    assert checks.coldfusion_check(numbers, positions)

    # distances below threshold
    positions_close = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.1]])
    with pytest.raises(MoleculeError):
        checks.coldfusion_check(numbers, positions_close, threshold=0.5)


def test_content() -> None:
    # Valid case
    numbers = torch.tensor([1, 8])
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]])
    assert checks.content_checks(numbers, positions)

    # Invalid case: atomic number too large (larger than pse.MAX_ELEMENT)
    numbers_large = torch.tensor([1, 200])
    with pytest.raises(MoleculeError):
        checks.content_checks(numbers_large, positions)

    # Invalid case: atomic number too small
    numbers_small = torch.tensor([1, 0])
    with pytest.raises(MoleculeError):
        checks.content_checks(numbers_small, positions)


def test_shape_valid() -> None:
    # Valid shapes
    numbers = torch.zeros((natoms,))
    positions = torch.zeros((natoms, ncart))
    assert checks.shape_checks(numbers, positions)


def test_shape_mismatched_shapes() -> None:
    # Mismatched shapes between numbers and positions
    numbers = torch.zeros((natoms + 1,))
    positions = torch.zeros((natoms, ncart))
    with pytest.raises(ValueError):
        checks.shape_checks(numbers, positions)

    numbers = torch.zeros(0)
    positions = torch.zeros((1,))
    with pytest.raises(ValueError):
        checks.shape_checks(numbers, positions)


def test_shape_incorrect_dimensions_numbers() -> None:
    nbatch = 10

    # batch dimension
    numbers = torch.zeros((nbatch, natoms))
    positions = torch.zeros((nbatch, natoms, ncart))
    with pytest.raises(ValueError):
        checks.shape_checks(numbers, positions)


def test_shape_incorrect_cartesian_directions() -> None:
    # Incorrect size for the last dimension of positions
    numbers = torch.zeros(natoms)
    positions = torch.zeros((natoms, ncart - 1))
    with pytest.raises(ValueError):
        checks.shape_checks(numbers, positions)
