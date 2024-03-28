# This file is part of tad-multicharge.
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
Test the checks for the numbers and positions given to the reader and writer.
"""
import pytest
import torch

from tad_mctc.exceptions import MoleculeError, MoleculeWarning
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


def test_deflatable() -> None:
    positions = torch.tensor([[0.0, 0.0, 1.5], [0.0, 0.0, 0.0]])

    with pytest.warns(MoleculeWarning):
        checks.deflatable_check(positions, raise_padding_warning=True)


def test_deflatable_skip_warning() -> None:
    positions = torch.tensor([[0.0, 0.0, 1.5], [0.0, 0.0, 0.0]])
    assert checks.deflatable_check(positions, raise_padding_warning=False)


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
