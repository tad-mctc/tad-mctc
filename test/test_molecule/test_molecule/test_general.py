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
Test the molecule representation.
"""

from __future__ import annotations

import pytest
import torch

from tad_mctc.exceptions import DeviceError, DtypeError
from tad_mctc.molecule.container import Mol
from tad_mctc.typing import MockTensor, Tensor, get_default_dtype

device = None


def test_fail() -> None:
    dummy = torch.randint(1, 118, (2,))

    with pytest.raises(TypeError):
        Mol("wrong", dummy)  # type: ignore

    with pytest.raises(TypeError):
        Mol(dummy, "wrong")  # type: ignore

    with pytest.raises(ValueError):
        Mol(dummy, dummy, "wrong")  # type: ignore


def test_shape() -> None:
    numbers = torch.randint(1, 118, (5,))
    positions = torch.randn((5, 3))

    # shape mismatch with positions
    with pytest.raises(RuntimeError):
        Mol(torch.randint(1, 118, (1,)), positions)

    # shape mismatch with numbers
    with pytest.raises(RuntimeError):
        Mol(numbers, torch.randn((4, 3)))

    # too many dimensions
    with pytest.raises(RuntimeError):
        Mol(torch.randint(1, 118, (1, 2, 3)), positions)

    # too many dimensions
    with pytest.raises(RuntimeError):
        Mol(numbers, torch.randn(1, 2, 3, 4))


def test_setter() -> None:
    numbers = torch.randint(1, 118, (5,))
    positions = torch.randn((5, 3))
    mol = Mol(numbers, positions)

    with pytest.raises(RuntimeError):
        mol.numbers = torch.randint(1, 118, (1,))

    with pytest.raises(RuntimeError):
        mol.positions = torch.randn(1)


def test_getter() -> None:
    numbers = torch.randint(1, 118, (5,))
    positions = torch.randn((5, 3))
    mol = Mol(numbers, positions)

    assert pytest.approx(numbers) == mol.numbers
    assert pytest.approx(positions) == mol.positions


def test_charge() -> None:
    numbers = torch.randint(1, 118, (5,))
    positions = torch.randn((5, 3))
    mol = Mol(numbers, positions, charge=1)

    # charge as int
    assert isinstance(mol.charge, Tensor)
    assert mol.charge.dtype == get_default_dtype()

    # charge as float
    mol.charge = 1.0
    assert isinstance(mol.charge, Tensor)
    assert mol.charge.dtype == get_default_dtype()

    # charge as Tensor
    mol.charge = torch.tensor(1.0)
    assert isinstance(mol.charge, Tensor)
    assert mol.charge.dtype == get_default_dtype()

    mol.charge = "1"
    assert isinstance(mol.charge, Tensor)
    assert mol.charge.dtype == get_default_dtype()

    # charge as wrong type
    with pytest.raises(ValueError):
        mol.charge = "a"  # type: ignore


def test_wrong_device() -> None:
    numbers = MockTensor(torch.randint(1, 118, (5,)))
    numbers.device = torch.device("cpu")
    numbers.dtype = torch.int64

    positions = MockTensor(torch.randn((5, 3)))
    positions.device = torch.device("cuda")
    positions.dtype = torch.float32

    with pytest.raises(DeviceError):
        Mol(numbers, positions)


def test_checks_device() -> None:
    numbers = MockTensor(torch.randint(1, 118, (5,)))
    numbers.device = torch.device("cpu")
    numbers.dtype = torch.int64

    positions = MockTensor(torch.randn((5, 3)))
    positions.device = torch.device("cpu")
    positions.dtype = torch.float32

    mol = Mol(numbers, positions)
    assert mol.checks() is None

    numbers.device = torch.device("cuda")
    mol._numbers = numbers
    with pytest.raises(DeviceError):
        mol.checks()


def test_checks_dtype() -> None:
    numbers = MockTensor(torch.randint(1, 118, (5,)))
    numbers.device = torch.device("cpu")
    numbers.dtype = torch.int64

    positions = MockTensor(torch.randn((5, 3)))
    positions.device = torch.device("cpu")
    positions.dtype = torch.float32

    mol = Mol(numbers, positions)
    assert mol.checks() is None

    numbers.dtype = torch.float32
    mol._numbers = numbers
    with pytest.raises(DtypeError):
        mol.checks()
