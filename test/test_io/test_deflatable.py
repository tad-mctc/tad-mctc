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
Test the check for deflating (padding clash).
"""

from pathlib import Path

import pytest
import torch

from tad_mctc.exceptions import MoleculeError, MoleculeWarning
from tad_mctc.io import read
from tad_mctc.units.length import AA2AU


def test_read_atom_exception() -> None:
    """
    Read a single atom placed at [0.0, 0.0, 0.0].
    """
    p = Path(__file__).parent.resolve() / "files" / "atom.xyz"

    with pytest.raises(MoleculeError):
        read.read(p, raise_padding_exception=True)


def test_read_atom_warning() -> None:
    """
    Read a single atom placed at [0.0, 0.0, 0.0].
    """
    p = Path(__file__).parent.resolve() / "files" / "atom.xyz"

    with pytest.warns(MoleculeWarning):
        read.read(p, raise_padding_exception=False)


def test_read_atom() -> None:
    """
    Read a single atom placed at [0.0, 0.0, 0.0]. The reader should modify the
    x-coordinate to avoid a clash with zero-padding.
    """
    p = Path(__file__).parent.resolve() / "files" / "atom.xyz"
    numbers, positions = read.read(
        p, raise_padding_exception=False, shift_for_last=True, shift_value=1.0
    )

    ref_numbers = torch.tensor([1])
    ref_positions = torch.tensor([[1.0, 1.0, 1.0]])

    assert (ref_numbers == numbers).all()
    assert pytest.approx(ref_positions.cpu()) == positions.cpu()


def test_read_atom2() -> None:
    """
    Read a single atom placed at [0.3, 0.3, 0.3].
    """
    p = Path(__file__).parent.resolve() / "files" / "atom2.xyz"
    numbers, positions = read.read(p)

    ref_numbers = torch.tensor([2])
    ref_positions = torch.tensor([[0.3, 0.3, 0.3]]) * AA2AU

    assert (ref_numbers == numbers).all()
    assert pytest.approx(ref_positions.cpu()) == positions.cpu()


def test_read_fail_last_zero() -> None:
    """
    Read a molecule where the last atom placed at [0.0, 0.0, 0.0]. This would
    clash with zero-padding; hence, we immediately throw an error.
    """
    p = Path(__file__).parent.resolve() / "fail" / "lastzero.xyz"
    with pytest.raises(MoleculeError):
        read.read(p, raise_padding_exception=True)
