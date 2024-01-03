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
        read.read_from_path(p, raise_padding_exception=True)


def test_read_atom_warning() -> None:
    """
    Read a single atom placed at [0.0, 0.0, 0.0].
    """
    p = Path(__file__).parent.resolve() / "files" / "atom.xyz"

    with pytest.warns(MoleculeWarning):
        read.read_from_path(p, raise_padding_exception=False)


def test_read_atom() -> None:
    """
    Read a single atom placed at [0.0, 0.0, 0.0]. The reader should modify the
    x-coordinate to avoid a clash with zero-padding.
    """
    p = Path(__file__).parent.resolve() / "files" / "atom.xyz"
    numbers, positions = read.read_from_path(
        p, raise_padding_exception=False, shift_for_last=True, shift_value=1.0
    )

    ref_numbers = torch.tensor([1])
    ref_positions = torch.tensor([[1.0, 1.0, 1.0]])

    assert (ref_numbers == numbers).all()
    assert pytest.approx(ref_positions) == positions


def test_read_atom2() -> None:
    """
    Read a single atom placed at [0.3, 0.3, 0.3].
    """
    p = Path(__file__).parent.resolve() / "files" / "atom2.xyz"
    numbers, positions = read.read_from_path(p)

    ref_numbers = torch.tensor([2])
    ref_positions = torch.tensor([[0.3, 0.3, 0.3]]) * AA2AU

    assert (ref_numbers == numbers).all()
    assert pytest.approx(ref_positions) == positions


def test_read_fail_last_zero() -> None:
    """
    Read a molecule where the last atom placed at [0.0, 0.0, 0.0]. This would
    clash with zero-padding; hence, we immediately throw an error.
    """
    p = Path(__file__).parent.resolve() / "fail" / "lastzero.xyz"
    with pytest.raises(MoleculeError):
        read.read_from_path(p, raise_padding_exception=True)
