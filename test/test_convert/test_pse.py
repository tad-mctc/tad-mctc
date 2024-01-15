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
Test interconversion of atomic symbols and numbers.
"""
import torch

from tad_mctc import convert

from ..conftest import DEVICE


def test_symbol_to_number() -> None:
    symbols = ["H", "He", "C", "C", "Eu"]
    numbers = torch.tensor([1, 2, 6, 6, 63], device=DEVICE)

    assert (convert.symbol_to_number(symbols) == numbers).all()


def test_number_to_symbol() -> None:
    symbols = ["H", "He", "C", "C", "Eu"]
    numbers = torch.tensor([1, 2, 6, 6, 63], device=DEVICE)

    assert convert.number_to_symbol(numbers) == symbols
