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
Test interconversion of atomic symbols and numbers.
"""

from __future__ import annotations

import pytest
import torch

from tad_mctc import convert

from ..conftest import DEVICE


def test_symbol_to_number() -> None:
    symbols = ["H", "He", "C", "C", "Eu"]
    numbers = torch.tensor([1, 2, 6, 6, 63], device=DEVICE)

    assert (convert.symbol_to_number(symbols) == numbers).all()


def test_symbol_to_number_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown element symbol"):
        convert.symbol_to_number(["H", "Xx"])


@pytest.mark.parametrize(
    "symbol,number",
    [
        ("H", 1),
        ("D", 1),  # deuterium special-cased to hydrogen
        ("T", 1),  # tritium special-cased to hydrogen
        ("C*", 6),  # decoration is stripped
        ("1H", 1),  # isotope mass-number prefix is stripped
        ("Xx", None),  # unknown element
        ("X", None),  # this package's own dummy/padding symbol, not real
    ],
)
def test_symbol_to_number_single(symbol: str, number: int | None) -> None:
    assert convert.symbol_to_number(symbol) == number


def test_number_to_symbol() -> None:
    symbols = ["H", "He", "C", "C", "Eu"]
    numbers = torch.tensor([1, 2, 6, 6, 63], device=DEVICE)

    assert convert.number_to_symbol(numbers) == symbols
