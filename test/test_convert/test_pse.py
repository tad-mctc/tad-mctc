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
