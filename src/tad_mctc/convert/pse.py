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
Conversion: PSE
===============

Mapping of the Periodic Systems of Elements (PSE) from atomic number to symbols
and vice versa.
"""

from __future__ import annotations

import torch

from ..data import pse
from ..typing import Sequence, Tensor

__all__ = ["symbol_to_number", "number_to_symbol"]


def symbol_to_number(symbols: Sequence[str]) -> Tensor:
    """
    Obtain atomic numbers from element symbols.

    Parameters
    ----------
    symbols : list[str]
        List of element symbols.

    Returns
    -------
    Tensor
        Atomic numbers corresponding to the given element symbols.
    """
    return torch.flatten(torch.tensor([pse.S2Z[s.title()] for s in symbols]))


def number_to_symbol(numbers: Tensor) -> list[str]:
    """
    Obtain element symbols from atomic numbers.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers.

    Returns
    -------
    list[str]
        Element symbols corresponding to the given atomic numbers.
    """
    return [pse.Z2S[n] for n in numbers.cpu().tolist()]
