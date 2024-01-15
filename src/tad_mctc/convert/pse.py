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
