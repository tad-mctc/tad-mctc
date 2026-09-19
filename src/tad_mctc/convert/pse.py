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

from typing import overload

import torch

from ..data import pse
from ..typing import Tensor

__all__ = ["symbol_to_number", "number_to_symbol"]

_LC_S2Z = {symbol.lower(): number for symbol, number in pse.S2Z.items()}


def _resolve_symbol(symbol: str) -> int | None:
    """
    Resolve a single element symbol the way mctc-lib's ``to_number`` does
    (``src/mctc/io/symbols.f90``): only the first two alphabetic characters
    of the (4-character-truncated) token are kept -- skipping digits, ``*``,
    and other decoration -- lowercased, and looked up, with ``d``/``t``
    (deuterium/tritium) special-cased to hydrogen. Returns ``None`` if the
    symbol cannot be resolved (unknown element or junk), leaving the choice
    of what to do about that (raise, fall back to parsing the token as a raw
    atomic number, ...) to the caller.

    Note: ``pse.S2Z`` also carries ``"X": 0`` as this package's own
    dummy/padding-atom convention (see ``batch``/masking code) -- mctc-lib
    has no such entry and its ``to_number`` never returns a "valid" 0, so a
    0 lookup here is treated the same as an unresolved symbol, not forwarded
    to the caller.

    mctc-lib's callers all first truncate the captured token to
    ``symbol_length`` (4) raw characters before calling ``to_number``
    (``token%last = min(token%last, token%first + symbol_length - 1)``), so
    a token like ``"***As"`` becomes ``"***A"`` -- losing the ``s`` -- before
    the 2-letter scan below ever sees it. Replicated here so a junk-prefixed
    symbol fails to resolve the same way.
    """
    symbol = symbol[:4]
    letters = ""
    for char in symbol:
        if len(letters) >= 2:
            break
        if char.isalpha() and char.isascii():
            letters += char.lower()

    if letters in ("d", "t"):
        return 1

    number = _LC_S2Z.get(letters)
    return number if number else None


@overload
def symbol_to_number(symbols: str) -> int | None: ...


@overload
def symbol_to_number(symbols: list[str] | tuple[str, ...]) -> Tensor: ...


def symbol_to_number(
    symbols: str | list[str] | tuple[str, ...],
) -> int | None | Tensor:
    """
    Obtain atomic numbers from element symbols, using mctc-lib's quirky
    ``to_number`` resolution (see :func:`_resolve_symbol`): decorations
    (digits, ``*``, isotope prefixes, ...) are stripped and ``d``/``t`` are
    resolved to hydrogen.

    Parameters
    ----------
    symbols : str | list[str] | tuple[str, ...]
        A single element symbol, or a list of element symbols.

    Returns
    -------
    int | None | Tensor
        For a single symbol, the atomic number, or ``None`` if it cannot be
        resolved. For a sequence of symbols, the atomic numbers
        corresponding to the given element symbols.

    Raises
    ------
    ValueError
        If a symbol in a sequence cannot be resolved to an atomic number.
    """
    if isinstance(symbols, str):
        return _resolve_symbol(symbols)

    numbers: list[int] = []
    for s in symbols:
        number = _resolve_symbol(s)
        if number is None:
            raise ValueError(f"Unknown element symbol {s!r}.")
        numbers.append(number)

    return torch.flatten(torch.tensor(numbers))


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
    return [pse.Z2S[n] for n in torch.atleast_1d(numbers).cpu().tolist()]
