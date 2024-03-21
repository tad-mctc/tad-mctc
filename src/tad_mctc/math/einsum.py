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
Math: Einsum
============

This module provides a wrapper for the `einsum` function from `opt_einsum`
package. If `opt_einsum` is not installed, it falls back to the `torch.einsum`.
"""

from __future__ import annotations

import torch
from ..typing import Any, Tensor, _wraps


__all__ = [
    "einsum",
    "einsum_greedy",
    "einsum_optimal",
]


try:
    from functools import partial

    from opt_einsum import contract

    @_wraps(contract)
    def _torch_einsum(*args: Any, optimize: Any = "greedy") -> Tensor:
        f = partial(contract, backend="torch", optimize=optimize)
        return f(*args)  # type: ignore

    @_wraps(contract)
    def einsum_greedy(*args: Any) -> Tensor:
        return partial(_torch_einsum, optimize="greedy")(*args)

    @_wraps(contract)
    def einsum_optimal(*args: Any) -> Tensor:
        return partial(_torch_einsum, optimize="optimal")(*args)

    @_wraps(contract)
    def einsum(*args: Any, optimize: Any = "greedy") -> Tensor:
        if optimize == "greedy":
            return einsum_greedy(*args)

        if optimize == "optimal":
            return einsum_optimal(*args)

        return _torch_einsum(*args, optimize=optimize)

except ImportError:

    @_wraps(torch.einsum)
    def einsum(*args: Any, optimize: Any = None) -> Tensor:
        if optimize is not None:
            from warnings import warn

            warn("Optimization not supported without 'opt_einsum' package.")

        return torch.einsum(*args)

    einsum_optimal = einsum_greedy = einsum
