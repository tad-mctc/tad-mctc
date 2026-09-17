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
Math: Einsum
============

This module provides a wrapper for the `einsum` function from `opt_einsum`
package. If `opt_einsum` is not installed, it falls back to the `torch.einsum`.
"""

from __future__ import annotations

from typing import Any

import torch

from ..tools import is_compiling
from ..typing import Tensor, _wraps

__all__ = [
    "einsum",
    "einsum_greedy",
    "einsum_optimal",
]


try:
    from functools import partial

    from opt_einsum import contract  # type: ignore[import-untyped]

    @_wraps(contract)
    def _torch_einsum(
        *args: Any, optimize: Any = "greedy"
    ) -> Tensor:  # pragma: no cover
        # Dynamo cannot trace `opt_einsum.contract`'s internals (it builds
        # up its contraction path with plain Python list mutation), so
        # `torch.compile(fullgraph=True)` fails on every call site of this
        # wrapper whenever `opt_einsum` is installed. Eager execution keeps
        # `opt_einsum`'s contraction-order optimization; under compilation,
        # Inductor does its own fusion, so that optimization buys little
        # there anyway.
        if is_compiling():
            return torch.einsum(*args)

        f = partial(contract, backend="torch", optimize=optimize)
        return f(*args)

    @_wraps(contract)
    def einsum_greedy(*args: Any) -> Tensor:
        return _torch_einsum(*args, optimize="greedy")

    @_wraps(contract)
    def einsum_optimal(*args: Any) -> Tensor:
        return _torch_einsum(*args, optimize="optimal")

    @_wraps(contract)
    def einsum(*args: Any, optimize: Any = "greedy") -> Tensor:
        if optimize == "greedy":
            return einsum_greedy(*args)

        if optimize == "optimal":
            return einsum_optimal(*args)

        return _torch_einsum(*args, optimize=optimize)

except ImportError:  # pragma: no cover

    @_wraps(torch.einsum)
    def einsum(*args: Any, optimize: Any = None) -> Tensor:
        if optimize is not None:
            # pylint: disable=import-outside-toplevel
            from warnings import warn

            warn("Optimization not supported without 'opt_einsum' package.")

        return torch.einsum(*args)

    einsum_optimal = einsum_greedy = einsum
