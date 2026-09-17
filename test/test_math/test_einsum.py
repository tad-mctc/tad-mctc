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
Test Einstein summations.
"""

from __future__ import annotations

import importlib
from unittest import mock

import pytest
import torch

from tad_mctc import math


def test_functions_existence() -> None:
    assert callable(math.einsum)
    assert callable(math.einsum_greedy)
    assert callable(math.einsum_optimal)


# These tests are pointless if `opt_einsum` is not installed. But they should
# pass anyway as they are just wrappers around the `torch.einsum` function.
def test_optimization_flags() -> None:
    operands = (torch.rand(2, 3), torch.rand(3, 4))
    equation = "ij,jk->ik"

    ref = torch.einsum(equation, *operands)
    assert pytest.approx(ref.cpu()) == math.einsum(equation, *operands).cpu()

    e_auto = math.einsum(equation, *operands, optimize="auto")
    assert pytest.approx(ref.cpu()) == e_auto.cpu()

    e_greedy = math.einsum_greedy(equation, *operands)
    assert pytest.approx(ref.cpu()) == e_greedy.cpu()
    e_greedy_2 = math.einsum(equation, *operands, optimize="greedy")
    assert pytest.approx(ref.cpu()) == e_greedy_2.cpu()

    e_optimal = math.einsum_optimal(equation, *operands)
    assert pytest.approx(ref.cpu()) == e_optimal.cpu()
    e_optimal_2 = math.einsum(equation, *operands, optimize="optimal")
    assert pytest.approx(ref.cpu()) == e_optimal_2.cpu()


def test_eager_still_uses_opt_einsum() -> None:
    """Eager-mode calls must keep going through `opt_einsum.contract`; only
    the compiling path in `_torch_einsum` is allowed to bypass it."""
    # `tad_mctc.math`'s `__init__` re-exports the `einsum` *function* under
    # the same name as this *submodule*, shadowing it as a package
    # attribute. `importlib` sidesteps that and returns the actual module.
    einsum_module = importlib.import_module("tad_mctc.math.einsum")

    if not hasattr(einsum_module, "contract"):
        pytest.skip("opt_einsum is not installed")

    operands = (torch.rand(2, 3), torch.rand(3, 4))
    ref = torch.einsum("ij,jk->ik", *operands)

    with mock.patch.object(
        einsum_module, "contract", wraps=einsum_module.contract
    ) as spy:
        result = math.einsum("ij,jk->ik", *operands)

    spy.assert_called_once()
    assert pytest.approx(ref.cpu()) == result.cpu()


def test_torch_compile_fullgraph_matches_eager() -> None:
    """`torch.compile(fullgraph=True)` must reproduce the eager result even
    when `opt_einsum` is installed: Dynamo cannot trace
    `opt_einsum.contract`'s internals ("Dynamo does not know how to trace
    method `__setitem__` of class `list`"), so `_torch_einsum` dispatches to
    `torch.einsum` while compiling instead."""
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile is not available")

    torch._dynamo.reset()

    operands = (
        torch.rand(2, 3, dtype=torch.float64),
        torch.rand(3, 4, dtype=torch.float64),
    )

    def f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return math.einsum("ij,jk->ik", x, y)

    compiled = torch.compile(f, fullgraph=True, dynamic=False)

    eager_value = f(*operands)
    compiled_value = compiled(*operands)

    assert pytest.approx(eager_value.cpu(), abs=1e-12) == compiled_value.cpu()
