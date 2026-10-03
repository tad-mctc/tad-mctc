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
Transform conformance of `tad_mctc.math`: `einsum`, `einsum_greedy` and
`einsum_optimal`, on the three-operand contraction ``ij,jk,kl->il``.

Exclusions: none. Under `torch.compile` the wrappers fall back to
`torch.einsum`, so the compile check tests that fallback.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from torch.func import vmap

from tad_mctc.autograd import (
    dgradcheck,
    dgradgradcheck,
    dgradgradgradcheck,
    jacfwd_matches_jacrev,
    no_vmap_fallback,
)
from tad_mctc.math import einsum, einsum_greedy, einsum_optimal

from ..conftest import DEVICE
from ..utils import DYNAMO_SUPPORTED, DYNAMO_UNSUPPORTED_REASON

EQUATION = "ij,jk,kl->il"

FUNCTIONS = [
    pytest.param(einsum, id="einsum"),
    pytest.param(einsum_greedy, id="einsum_greedy"),
    pytest.param(einsum_optimal, id="einsum_optimal"),
]


def _operands(batch: int | None = None) -> tuple[torch.Tensor, ...]:
    gen = torch.Generator().manual_seed(0)
    lead = () if batch is None else (batch,)
    shapes = [(2, 3), (3, 4), (4, 2)]
    return tuple(
        torch.rand(*lead, *shape, generator=gen, dtype=torch.float64).to(DEVICE)
        for shape in shapes
    )


@pytest.mark.parametrize("f", FUNCTIONS)
def test_vmap_matches_loop(f: Callable[..., torch.Tensor]) -> None:
    a, b, c = _operands(batch=2)

    def contract(
        a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
    ) -> torch.Tensor:
        return f(EQUATION, a, b, c)

    with no_vmap_fallback():
        batched = vmap(contract)(a, b, c)

    looped = torch.stack([contract(a[i], b[i], c[i]) for i in range(2)])
    assert torch.allclose(batched, looped)


@pytest.mark.grad
@pytest.mark.parametrize("f", FUNCTIONS)
def test_gradcheck_orders(f: Callable[..., torch.Tensor]) -> None:
    def contract(
        a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
    ) -> torch.Tensor:
        return f(EQUATION, a, b, c)

    def inputs() -> tuple[torch.Tensor, ...]:
        return tuple(x.requires_grad_() for x in _operands())

    assert dgradcheck(contract, inputs())
    assert dgradgradcheck(contract, inputs())
    assert dgradgradgradcheck(contract, inputs())


@pytest.mark.parametrize("f", FUNCTIONS)
def test_forward_matches_reverse(f: Callable[..., torch.Tensor]) -> None:
    _, b, c = _operands()

    def contract(a: torch.Tensor) -> torch.Tensor:
        return f(EQUATION, a, b, c)

    assert jacfwd_matches_jacrev(contract, _operands()[0])


@pytest.mark.skipif(not DYNAMO_SUPPORTED, reason=DYNAMO_UNSUPPORTED_REASON)
@pytest.mark.parametrize("f", FUNCTIONS)
def test_compile_matches_eager(f: Callable[..., torch.Tensor]) -> None:
    a, b, c = _operands()

    def contract(
        a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
    ) -> torch.Tensor:
        return f(EQUATION, a, b, c)

    compiled = torch.compile(contract, fullgraph=True)
    assert torch.allclose(compiled(a, b, c), contract(a, b, c))
