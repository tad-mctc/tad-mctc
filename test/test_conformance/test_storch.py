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
Transform conformance of `tad_mctc.storch`: `safe_divide`, `safe_sqrt`,
`safe_pow`, `safe_reciprocal` and `cdist`.

Every function is checked for `vmap` (without fallback), reverse mode up to
third order, forward mode, `torch.compile(fullgraph=True)` and for finite
derivatives up to third order at the masked point (0).

Exclusions: none. The gradient checks run at regular (non-zero) points,
because a finite-difference check straddling the clamp at 0 is meaningless;
the masked point is covered by the finiteness checks.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

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
from tad_mctc.storch import (
    cdist,
    safe_divide,
    safe_pow,
    safe_reciprocal,
    safe_sqrt,
)

from ..conftest import DEVICE
from ..utils import (
    DYNAMO_SUPPORTED,
    DYNAMO_UNSUPPORTED_REASON,
    compile_fullgraph,
)

DD = {"device": DEVICE, "dtype": torch.float64}

# regular points and a vector that contains the masked point (0)
REGULAR = (0.3, 0.7, 1.2, 2.0)
MASKED = (0.0, 0.5, 0.0, 1.5)


def _t(values: tuple[float, ...], grad: bool = False) -> torch.Tensor:
    return torch.tensor(values, **DD, requires_grad=grad)  # type: ignore[arg-type]


def _points(n: int = 4) -> torch.Tensor:
    gen = torch.Generator().manual_seed(0)
    return torch.rand(n, 3, generator=gen, dtype=torch.float64).to(DEVICE)


# -- functions of one tensor ----------------------------------------------


def _divide(x: torch.Tensor) -> torch.Tensor:
    return safe_divide(torch.ones_like(x), x)


def _sqrt(x: torch.Tensor) -> torch.Tensor:
    return safe_sqrt(x)


def _pow(x: torch.Tensor) -> torch.Tensor:
    return safe_pow(x, 1.5)


def _pow_tensor(x: torch.Tensor) -> torch.Tensor:
    return safe_pow(x, torch.tensor(1.5, **DD))  # type: ignore[arg-type]


def _reciprocal(x: torch.Tensor) -> torch.Tensor:
    return safe_reciprocal(x)


def _cdist(x: torch.Tensor) -> torch.Tensor:
    return cdist(x, x * 0.5 + 0.1)


def _cdist_self(x: torch.Tensor) -> torch.Tensor:
    return cdist(x)


def _regular(f: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    return _t(REGULAR) if f not in (_cdist, _cdist_self) else _points()


def _masked(f: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    if f in (_cdist, _cdist_self):
        # duplicated points: zero distances
        x = _points()
        x[1] = x[0]
        return x
    return _t(MASKED)


FUNCTIONS = [
    pytest.param(_divide, id="safe_divide"),
    pytest.param(_sqrt, id="safe_sqrt"),
    pytest.param(_pow, id="safe_pow"),
    pytest.param(_pow_tensor, id="safe_pow-tensor-exponent"),
    pytest.param(_reciprocal, id="safe_reciprocal"),
    pytest.param(_cdist, id="cdist"),
    pytest.param(_cdist_self, id="cdist-self"),
]


def _derivatives(
    f: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor
) -> list[torch.Tensor]:
    """First, second and third derivative of ``f(x).sum()``."""
    x = x.clone().requires_grad_()
    out = f(x).sum()
    results = []
    for _ in range(3):
        (grad,) = torch.autograd.grad(out, x, create_graph=True)
        results.append(grad)
        out = grad.sum()
    return results


@pytest.mark.parametrize("f", FUNCTIONS)
def test_vmap_matches_loop(f: Callable[[torch.Tensor], torch.Tensor]) -> None:
    batch = torch.stack([_regular(f), _masked(f)])

    with no_vmap_fallback():
        batched = vmap(f)(batch)

    looped = torch.stack([f(x) for x in batch])
    assert torch.allclose(batched, looped)


@pytest.mark.grad
@pytest.mark.parametrize("f", FUNCTIONS)
def test_gradcheck_orders(f: Callable[[torch.Tensor], torch.Tensor]) -> None:
    assert dgradcheck(f, _regular(f).requires_grad_())
    assert dgradgradcheck(f, _regular(f).requires_grad_())
    assert dgradgradgradcheck(f, _regular(f).requires_grad_())


@pytest.mark.parametrize("f", FUNCTIONS)
def test_forward_matches_reverse(
    f: Callable[[torch.Tensor], torch.Tensor],
) -> None:
    assert jacfwd_matches_jacrev(f, _regular(f))


@pytest.mark.parametrize("f", FUNCTIONS)
def test_derivatives_finite_at_masked_point(
    f: Callable[[torch.Tensor], torch.Tensor],
) -> None:
    for derivative in _derivatives(f, _masked(f)):
        assert torch.isfinite(derivative).all()


@pytest.mark.skipif(not DYNAMO_SUPPORTED, reason=DYNAMO_UNSUPPORTED_REASON)
@pytest.mark.parametrize("f", FUNCTIONS)
def test_compile_matches_eager(
    f: Callable[[torch.Tensor], torch.Tensor],
) -> None:
    compiled = compile_fullgraph(f)
    for x in (_regular(f), _masked(f)):
        assert torch.allclose(compiled(x), f(x))


# -- functions of two tensors ---------------------------------------------


def test_divide_two_inputs() -> None:
    def inputs() -> tuple[torch.Tensor, torch.Tensor]:
        return _t(REGULAR, True), _t((1.5, 0.5, 2.5, 1.0), True)

    # the checks detach their inputs, so every check gets fresh ones
    assert dgradcheck(safe_divide, inputs())
    assert dgradgradcheck(safe_divide, inputs())
    assert dgradgradgradcheck(safe_divide, inputs())


def test_pow_exponent_one_keeps_unit_derivative_at_zero() -> None:
    x = _t(MASKED).requires_grad_()

    for exponent in (1.0, torch.tensor(1.0, **DD)):  # type: ignore[arg-type]
        (grad,) = torch.autograd.grad(safe_pow(x, exponent).sum(), x)
        assert torch.equal(grad, torch.ones_like(x))


def _second_derivative(
    f: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    x = _t(MASKED).requires_grad_()
    (grad,) = torch.autograd.grad(f(x).sum(), x, create_graph=True)
    (grad2,) = torch.autograd.grad(grad.sum(), x)
    return grad2


def test_pow_whole_exponent_second_derivative_at_zero() -> None:
    for exponent in (2.0, torch.tensor(2.0, **DD)):  # type: ignore[arg-type]
        grad2 = _second_derivative(partial(safe_pow, exponent=exponent))
        assert torch.equal(grad2, torch.full_like(grad2, 2.0))


def test_pow_fractional_exponent_derivatives_finite_at_zero() -> None:
    for exponent in (1.5, torch.tensor(1.5, **DD)):  # type: ignore[arg-type]
        for derivative in _derivatives(
            partial(safe_pow, exponent=exponent), _t(MASKED)
        ):
            assert torch.isfinite(derivative).all()
