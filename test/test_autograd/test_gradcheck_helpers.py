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
Test the gradcheck helpers `dgradgradgradcheck`, `jacfwd_matches_jacrev`,
`no_vmap_fallback` and `positions_gradchecker`.

Each helper is run on a correct function (`torch.sin`) and on a function with
a deliberately broken custom derivative.
"""

from __future__ import annotations

from typing import Any, cast

import pytest
import torch
from torch.autograd.gradcheck import GradcheckError

from tad_mctc.autograd import (
    dgradcheck,
    dgradgradcheck,
    dgradgradgradcheck,
    jacfwd_matches_jacrev,
    no_vmap_fallback,
    positions_gradchecker,
)
from tad_mctc.io.structure import Structure

from ..conftest import DEVICE


def _x() -> torch.Tensor:
    gen = torch.Generator().manual_seed(0)
    x = torch.rand(3, generator=gen, dtype=torch.float64, device="cpu")
    return x.to(DEVICE).requires_grad_()


class WrongFirstDerivative(torch.autograd.Function):
    """``x**2`` whose backward uses ``3x`` instead of ``2x``."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return x**2

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        (g,) = grad_outputs
        (x,) = ctx.saved_tensors
        return 3 * x * g


class _WrongMul(torch.autograd.Function):
    """``a * b`` whose derivative with respect to ``a`` is 3x too large."""

    @staticmethod
    def forward(ctx: Any, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(a, b)
        return a * b

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        (g,) = grad_outputs
        a, b = ctx.saved_tensors
        return 3 * g * b, g * a


class WrongSecondDerivative(torch.autograd.Function):
    """``x**2`` with a correct first derivative, but a backward that is
    itself differentiated wrongly."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return x**2

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        (g,) = grad_outputs
        (x,) = ctx.saved_tensors
        return _WrongMul.apply(2 * x, g)


class _Triple(torch.autograd.Function):
    """``3 * x**2`` with a correct first derivative, which is itself
    differentiated wrongly."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return 3 * x**2

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        (g,) = grad_outputs
        (x,) = ctx.saved_tensors
        return _WrongMul.apply(6 * x, g)


class WrongThirdDerivative(torch.autograd.Function):
    """``x**3`` with correct first and second derivatives and a wrong
    third one."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return x**3

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        (g,) = grad_outputs
        (x,) = ctx.saved_tensors
        return g * _Triple.apply(x)


class WrongJvp(torch.autograd.Function):
    """``x**2`` with a correct backward and a wrong forward-mode rule."""

    generate_vmap_rule = True

    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        return x**2

    @staticmethod
    def setup_context(ctx: Any, inputs: Any, output: Any) -> None:
        ctx.save_for_backward(inputs[0])
        ctx.save_for_forward(inputs[0])

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        (g,) = grad_outputs
        (x,) = ctx.saved_tensors
        return 2 * x * g

    @staticmethod
    def jvp(ctx: Any, *grad_inputs: Any) -> Any:
        (gx,) = grad_inputs
        (x,) = ctx.saved_tensors
        return 3 * x * gx


def _wrong_first(x: torch.Tensor) -> torch.Tensor:
    return cast(torch.Tensor, WrongFirstDerivative.apply(x))


def _wrong_second(x: torch.Tensor) -> torch.Tensor:
    return cast(torch.Tensor, WrongSecondDerivative.apply(x))


def _wrong_third(x: torch.Tensor) -> torch.Tensor:
    return cast(torch.Tensor, WrongThirdDerivative.apply(x))


def _wrong_jvp(x: torch.Tensor) -> torch.Tensor:
    return cast(torch.Tensor, WrongJvp.apply(x))


@pytest.mark.grad
def test_dgradgradgradcheck_passes() -> None:
    assert dgradgradgradcheck(torch.sin, _x())


@pytest.mark.grad
def test_dgradgradgradcheck_fails_on_wrong_third_derivative() -> None:
    # correct up to second order, wrong in the third derivative
    assert dgradcheck(_wrong_third, _x())
    assert dgradgradcheck(_wrong_third, _x())
    with pytest.raises(GradcheckError):
        dgradgradgradcheck(_wrong_third, _x())


@pytest.mark.grad
def test_dgradgradcheck_fails_on_wrong_second_derivative() -> None:
    assert dgradcheck(_wrong_second, _x())
    with pytest.raises(GradcheckError):
        dgradgradcheck(_wrong_second, _x())


@pytest.mark.grad
def test_dgradcheck_fails_on_wrong_first_derivative() -> None:
    with pytest.raises(GradcheckError):
        dgradcheck(_wrong_first, _x())


def test_jacfwd_matches_jacrev_passes() -> None:
    assert jacfwd_matches_jacrev(torch.sin, _x().detach())


def test_jacfwd_matches_jacrev_fails_on_wrong_jvp() -> None:
    assert not jacfwd_matches_jacrev(_wrong_jvp, _x().detach())


def test_no_vmap_fallback_passes_for_batched_operation() -> None:
    x = torch.rand(4, 3, dtype=torch.float64)
    with no_vmap_fallback():
        out = torch.func.vmap(torch.sin)(x)
    assert torch.equal(out, x.sin())


def test_no_vmap_fallback_raises_without_batching_rule() -> None:
    x = torch.rand(4, 5, dtype=torch.float64)

    # `histc` has no batching rule: it silently loops without the context
    with no_vmap_fallback():
        with pytest.raises(RuntimeError, match="vmap fallback"):
            torch.func.vmap(torch.histc)(x)

    # vmap works again afterwards (and warns about the slow fallback)
    with pytest.warns(UserWarning, match="batching rule"):
        out = torch.func.vmap(torch.histc)(x)
    assert out.shape == (4, 100)


def test_no_vmap_fallback_restores_state() -> None:
    functorch: Any = getattr(torch._C, "_functorch")
    before = functorch._is_vmap_fallback_enabled()
    with no_vmap_fallback():
        assert not functorch._is_vmap_fallback_enabled()
    assert functorch._is_vmap_fallback_enabled() == before


def test_positions_gradchecker() -> None:
    structure = Structure(
        numbers=torch.tensor([1, 1]),
        positions=torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], dtype=torch.float64
        ),
        charge=torch.tensor(1.0, dtype=torch.float64),
    )

    def function(s: Structure) -> torch.Tensor:
        assert s.charge is not None
        return s.charge * torch.linalg.vector_norm(s.positions, dim=-1)

    func, positions = positions_gradchecker(function, structure)

    assert positions.requires_grad
    assert positions is not structure.positions
    assert not structure.positions.requires_grad
    assert torch.equal(func(positions), function(structure))
    assert dgradcheck(func, positions)
