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
"""Count real operator work across implicit numerical batching boundaries."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable
from typing import Any

import pytest
import torch
from torch.func import jacfwd, jacrev, vmap

from tad_mctc.autograd import no_vmap_fallback
from tad_mctc.storch.linsolve import (
    STATUS_CONVERGED,
    ConstrainedSolveResult,
    LinearSolveResult,
    constrained_solve,
    implicit_solve,
    require_convergence,
)
from tad_mctc.tools.testing import requires_compile
from tad_mctc.typing import Tensor

from ..conftest import DEVICE


@pytest.mark.parametrize("method", ["cg", "bicgstab", "gmres", "projected_cg"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("fast_factor", [None, 1.0, -1.0])
def test_mapped_operator_work(
    method: str, dtype: torch.dtype, fast_factor: float | None
) -> None:
    """Mapped solves do the work of an explicit batch, including fallback."""
    diagonal = torch.tensor(
        [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 3.0, 4.0]], device=DEVICE, dtype=dtype
    )
    rhs = torch.ones_like(diagonal)
    calls: Counter[str] = Counter()

    def matvec(vector: Tensor, values: Tensor) -> Tensor:
        calls["exact"] += 1
        return vector * values

    def fast_matvec(vector: Tensor, values: Tensor) -> Tensor:
        calls["fast"] += 1

        # One lane needs an exact fallback; the accepted lane must keep
        # its solution and report no fallback iterations.
        factor = torch.where(values[..., :1] == 1.0, 1.0, fast_factor)
        return factor * vector * values

    def precondition(vector: Tensor, values: Tensor) -> Tensor:
        calls["preconditioner"] += 1
        return vector

    def solve(
        values: Tensor, right_hand_side: Tensor
    ) -> LinearSolveResult | ConstrainedSolveResult:
        options = dict(
            preconditioner=precondition,
            maxiter=40,
            fast_matvec=None if fast_factor is None else fast_matvec,
            fallback=True,
        )
        if method == "projected_cg":
            constraint = values.new_ones(1, 4)
            total = values.new_tensor([0.3])
            return constrained_solve(
                matvec, right_hand_side, constraint, total, values, **options
            )
        return implicit_solve(
            matvec, right_hand_side, values, method=method, **options
        )

    reference = solve(diagonal, rhs)
    reference_calls = calls.copy()
    calls.clear()

    actual = vmap(solve)(diagonal, rhs)

    for result, expected in zip(actual, reference):
        torch.testing.assert_close(result, expected, rtol=1e-12, atol=1e-12)
    assert calls == reference_calls
    assert (actual.status == STATUS_CONVERGED).all()
    if fast_factor == -1.0:
        assert actual.fallback_iterations[0] == 0
        assert actual.fallback_iterations[1] > 0
        assert actual.refinement_status[0] == STATUS_CONVERGED
        assert actual.refinement_status[1] != STATUS_CONVERGED
    assert reference_calls["exact"] + reference_calls["fast"] < 40


@pytest.mark.parametrize("vdims", [1, 2])
def test_nested_nonleading_maps(vdims: int) -> None:
    """Nested maps preserve axes, broadcast parameters and mapped guesses."""
    dd = {"device": DEVICE, "dtype": torch.double}
    values = torch.tensor([1.0, 2.0, 3.0], **dd)
    shape = (2, 3, 2, 3) if vdims == 2 else (2, 3, 2)
    rhs = torch.arange(math.prod(shape), **dd).reshape(shape) + 1
    initial = rhs * 0.1
    calls: Counter[str] = Counter()

    def matvec(vector: Tensor, diagonal: Tensor, unused: str) -> Tensor:
        calls["operator"] += 1
        return vector * diagonal

    def solve(right_hand_side: Tensor, guess: Tensor) -> Tensor:
        return implicit_solve(
            matvec,
            right_hand_side,
            values,
            "static",
            x0=guess,
            vdims=vdims,
            maxiter=40,
        ).solution

    reference = solve(rhs.movedim(2, 1), initial.movedim(2, 1)).movedim(1, 2)
    reference_calls = calls.copy()
    calls.clear()

    # The inner map consumes a nonleading axis. Promote both map levels
    # without changing the order of the ordinary vector dimensions.
    inner = vmap(solve, in_dims=(1, 1), out_dims=1)
    nested = vmap(inner)
    with no_vmap_fallback():
        actual = nested(rhs, initial)

    torch.testing.assert_close(actual, reference, rtol=1e-12, atol=1e-12)
    assert calls == reference_calls


@pytest.mark.parametrize("transform", [jacrev, jacfwd])
@pytest.mark.parametrize("constrained", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_derivative_operator_work(
    transform: Callable[..., Any], constrained: bool, dtype: torch.dtype
) -> None:
    """Derivative solves stop early without differentiating iteration history."""
    dd = {"device": DEVICE, "dtype": dtype}
    rhs = torch.tensor([1.0, 2.0, 3.0, 4.0], **dd)
    parameter = torch.tensor(2.0, **dd)
    constraint = torch.ones(1, 4, **dd)
    value = torch.tensor([0.3], **dd)
    calls: Counter[str] = Counter()

    def matvec(vector: Tensor, scale: Tensor) -> Tensor:
        calls["operator"] += 1
        return scale * vector

    def solve(scale: Tensor) -> Tensor:
        if constrained:
            result = constrained_solve(
                matvec,
                rhs,
                constraint,
                value,
                scale,
                maxiter=60,
                fast_matvec=matvec,
                fallback=True,
            )
        else:
            result = implicit_solve(
                matvec,
                rhs,
                scale,
                maxiter=60,
                fast_matvec=matvec,
                fallback=True,
            )
        return result.solution

    def reference(scale: Tensor) -> Tensor:
        if constrained:
            return (rhs - rhs.mean()) / scale + value / rhs.numel()
        return rhs / scale

    differentiate = transform(solve)
    with no_vmap_fallback():
        actual = differentiate(parameter)

    expected = transform(reference)(parameter)
    tol = 2e-5 if dtype == torch.float32 else 1e-11
    torch.testing.assert_close(actual, expected, rtol=tol, atol=tol)
    assert calls["operator"] < 20


@pytest.mark.parametrize(
    "failure", ["indefinite", "singular", "limit", "nonfinite", "stagnation"]
)
@pytest.mark.parametrize("constrained", [False, True])
def test_failure_diagnostics(failure: str, constrained: bool) -> None:
    """Promotion preserves failure statuses, residual checks and valid lanes."""
    dd = {"device": DEVICE, "dtype": torch.double}
    diagonals = torch.tensor([[1.0, 1.0, 1.0, 1.0], [1.0, 2.0, 3.0, 4.0]], **dd)
    if failure == "stagnation":
        diagonals[1, 0] = 2.0
    if failure == "indefinite":
        diagonals[1] = -diagonals[1]
    if failure == "singular":
        diagonals[1] = 0.0
    if failure == "nonfinite":
        diagonals[1] = torch.nan
    rhs = torch.tensor([[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]], **dd)
    constraint = torch.ones(1, 4, **dd)
    value = torch.zeros(1, **dd)

    def matvec(vector: Tensor, diagonal: Tensor) -> Tensor:
        return diagonal * vector

    def fast(vector: Tensor, diagonal: Tensor) -> Tensor:
        factor = torch.where(diagonal[..., :1] == 1.0, 1.0, 0.2)
        return factor * vector * diagonal

    def solve(
        diagonal: Tensor,
        right_hand_side: Tensor,
        early_exit: bool | None = None,
    ) -> LinearSolveResult | ConstrainedSolveResult:
        options = dict(
            maxiter=1 if failure == "limit" else 40,
            early_exit=early_exit,
            fast_matvec=fast if failure == "stagnation" else None,
        )
        if constrained:
            return constrained_solve(
                matvec, right_hand_side, constraint, value, diagonal, **options
            )
        return implicit_solve(matvec, right_hand_side, diagonal, **options)

    actual = vmap(solve)(diagonals, rhs)
    reference = solve(diagonals, rhs, early_exit=False)

    for result, expected in zip(actual, reference):
        torch.testing.assert_close(
            result, expected, rtol=1e-12, atol=1e-12, equal_nan=True
        )
    assert actual.status[0] == STATUS_CONVERGED
    assert actual.status[1] != STATUS_CONVERGED


@pytest.mark.parametrize("transform", [jacrev, jacfwd])
@pytest.mark.parametrize("order", [1, 2, 3])
def test_failed_shared_parameter_derivatives(
    transform: Callable[..., Any], order: int
) -> None:
    """Failed active derivatives remain NaN, including zero higher orders."""
    dd = {"device": DEVICE, "dtype": torch.double}
    diagonals = torch.tensor([[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]], **dd)
    parameter = torch.tensor(2.0, **dd)

    def matvec(vector: Tensor, diagonal: Tensor) -> Tensor:
        return diagonal * vector

    def solve(diagonal: Tensor, scale: Tensor) -> Tensor:
        rhs = scale * torch.ones_like(diagonal)
        result = implicit_solve(matvec, rhs, diagonal, maxiter=40)
        return require_convergence(result)

    def shared(scale: Tensor) -> Tensor:
        return vmap(solve, in_dims=(0, None))(diagonals, scale)

    # Select accepted outputs before forming higher derivatives with a
    # shared input. A joint failed Hessian can poison ordinary arithmetic
    # outside the convergence mask; that remains explicitly unsupported.
    def accepted(scale: Tensor) -> Tensor:
        return shared(scale)[0]

    def rejected(scale: Tensor) -> Tensor:
        return shared(scale)[1]

    accepted_derivative = accepted
    rejected_derivative = rejected
    for _ in range(order):
        accepted_derivative = transform(accepted_derivative)
        rejected_derivative = transform(rejected_derivative)

    with no_vmap_fallback():
        good = accepted_derivative(parameter)
        bad = rejected_derivative(parameter)

    torch.testing.assert_close(
        good, torch.ones_like(good) if order == 1 else torch.zeros_like(good)
    )
    assert torch.isnan(bad).all()


def test_explicit_fixed_budget_is_retained() -> None:
    """A caller that disables early exit still gets the fixed-budget loop."""
    rhs = torch.ones(2, 3, device=DEVICE, dtype=torch.double)
    calls = Counter()

    def matvec(vector: Tensor) -> Tensor:
        calls["operator"] += 1
        return vector

    def solve(value: Tensor) -> LinearSolveResult:
        return implicit_solve(matvec, value, maxiter=20, early_exit=False)

    actual = vmap(solve)(rhs)

    assert calls["operator"] == 22
    assert (actual.iterations == 1).all()


@pytest.mark.parametrize("transform", [jacrev, jacfwd])
def test_mapped_operator_with_shared_rhs(transform: Callable[..., Any]) -> None:
    """A shared initial residual can become mapped after the first product."""
    rhs = torch.ones(3, device=DEVICE, dtype=torch.double)
    parameters = torch.tensor([1.0, 2.0], device=DEVICE, dtype=torch.double)
    calls = Counter()

    def matvec(vector: Tensor, scale: Tensor) -> Tensor:
        calls["operator"] += 1
        return vector * scale

    def solve(scale: Tensor) -> Tensor:
        return implicit_solve(
            matvec, rhs, scale, fast_matvec=matvec, fallback=True, maxiter=40
        ).solution

    mapped = vmap(solve)
    derivative = transform(mapped)
    with no_vmap_fallback():
        actual = mapped(parameters)
        gradient = derivative(parameters)

    expected = rhs / parameters[:, None]
    torch.testing.assert_close(actual, expected)
    expected_gradient = torch.zeros(2, 3, 2, dtype=rhs.dtype, device=rhs.device)
    expected_gradient[0, :, 0] = -rhs / parameters[0].square()
    expected_gradient[1, :, 1] = -rhs / parameters[1].square()
    torch.testing.assert_close(gradient, expected_gradient)
    assert calls["operator"] < 25


@pytest.mark.parametrize("mapped_parameter", [False, True])
@pytest.mark.parametrize(
    "transform,order",
    [
        (None, 0),
        (jacrev, 1),
        (jacfwd, 1),
        (jacrev, 2),
        (jacfwd, 2),
        (jacrev, 3),
        (jacfwd, 3),
    ],
)
def test_mapped_linearization(
    mapped_parameter: bool, transform: Callable[..., Any] | None, order: int
) -> None:
    """Mapped cached solves agree with loops and a dense derivative oracle."""
    dd = {"device": DEVICE, "dtype": torch.double}
    matrices = torch.tensor(
        [[[2.0, 0.2], [0.2, 3.0]], [[3.0, 0.1], [0.1, 4.0]]], **dd
    )
    rhs = torch.tensor([[1.0, 2.0], [3.0, 1.0]], **dd)
    shared = torch.tensor(0.3, **dd)
    calls: Counter[str] = Counter()

    def matrix(parameter: Tensor, offset: Tensor) -> Tensor:
        return parameter @ parameter.mT + (2.0 + offset) * torch.eye(2, **dd)

    def matvec(vector: Tensor, parameter: Tensor, offset: Tensor) -> Tensor:
        calls["ordinary"] += 1
        return matrix(parameter, offset) @ vector

    def linearize(
        parameter: Tensor, offset: Tensor
    ) -> Callable[[Tensor], Tensor]:
        calls["setup"] += 1
        cached = matrix(parameter, offset)

        def product(vector: Tensor) -> Tensor:
            calls["cached"] += 1
            return cached @ vector

        return product

    def solve(
        parameter: Tensor, right_hand_side: Tensor, offset: Tensor
    ) -> Tensor:
        result = implicit_solve(
            matvec,
            right_hand_side,
            parameter,
            offset,
            linearize=linearize,
            rlinearize=linearize,
            maxiter=40,
        )
        return result.solution

    def dense(
        parameter: Tensor, right_hand_side: Tensor, offset: Tensor
    ) -> Tensor:
        factor = torch.linalg.cholesky(matrix(parameter, offset))
        return torch.cholesky_solve(right_hand_side[:, None], factor).squeeze(
            -1
        )

    # Map RHS columns and optionally operator matrices along nonleading
    # axes. The scalar operator parameter is shared by both systems.
    matrix_values = matrices.movedim(0, -1) if mapped_parameter else matrices[0]
    rhs_columns = rhs.mT
    dimensions = (2 if mapped_parameter else None, 1, None)
    mapped_solve = vmap(solve, in_dims=dimensions)

    def evaluate(offset: Tensor) -> Tensor:
        return mapped_solve(matrix_values, rhs_columns, offset)

    def independent(offset: Tensor, operator: Callable[..., Tensor]) -> Tensor:
        values = []
        for index in range(2):
            parameter = matrices[index] if mapped_parameter else matrices[0]
            values.append(operator(parameter, rhs[index], offset))
        return torch.stack(values)

    def loop(offset: Tensor) -> Tensor:
        return independent(offset, solve)

    def oracle(offset: Tensor) -> Tensor:
        return independent(offset, dense)

    for _ in range(order):
        assert transform is not None
        evaluate = transform(evaluate)
        loop = transform(loop)
        oracle = transform(oracle)

    expected = oracle(shared)
    loop_result = loop(shared)
    loop_calls = calls.copy()
    calls.clear()

    with no_vmap_fallback():
        actual = evaluate(shared)

    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-11)
    torch.testing.assert_close(actual, loop_result, rtol=1e-10, atol=1e-11)
    assert loop_calls["cached"] > 0

    assert calls["cached"] > 0
    assert calls["setup"] > 0
    if order == 0:
        assert calls["ordinary"] == 0
        if mapped_parameter:
            # Preserve the base implementation's cache and fixed budget.
            assert calls["cached"] == 42
        else:
            assert calls["cached"] < 40
    else:
        assert calls["ordinary"] > 0
        if not mapped_parameter:
            assert calls["setup"] >= 2


@requires_compile
@pytest.mark.usefixtures("reset_dynamo")
@pytest.mark.parametrize("refined", [False, True])
def test_compiled_fixed_budget(refined: bool) -> None:
    """Compiled solves keep their existing Krylov and fallback budgets."""
    import operator

    rhs = torch.ones(3, device=DEVICE, dtype=torch.double)
    matrix = torch.eye(3, device=DEVICE, dtype=torch.double)
    budget = 7
    product_counts: list[int] = []

    def matvec(vector: Tensor, values: Tensor) -> Tensor:
        return values @ vector

    def solve(right_hand_side: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        result = implicit_solve(
            matvec,
            right_hand_side,
            matrix,
            maxiter=budget,
            fast_matvec=matvec if refined else None,
            refinements=0,
            fallback=True,
        )
        return result.solution, result.iterations, result.fallback_iterations

    def backend(
        graph: torch.fx.GraphModule, inputs: list[Tensor]
    ) -> Callable[..., Any]:
        product_counts.append(
            sum(
                node.op == "call_function" and node.target is operator.matmul
                for node in graph.graph.nodes
            )
        )
        return graph.forward

    compiled_solve = torch.compile(solve, backend=backend, fullgraph=True)
    solution, iterations, fallback_iterations = compiled_solve(rhs)

    torch.testing.assert_close(solution, rhs)
    assert iterations == 1
    assert fallback_iterations == 0

    # Setup and final true-residual checks surround every fixed-budget
    # Krylov loop. Refinement adds one exact check and an exact fallback,
    # even though that fallback reports zero accepted iterations.
    expected = 2 * (budget + 2) + 1 if refined else budget + 2
    assert product_counts == [expected]


def test_mapped_refinement_with_linearization() -> None:
    """Mapped refinement retains dynamic termination and a correct oracle."""
    rhs = torch.ones(3, device=DEVICE, dtype=torch.double)
    scales = torch.tensor([1.0, 2.0], device=DEVICE, dtype=torch.double)
    calls: Counter[str] = Counter()

    def matvec(vector: Tensor, scale: Tensor) -> Tensor:
        calls["ordinary"] += 1
        return scale * vector

    def linearize(scale: Tensor) -> Callable[[Tensor], Tensor]:
        calls["setup"] += 1

        def product(vector: Tensor) -> Tensor:
            calls["cached"] += 1
            return scale * vector

        return product

    def solve(scale: Tensor) -> Tensor:
        result = implicit_solve(
            matvec,
            rhs,
            scale,
            linearize=linearize,
            fast_matvec=matvec,
            fallback=True,
            maxiter=40,
        )
        return result.solution

    actual = vmap(solve)(scales)

    torch.testing.assert_close(actual, rhs / scales[:, None])
    assert calls["setup"] == 0
    assert calls["cached"] == 0
    assert calls["ordinary"] < 10
