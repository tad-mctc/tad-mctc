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
"""Failed convergence masks must invalidate derivatives as well as values."""

from __future__ import annotations

import pytest
import torch
from torch.func import jacfwd, jacrev, vmap

from tad_mctc._version import __tversion__
from tad_mctc.autograd import no_vmap_fallback
from tad_mctc.storch.linsolve import (
    STATUS_CONVERGED,
    STATUS_NOT_CONVERGED,
    ConstrainedSolveResult,
    LinearSolveResult,
    implicit_solve,
    mask_converged,
    require_convergence,
)
from tad_mctc.tools.testing import requires_compile
from tad_mctc.typing import Tensor

from ..conftest import DEVICE

DERIVATIVES = {
    "reverse": jacrev,
    "forward": jacfwd,
    "reverse-reverse": lambda f: jacrev(jacrev(f)),
    "forward-reverse": lambda f: jacfwd(jacrev(f)),
    "reverse-forward": lambda f: jacrev(jacfwd(f)),
    "forward-forward": lambda f: jacfwd(jacfwd(f)),
    "third": lambda f: jacfwd(jacrev(jacrev(f))),
    "third-forward": lambda f: jacfwd(jacfwd(jacfwd(f))),
}


def _solution(value: Tensor, vdims: int) -> Tensor:
    """Analytic reference, including nonzero derivatives at each tested order."""
    return torch.stack((value.sin(), value.square())).reshape(
        (1,) * (vdims - 1) + (2,)
    )


def _checked(
    value: Tensor, converged: Tensor, vdims: int, constrained: bool
) -> Tensor:
    """Supply controlled statuses independently of solver convergence behavior."""
    solution = _solution(value, vdims)
    residual = torch.where(converged, 0.0, 1.0).to(value)
    target = torch.ones_like(residual)
    iterations = torch.zeros_like(converged, dtype=torch.int64)
    status = torch.where(converged, STATUS_CONVERGED, STATUS_NOT_CONVERGED)
    if constrained:
        result = ConstrainedSolveResult(
            solution,
            value.reshape(1),
            residual,
            target,
            iterations,
            status,
            residual,
        )
    else:
        result = LinearSolveResult(
            solution, residual, target, iterations, status
        )
    return require_convergence(result, vdims=vdims)


def test_public_mask_exports_and_broadcasting() -> None:
    """The public mask preserves shape and broadcasts a per-system boolean mask."""
    from tad_mctc import storch

    assert storch.mask_converged is mask_converged
    solution = torch.arange(6, dtype=torch.double, device=DEVICE).reshape(2, 3)
    accepted = torch.tensor([[True], [False]], device=DEVICE)
    masked = mask_converged(solution, accepted)
    assert masked.shape == solution.shape
    assert torch.equal(masked[0], solution[0])
    assert torch.isnan(masked[1]).all()
    assert torch.equal(
        mask_converged(solution, accepted.new_tensor(True)), solution
    )
    assert torch.isnan(
        mask_converged(solution, accepted.new_tensor(False))
    ).all()


def test_public_mask_rejects_invalid_masks() -> None:
    """Reject nonboolean masks and broadcasting that would expand the solution."""
    solution = torch.ones(2, 3, dtype=torch.double, device=DEVICE)
    with pytest.raises(TypeError, match="boolean"):
        mask_converged(solution, torch.ones(2, 1, device=DEVICE))
    with pytest.raises(RuntimeError):
        mask_converged(
            solution, torch.ones(4, 2, 3, dtype=torch.bool, device=DEVICE)
        )


@pytest.mark.parametrize("derivative", DERIVATIVES)
def test_linear_solution_zero_higher_derivatives(derivative: str) -> None:
    """For x(b)=b, valid derivatives are 1/0/0 while failed derivatives stay NaN."""
    values = torch.tensor([1.0, 2.0], dtype=torch.double, device=DEVICE)
    accepted = torch.tensor([True, False], device=DEVICE)

    def checked(value: Tensor, valid: Tensor) -> Tensor:
        return mask_converged(value, valid)

    actual = vmap(DERIVATIVES[derivative](checked))(values, accepted)
    expected = values.new_tensor(
        1.0 if derivative in ("reverse", "forward") else 0.0
    )
    assert torch.equal(actual[0], expected)
    assert torch.isnan(actual[1])


def test_select_before_nonlinear_downstream_operations() -> None:
    """Selection after a nonlinear NaN operation can poison even first derivatives."""
    accepted = torch.tensor([True, False], device=DEVICE)
    value = torch.tensor(2.0, dtype=torch.double, device=DEVICE)

    def masked(value: Tensor) -> Tensor:
        return mask_converged(torch.stack((value, 2 * value)), accepted)

    def select_after_square(value: Tensor) -> Tensor:
        return masked(value).square()[0]

    def select_before_square(value: Tensor) -> Tensor:
        return masked(value)[0].square()

    assert select_after_square(value) == 4.0
    # Squaring the failed lane creates 0 * NaN before the mask sees its VJP.
    # This is an explicit limitation, not a guarantee about downstream code.
    assert torch.isnan(jacrev(select_after_square)(value))
    assert jacrev(select_before_square)(value) == 4.0


def test_reverse_only_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise the compiled dispatch in Python as well as the real Dynamo test."""
    from tad_mctc.storch.linsolve import _mask

    # Dynamo executes these lines while tracing bytecode, so coverage does
    # not see them in the compile test. Force just the dispatch decision
    # here to check its reverse-only AD class with normal autograd too.
    monkeypatch.setattr(_mask, "is_compiling", lambda: True)
    values = torch.tensor(
        [1.0, 2.0], dtype=torch.double, device=DEVICE, requires_grad=True
    )
    accepted = torch.tensor([True, False], device=DEVICE)
    masked = mask_converged(values, accepted)
    gradient = torch.autograd.grad(masked.sum(), values)[0]
    assert gradient[0] == 1.0
    assert torch.isnan(gradient[1])

    with pytest.raises(NotImplementedError, match="Forward-mode"):
        jacfwd(lambda value: mask_converged(value, accepted))(values.detach())


@pytest.mark.parametrize("vdims", [1, 2])
@pytest.mark.parametrize("constrained", [False, True])
@pytest.mark.parametrize("derivative", DERIVATIVES)
def test_convergence_mask_derivatives(
    vdims: int, constrained: bool, derivative: str
) -> None:
    """Per-system values and nested derivatives fail closed in a mixed batch."""
    values = torch.tensor([0.7, 1.2], dtype=torch.double, device=DEVICE)
    statuses = torch.tensor([True, False], device=DEVICE)

    def checked(value: Tensor, converged: Tensor) -> Tensor:
        return _checked(value, converged, vdims, constrained)

    primal = vmap(checked)(values, statuses)
    assert torch.equal(primal[0], _solution(values[0], vdims))
    assert torch.isnan(primal[1]).all()

    differentiate = DERIVATIVES[derivative]
    with no_vmap_fallback():
        derivatives = vmap(differentiate(checked))(values, statuses)
    reference = differentiate(lambda value: _solution(value, vdims))(values[0])
    assert torch.allclose(derivatives[0], reference, atol=1e-12, rtol=0)
    assert torch.isnan(derivatives[1]).all()


@pytest.mark.parametrize("differentiate", [jacrev, jacfwd])
def test_convergence_mask_joint_jacobian(differentiate) -> None:  # type: ignore[no-untyped-def]
    """A valid output row retains zero cross-lane derivatives."""
    values = torch.tensor([0.7, 1.2], dtype=torch.double, device=DEVICE)
    statuses = torch.tensor([True, False], device=DEVICE)

    def batch(value: Tensor) -> Tensor:
        return vmap(lambda x, valid: _checked(x, valid, 1, False))(
            value, statuses
        )

    jacobian = differentiate(batch)(values)
    reference = differentiate(lambda value: _solution(value, 1))(values[0])
    assert torch.allclose(jacobian[0, :, 0], reference, atol=1e-12, rtol=0)
    assert torch.equal(jacobian[0, :, 1], torch.zeros_like(reference))
    assert torch.isnan(jacobian[1, :, 1]).all()


def test_convergence_mask_nested_vmap() -> None:
    """Nested batching rules retain the mask on each independent system."""
    values = torch.tensor(
        [[0.7, 1.2], [1.2, 0.7]], dtype=torch.double, device=DEVICE
    )
    statuses = torch.tensor([[True, False], [False, True]], device=DEVICE)
    derivative = jacrev(lambda value, valid: _checked(value, valid, 1, False))
    with no_vmap_fallback():
        actual = vmap(vmap(derivative))(values, statuses)
    reference = jacrev(lambda value: _solution(value, 1))(values[0, 0])
    assert torch.allclose(actual[0, 0], reference, atol=1e-12, rtol=0)
    assert torch.allclose(actual[1, 1], reference, atol=1e-12, rtol=0)
    assert torch.isnan(actual[0, 1]).all()
    assert torch.isnan(actual[1, 0]).all()


@pytest.mark.parametrize("differentiate", [jacrev, jacfwd])
def test_convergence_mask_shared_input(differentiate) -> None:  # type: ignore[no-untyped-def]
    """A shared differentiable input does not mix failed and valid output rows."""
    statuses = torch.tensor([True, False], device=DEVICE)
    value = torch.tensor(0.7, dtype=torch.double, device=DEVICE)

    def batch(value: Tensor) -> Tensor:
        return vmap(lambda valid: _checked(value, valid, 1, False))(statuses)

    derivatives = differentiate(batch)(value)
    reference = differentiate(lambda value: _solution(value, 1))(value)
    assert torch.allclose(derivatives[0], reference, atol=1e-12, rtol=0)
    assert torch.isnan(derivatives[1]).all()


@pytest.mark.parametrize("derivative", DERIVATIVES)
def test_valid_output_shared_input_higher_derivatives(derivative: str) -> None:
    """Selecting the valid output preserves its higher derivatives with shared inputs."""
    statuses = torch.tensor([True, False], device=DEVICE)
    value = torch.tensor(0.7, dtype=torch.double, device=DEVICE)

    def valid_output(value: Tensor) -> Tensor:
        batch = vmap(lambda valid: _checked(value, valid, 1, False))(statuses)
        return batch[0]

    differentiate = DERIVATIVES[derivative]
    actual = differentiate(valid_output)(value)
    reference = differentiate(lambda value: _solution(value, 1))(value)
    assert torch.allclose(actual, reference, atol=1e-12, rtol=0)


def test_implicit_convergence_mask_derivatives() -> None:
    """An actual finite, unconverged CG solution cannot yield finite Jacobians."""
    matrices = torch.stack(
        (torch.eye(2), torch.diag(torch.tensor([2.0, 3.0])))
    ).to(device=DEVICE, dtype=torch.double)
    value = matrices.new_tensor(1.0)

    def checked(matrix: Tensor, value: Tensor) -> Tensor:
        result = implicit_solve(
            lambda x, a: a @ x, value.expand(2), matrix, maxiter=1
        )
        return require_convergence(result)

    derivatives = vmap(jacrev(checked, argnums=1), in_dims=(0, None))(
        matrices, value
    )
    assert torch.allclose(derivatives[0], torch.ones_like(derivatives[0]))
    assert torch.isnan(derivatives[1]).all()


@requires_compile
@pytest.mark.skipif(
    __tversion__ < (2, 11, 0),
    reason="Compiled jacrev of custom Functions needs PyTorch >= 2.11",
)
@pytest.mark.usefixtures("reset_dynamo")
def test_convergence_mask_compile_jacrev() -> None:
    """Compiled reverse-mode masking invalidates the failed Jacobian block."""
    values = torch.tensor([0.7, 1.2], dtype=torch.double, device=DEVICE)
    statuses = torch.tensor([True, False], device=DEVICE)

    def batch(values: Tensor) -> Tensor:
        # An ordinary leading batch dimension is supported by compiled
        # jacrev; compiled higher transforms are intentionally not tested.
        solution = torch.stack((values.sin(), values.square()), dim=-1)
        residual = torch.where(statuses, 0.0, 1.0).to(values)
        result = LinearSolveResult(
            solution,
            residual,
            torch.ones_like(residual),
            torch.zeros_like(statuses, dtype=torch.int64),
            torch.where(statuses, STATUS_CONVERGED, STATUS_NOT_CONVERGED),
        )
        return require_convergence(result)

    jacobian = torch.compile(jacrev(batch), backend="eager", fullgraph=True)(
        values
    )
    reference = jacrev(lambda value: _solution(value, 1))(values[0])
    assert torch.allclose(jacobian[0, :, 0], reference, atol=1e-12, rtol=0)
    assert torch.equal(jacobian[0, :, 1], torch.zeros_like(reference))
    assert torch.isnan(jacobian[1, :, 1]).all()


@requires_compile
@pytest.mark.skipif(
    __tversion__ < (2, 11, 0),
    reason="Compiled jacrev of custom Functions needs PyTorch >= 2.11",
)
@pytest.mark.usefixtures("reset_dynamo")
@pytest.mark.xfail(
    strict=True,
    reason="Dynamo inlines the custom Function under compiled vmap and loses its VJP",
)
def test_compiled_vmap_mask_derivatives_are_unsupported() -> None:
    """Track the upstream limitation separately from supported compiled batching."""
    values = torch.tensor([0.7, 1.2], dtype=torch.double, device=DEVICE)
    statuses = torch.tensor([True, False], device=DEVICE)

    def batch(values: Tensor) -> Tensor:
        return vmap(lambda value, valid: _checked(value, valid, 1, False))(
            values, statuses
        )

    jacobian = torch.compile(jacrev(batch), backend="eager", fullgraph=True)(
        values
    )
    reference = jacrev(lambda value: _solution(value, 1))(values[0])
    assert torch.allclose(jacobian[0, :, 0], reference, atol=1e-12, rtol=0)
    assert torch.isnan(jacobian[1, :, 1]).all()
