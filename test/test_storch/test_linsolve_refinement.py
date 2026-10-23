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
Test the iterative refinement of implicit solves with a fast operator.

The fast operator is the same matrix applied in float32: alone, it gives
solutions and derivatives accurate to about 1e-6; refined, both must have
the float64 accuracy of the exact operator.
"""

from __future__ import annotations

import pytest
import torch
from torch.func import jacfwd, jacrev, vmap

from tad_mctc.storch.linsolve import (
    STATUS_BREAKDOWN,
    STATUS_CONVERGED,
    STATUS_NOT_CONVERGED,
    STATUS_STAGNATED,
    constrained_solve,
    dense_solve,
    implicit_solve,
    require_convergence,
)
from tad_mctc.storch.linsolve.implicit import Method
from tad_mctc.typing import Tensor

from ..conftest import DEVICE

DD = {"device": DEVICE, "dtype": torch.double}


def matvec(x: Tensor, a: Tensor) -> Tensor:
    return torch.einsum("...ij,...j->...i", a, x)


def fast(x: Tensor, a: Tensor) -> Tensor:
    """The operator in float32."""
    return torch.einsum("...ij,...j->...i", a.float(), x.float()).double()


def _randn(*shape: int, seed: int) -> Tensor:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(
        *shape, generator=gen, dtype=torch.double, device="cpu"
    ).to(DEVICE)


def matrix(method: str, n: int = 12, batch: tuple[int, ...] = ()) -> Tensor:
    s = _randn(*batch, n, n, seed=0)
    eye = torch.eye(n, **DD)
    if method == "cg":
        return s @ s.mT + n * eye
    return s + 2 * n**0.5 * eye


def assert_close(x: Tensor, y: Tensor, tol: float) -> None:
    assert x.shape == y.shape
    assert pytest.approx(y.detach().cpu(), abs=tol, rel=tol) == x.detach().cpu()


@pytest.mark.parametrize("method", ["cg", "bicgstab", "gmres"])
def test_accuracy(method: Method) -> None:
    a, b = matrix(method), _randn(12, seed=1)
    ref = torch.linalg.solve(a, b)

    alone = implicit_solve(fast, b, a, method=method, rtol=1e-5)
    refined = implicit_solve(
        matvec, b, a, method=method, fast_matvec=fast, rtol=1e-12
    )

    assert (alone.solution - ref).abs().max() > 1e-9
    assert refined.status.item() == STATUS_CONVERGED
    assert refined.residual <= refined.target
    assert_close(refined.solution, ref, 1e-11)


@pytest.mark.parametrize("method", ["cg", "gmres"])
def test_derivatives(method: Method) -> None:
    """
    Every derivative solve is refined as well. CG needs a symmetric matrix,
    also under perturbations, so A is parametrized symmetrically for it.
    """
    a, b = matrix(method), _randn(12, seed=1)
    w = _randn(12, seed=2)

    def loss(solver):
        def f(a: Tensor, b: Tensor) -> Tensor:
            if method == "cg":
                a = 0.5 * (a + a.mT)
            x = solver(a, b)
            return (w * x).sum() + (x * x).sum()

        return f

    def refined(a: Tensor, b: Tensor) -> Tensor:
        return implicit_solve(
            matvec, b, a, method=method, fast_matvec=fast, rtol=1e-12
        ).solution

    def reference(a: Tensor, b: Tensor) -> Tensor:
        return dense_solve(a, b, "lu")

    for t in (jacrev, lambda f, a: jacfwd(jacrev(f, a), a)):
        for argnum in (0, 1):
            out = t(loss(refined), argnum)(a, b)
            ref = t(loss(reference), argnum)(a, b)
            assert_close(out, ref, 1e-9)


def test_batch_and_vmap() -> None:
    a, b = matrix("cg", batch=(3,)), _randn(3, 12, seed=1)
    ref = torch.linalg.solve(a, b)

    batched = implicit_solve(matvec, b, a, fast_matvec=fast, rtol=1e-12)
    assert batched.converged.all()
    assert_close(batched.solution, ref, 1e-11)

    def solve(a: Tensor, b: Tensor) -> Tensor:
        return implicit_solve(
            matvec, b, a, fast_matvec=fast, rtol=1e-12
        ).solution

    assert_close(vmap(solve)(a, b), ref, 1e-11)
    grad = vmap(jacrev(lambda a, b: solve(a, b).sum(), argnums=1))(a, b)
    ones = torch.ones(3, 12, **DD)
    assert_close(grad, torch.linalg.solve(a.mT, ones), 1e-10)


def test_stagnation_and_limit() -> None:
    """
    A fast operator far from the exact one: overshooting corrections are
    rejected (stagnation), slowly contracting ones run out of steps.
    """
    a, b = matrix("cg"), _randn(12, seed=1)

    def scaled(factor: float):
        return lambda x, a: factor * matvec(x, a)

    result = implicit_solve(matvec, b, a, fast_matvec=scaled(0.2))
    assert result.status.item() == STATUS_STAGNATED

    # each correction leaves a quarter of the error
    result = implicit_solve(
        matvec, b, a, fast_matvec=scaled(0.8), rtol=1e-12, refinements=2
    )
    assert result.status.item() == STATUS_NOT_CONVERGED
    result = implicit_solve(
        matvec, b, a, fast_matvec=scaled(0.8), rtol=1e-12, refinements=30
    )
    assert result.status.item() == STATUS_CONVERGED


@pytest.mark.parametrize(
    "factor", [0.2, -1.0], ids=["stagnation", "indefinite"]
)
def test_fallback(factor: float) -> None:
    """
    With the fallback, systems the fast operator cannot refine (rejected
    corrections, an indefinite fast operator) are solved with the exact
    one, including the derivative solves; the others are unaffected.
    """
    a, b = matrix("cg", batch=(2,)), _randn(2, 12, seed=1)
    ref = torch.linalg.solve(a, b)

    def fast_or_bad(x: Tensor, a: Tensor) -> Tensor:
        good = fast(x, a)
        return torch.cat((good[:1], factor * matvec(x, a)[1:]))

    result = implicit_solve(matvec, b, a, fast_matvec=fast_or_bad, rtol=1e-12)
    assert result.status[0].item() == STATUS_CONVERGED
    assert result.status[1].item() != STATUS_CONVERGED

    def solve(a: Tensor, b: Tensor) -> Tensor:
        out = implicit_solve(
            matvec, b, a, fast_matvec=fast_or_bad, rtol=1e-12, fallback=True
        )
        assert out.converged.all()
        return out.solution

    fallback = implicit_solve(
        matvec, b, a, fast_matvec=fast_or_bad, rtol=1e-12, fallback=True
    )
    assert fallback.converged.all()
    assert fallback.iterations[1] > result.iterations[1]
    assert (fallback.solution[0] == result.solution[0]).all()
    assert_close(solve(a, b), ref, 1e-11)

    # the diagnostics tell the refinement and the fallback apart
    assert fallback.refinement_status is not None
    assert fallback.fallback_iterations is not None
    assert fallback.refinement_status.tolist() == result.status.tolist()
    assert fallback.fallback_iterations[0] == 0
    assert fallback.fallback_iterations[1] > 0
    assert (
        fallback.iterations - fallback.fallback_iterations == result.iterations
    ).all()

    grad = jacrev(lambda b: solve(a, b).sum())(b)
    ones = torch.ones(2, 12, **DD)
    assert_close(grad, torch.linalg.solve(a.mT, ones), 1e-10)


def test_fallback_is_a_retry() -> None:
    """
    The fallback reuses the method and the preconditioner: where these do
    not suit the exact operator either, it fails as well, and says so.
    """
    a, b = matrix("cg"), _randn(12, seed=1)

    def indefinite(x: Tensor, a: Tensor) -> Tensor:
        return -fast(x, a)

    def negative(r: Tensor, a: Tensor) -> Tensor:
        return -r  # not positive definite: no CG with it converges

    result = implicit_solve(
        matvec,
        b,
        a,
        fast_matvec=indefinite,
        preconditioner=negative,
        fallback=True,
    )
    assert result.refinement_status is not None
    assert result.fallback_iterations is not None
    assert result.refinement_status.item() == STATUS_BREAKDOWN
    assert result.status.item() == STATUS_BREAKDOWN
    with pytest.raises(RuntimeError):
        require_convergence(result)


def test_diagnostics_without_fast_operator() -> None:
    a, b = matrix("cg", batch=(2,)), _randn(2, 12, seed=1)
    result = implicit_solve(matvec, b, a)
    assert result.refinement_status is not None
    assert result.fallback_iterations is not None
    assert (result.refinement_status == result.status).all()
    assert (result.fallback_iterations == 0).all()


def test_initial_guess() -> None:
    a, b = matrix("cg"), _randn(12, seed=1)
    ref = torch.linalg.solve(a, b)
    result = implicit_solve(
        matvec, b, a, fast_matvec=fast, x0=ref + 1e-3, rtol=1e-12
    )
    assert result.converged.item()
    assert_close(result.solution, ref, 1e-11)


def test_constrained() -> None:
    n = 10
    a, b = matrix("cg", n), _randn(n, seed=1)
    c, d = torch.ones(1, n, **DD), torch.tensor([0.5], **DD)

    def solve(a: Tensor, d: Tensor, fast_matvec) -> Tensor:
        return constrained_solve(
            matvec, b, c, d, a, fast_matvec=fast_matvec, rtol=1e-12
        ).solution

    result = constrained_solve(matvec, b, c, d, a, fast_matvec=fast, rtol=1e-12)
    assert result.converged.item()
    assert_close(result.solution, solve(a, d, None), 1e-11)

    def loss(a: Tensor, d: Tensor, refine: bool) -> Tensor:
        return (solve(a, d, fast if refine else None) ** 2).sum()

    for argnum in (0, 1):
        out = jacrev(loss, argnum)(a, d, True)
        ref = jacrev(loss, argnum)(a, d, False)
        assert_close(out, ref, 1e-9)


def test_invalid() -> None:
    a, b = matrix("cg"), _randn(12, seed=1)
    with pytest.raises(ValueError, match="refinements"):
        implicit_solve(matvec, b, a, fast_matvec=fast, refinements=-1)
