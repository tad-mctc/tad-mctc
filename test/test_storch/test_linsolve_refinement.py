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
    STATUS_CONVERGED,
    STATUS_NOT_CONVERGED,
    STATUS_STAGNATED,
    constrained_solve,
    dense_solve,
    implicit_solve,
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
