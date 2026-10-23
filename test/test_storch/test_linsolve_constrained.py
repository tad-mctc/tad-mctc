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
Test the equality-constrained solve and its implicit derivatives.

The reference is the dense bordered (KKT) system solved by `dense_solve`
with LU, whose derivatives of every order are exact (unlike those of
`torch.linalg.solve`, see `test_linsolve_implicit.py`).
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from torch.func import jacfwd, jacrev, vmap

from tad_mctc.exceptions import DtypeError
from tad_mctc.storch.linsolve import (
    STATUS_BREAKDOWN,
    ConstrainedSolveResult,
    constrained_solve,
    dense_solve,
    require_convergence,
)
from tad_mctc.storch.linsolve.krylov import projected_cg
from tad_mctc.typing import Tensor

from ..conftest import DEVICE


def matvec(x: Tensor, a: Tensor) -> Tensor:
    return torch.einsum("...ij,...j->...i", a, x)


def jacobi(r: Tensor, a: Tensor) -> Tensor:
    return r / a.diagonal(dim1=-2, dim2=-1)


def _randn(*shape: int, seed: int) -> Tensor:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(
        *shape, generator=gen, dtype=torch.double, device="cpu"
    ).to(DEVICE)


def spd(n: int, batch: tuple[int, ...] = (), seed: int = 0) -> Tensor:
    s = _randn(*batch, n, n, seed=seed)
    return s @ s.mT + n * torch.eye(n, dtype=torch.double, device=DEVICE)


def kkt_matrix(a: Tensor, c: Tensor) -> Tensor:
    m = c.shape[-2]
    batch = torch.broadcast_shapes(a.shape[:-2], c.shape[:-2])
    a, c = a.expand(*batch, *a.shape[-2:]), c.expand(*batch, *c.shape[-2:])
    zeros = torch.zeros(*batch, m, m, dtype=a.dtype, device=a.device)
    return torch.cat(
        (torch.cat((a, c.mT), -1), torch.cat((c, zeros), -1)), dim=-2
    )


def reference(
    a: Tensor, b: Tensor, c: Tensor, d: Tensor
) -> tuple[Tensor, Tensor]:
    """Dense KKT solve: solution and multipliers."""
    n = b.shape[-1]
    k = kkt_matrix(a, c)
    batch = k.shape[:-2]
    rhs = torch.cat((b.expand(*batch, n), d.expand(*batch, d.shape[-1])), -1)
    v = dense_solve(k, rhs, "lu")
    return v[..., :n], v[..., n:]


def solve(a: Tensor, b: Tensor, c: Tensor, d: Tensor, **kwargs) -> Tensor:
    return constrained_solve(matvec, b, c, d, a, **kwargs).solution


def assert_close(x: Tensor, y: Tensor, tol: float = 1e-8) -> None:
    assert x.shape == y.shape
    assert pytest.approx(y.cpu(), abs=tol, rel=tol) == x.cpu()


@pytest.mark.parametrize("m", [1, 2])
@pytest.mark.parametrize("batch", [(), (3,)], ids=["single", "batch"])
def test_solution(m: int, batch: tuple[int, ...]) -> None:
    n = 8
    a, b = spd(n, batch), _randn(*batch, n, seed=1)
    c, d = _randn(*batch, m, n, seed=2), _randn(*batch, m, seed=3)

    result = constrained_solve(matvec, b, c, d, a)
    x, mu = reference(a, b, c, d)

    assert isinstance(result, ConstrainedSolveResult)
    assert result.converged.all()
    assert result.residual.shape == batch
    assert_close(result.solution, x)
    assert_close(result.multiplier, mu)
    assert_close(
        matvec(result.solution, a) + (c.mT @ mu.unsqueeze(-1)).squeeze(-1), b
    )


def test_broadcasting() -> None:
    """One constraint shared by a batch, a batch of values."""
    n = 6
    a, b = spd(n, (4,)), _randn(4, n, seed=1)
    c = torch.ones(1, n, dtype=torch.double, device=DEVICE)
    d = _randn(4, 1, seed=3)

    result = constrained_solve(matvec, b, c, d, a)
    x, _ = reference(a, b, c, d)

    assert_close(result.solution, x)
    assert_close(result.solution.sum(-1, keepdim=True), d)


def test_indefinite_operator() -> None:
    """A is indefinite, but positive definite on the null space of C."""
    n = 10
    c = torch.ones(1, n, dtype=torch.double, device=DEVICE)
    proj = torch.eye(n, dtype=torch.double, device=DEVICE) - c.mT @ c / n
    a = proj @ spd(n) @ proj - 5.0 * c.mT @ c / n
    assert torch.linalg.eigvalsh(a)[0] < 0

    b, d = _randn(n, seed=1), _randn(1, seed=3)
    result = constrained_solve(matvec, b, c, d, a)
    x, _ = reference(a, b, c, d)

    assert result.converged.item()
    assert_close(result.solution, x)


def test_breakdown() -> None:
    """Negative curvature on the null space, a rank-deficient C."""
    n = 6
    a, b = -spd(n), _randn(n, seed=1)
    c, d = torch.ones(1, n, dtype=torch.double, device=DEVICE), _randn(
        1, seed=3
    )
    result = constrained_solve(matvec, b, c, d, a)
    assert result.status.item() == STATUS_BREAKDOWN

    a = spd(n)
    c = torch.ones(2, n, dtype=torch.double, device=DEVICE)
    result = constrained_solve(matvec, b, c, torch.ones(2, **_dd()), a)
    assert result.status.item() == STATUS_BREAKDOWN


def _dd() -> dict:
    return {"dtype": torch.double, "device": DEVICE}


@pytest.mark.parametrize("fast", [False, True], ids=["exact", "refined"])
def test_constraint_residual(fast: bool) -> None:
    """
    The constraints hold to rounding, independent of the tolerance of the
    bordered system: here a loose one, with a large right-hand side, in a
    padded batch with two constraints.
    """
    n, real = 200, 150
    a, b = spd(n, (2,)), 1e3 * _randn(2, n, seed=1)
    a[1, real:, :] = 0.0
    a[1, :, real:] = 0.0
    a[1, real:, real:] = torch.eye(n - real, **_dd())
    b[1, real:] = 0.0
    c = _randn(2, 2, n, seed=2)
    c[1, :, real:] = 0.0
    d = _randn(2, 2, seed=3)

    def fast_matvec(x: Tensor, a: Tensor) -> Tensor:
        return matvec(x.float(), a.float()).double()

    result = constrained_solve(
        matvec, b, c, d, a, rtol=1e-6, fast_matvec=fast_matvec if fast else None
    )
    assert result.converged.all()
    violation = (c @ result.solution.unsqueeze(-1)).squeeze(-1) - d
    assert_close(result.constraint_residual, violation.norm(dim=-1), 1e-15)

    # rounding: far below the tolerance of the bordered system
    eps = torch.finfo(torch.double).eps
    scale = (c.abs() * result.solution.abs().unsqueeze(-2)).sum(-1).amax(-1)
    assert (result.constraint_residual <= 100 * eps * scale).all()
    assert (result.constraint_residual < 1e-6 * result.target).all()
    assert (result.solution[1, real:] == 0).all()


def test_total_charge_derivatives() -> None:
    """The total is fixed: its derivative is zero for the operator and the
    right-hand side, and one for the value."""
    n = 12
    a, b = spd(n), _randn(n, seed=1)
    c, d = torch.ones(1, n, **_dd()), torch.tensor([0.5], **_dd())

    def total(a: Tensor, b: Tensor, d: Tensor) -> Tensor:
        return constrained_solve(matvec, b, c, d, a, rtol=1e-12).solution.sum()

    da, db, dd = jacrev(total, argnums=(0, 1, 2))(a, b, d)
    assert da.abs().max() < 1e-13 and db.abs().max() < 1e-13
    assert_close(dd, torch.ones(1, **_dd()), 1e-12)


def test_constraint_operator() -> None:
    """With the constraint part, the rows of C need no products with A."""
    a, b = spd(10), _randn(10, seed=1)
    c, d = _randn(3, 10, seed=2), _randn(3, seed=3)

    calls = []

    def bordered(v: Tensor, a: Tensor, c: Tensor) -> Tensor:
        calls.append(1)
        return constraint(v, a, c) + torch.cat(
            (matvec(v[..., :10], a), torch.zeros_like(d))
        )

    def constraint(v: Tensor, a: Tensor, c: Tensor) -> Tensor:
        x, mu = v[..., :10], v[..., 10:]
        return torch.cat((c.mT @ mu, c @ x))

    rhs = torch.cat((b, d))
    out = projected_cg(bordered, rhs, a, c, constraints=3)
    without = len(calls)
    calls.clear()
    ref = projected_cg(
        bordered, rhs, a, c, constraints=3, constraint=constraint
    )
    assert len(calls) == without - 3
    assert_close(out.solution, ref.solution, 1e-10)
    assert_close(ref.solution[:10], reference(a, b, c, d)[0])


def test_padding() -> None:
    """
    Padding as in a charge model: unit diagonal, no coupling, and a zero
    constraint coefficient leave the padded entries at their right-hand
    side, here zero.
    """
    n, real = 8, 5
    a = spd(n)
    a[real:, :] = 0.0
    a[:, real:] = 0.0
    a[real:, real:] = torch.eye(n - real, **_dd())
    b = _randn(n, seed=1)
    b[real:] = 0.0
    c = torch.zeros(1, n, **_dd())
    c[0, :real] = 1.0
    d = torch.tensor([-1.0], **_dd())

    result = constrained_solve(matvec, b, c, d, a)
    x, _ = reference(a[:real, :real], b[:real], c[:, :real], d)

    assert_close(result.solution[:real], x)
    assert (result.solution[real:] == 0).all()


def test_preconditioner_and_x0() -> None:
    """Badly scaled A: Jacobi cuts the iterations; a guess is corrected."""
    n = 30
    scale = torch.logspace(0, 3, n, **_dd())
    a = spd(n) * scale.unsqueeze(-1) * scale.unsqueeze(-2)
    b, c, d = _randn(n, seed=1), _randn(1, n, seed=2), _randn(1, seed=3)
    x, _ = reference(a, b, c, d)

    plain = constrained_solve(matvec, b, c, d, a, maxiter=1000)
    prec = constrained_solve(matvec, b, c, d, a, preconditioner=jacobi)
    assert prec.converged.item()
    assert prec.iterations < plain.iterations
    assert_close(prec.solution, x, 1e-6)

    # an infeasible guess is moved onto the constraint first
    guess = constrained_solve(
        matvec, b, c, d, a, preconditioner=jacobi, x0=x + 1.0
    )
    assert guess.converged.item()
    assert_close(guess.solution, x, 1e-6)


def test_early_exit() -> None:
    n = 6
    a, b = spd(n, (3,)), _randn(3, n, seed=1)
    c, d = _randn(3, 1, n, seed=2), _randn(3, 1, seed=3)

    early = constrained_solve(matvec, b, c, d, a, early_exit=True)
    full = constrained_solve(matvec, b, c, d, a, early_exit=False, maxiter=40)
    for x, y in zip(early, full):
        assert (x == y).all()


def test_require_convergence() -> None:
    n = 6
    a = torch.stack((spd(n), -spd(n)))
    b, c = _randn(2, n, seed=1), torch.ones(1, n, **_dd())
    d = _randn(2, 1, seed=3)

    with pytest.raises(Exception, match="1 of 2 system"):
        require_convergence(constrained_solve(matvec, b, c, d, a))

    def f(a: Tensor, b: Tensor, d: Tensor) -> Tensor:
        return require_convergence(constrained_solve(matvec, b, c, d, a))

    out = vmap(f)(a, b, d)
    assert_close(out[0], reference(a[0], b[0], c, d[0])[0])
    assert torch.isnan(out[1]).all()


########################################################################
# derivatives

N, M = 6, 2


def _inputs() -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return (
        spd(N),
        _randn(N, seed=1),
        _randn(M, N, seed=2),
        _randn(M, seed=3),
    )


def _weights() -> tuple[Tensor, Tensor]:
    return _randn(N, seed=4), _randn(M, seed=5)


def loss(
    fn: Callable[..., tuple[Tensor, Tensor]],
) -> Callable[[Tensor, Tensor, Tensor, Tensor], Tensor]:
    """A scalar of the solution and the multipliers (a symmetric A)."""
    w, v = _weights()

    def f(s: Tensor, b: Tensor, c: Tensor, d: Tensor) -> Tensor:
        a = 0.5 * (s + s.mT)
        x, mu = fn(a, b, c, d)
        return (w * x).sum() + (v * mu).sum() + (x * x).sum()

    return f


def _solver(a: Tensor, b: Tensor, c: Tensor, d: Tensor):
    result = constrained_solve(matvec, b, c, d, a, rtol=1e-13)
    return result.solution, result.multiplier


TRANSFORMS = {
    "jacrev": jacrev,
    "jacfwd": jacfwd,
    "jacrev-jacrev": lambda f, argnums: jacrev(jacrev(f, argnums), argnums),
    "jacfwd-jacrev": lambda f, argnums: jacfwd(jacrev(f, argnums), argnums),
    "jacrev-jacfwd": lambda f, argnums: jacrev(jacfwd(f, argnums), argnums),
    "jacfwd-jacfwd": lambda f, argnums: jacfwd(jacfwd(f, argnums), argnums),
}


@pytest.mark.parametrize("transform", TRANSFORMS)
@pytest.mark.parametrize("argnum", [0, 1, 2, 3], ids=["A", "b", "C", "d"])
def test_derivatives(transform: str, argnum: int) -> None:
    inputs = _inputs()
    t = TRANSFORMS[transform]
    out = t(loss(_solver), argnum)(*inputs)
    ref = t(loss(reference), argnum)(*inputs)
    assert_close(out, ref, 1e-7)


def test_vmap_of_gradient() -> None:
    """`vmap` over a batch of the gradients with respect to C and d."""
    a = spd(N, (3,))
    b, c, d = (
        _randn(3, N, seed=1),
        _randn(3, M, N, seed=2),
        _randn(3, M, seed=3),
    )
    grad = jacrev(loss(_solver), argnums=(2, 3))
    ref = jacrev(loss(reference), argnums=(2, 3))

    out = vmap(grad)(a, b, c, d)
    expected = vmap(ref)(a, b, c, d)
    for x, y in zip(out, expected):
        assert_close(x, y, 1e-7)


def test_preconditioned_derivatives() -> None:
    """The derivative solves use the preconditioner as well."""
    inputs = _inputs()

    def solver(a, b, c, d):
        result = constrained_solve(
            matvec, b, c, d, a, preconditioner=jacobi, rtol=1e-13
        )
        return result.solution, result.multiplier

    out = jacfwd(jacrev(loss(solver)))(*inputs)
    ref = jacfwd(jacrev(loss(reference)))(*inputs)
    assert_close(out, ref, 1e-7)


########################################################################
# arguments


def test_invalid_shapes() -> None:
    a, b, c, d = _inputs()
    with pytest.raises(ValueError, match="constraint matrix"):
        constrained_solve(matvec, b, c[..., :-1], d, a)
    with pytest.raises(ValueError, match="constraint matrix"):
        constrained_solve(matvec, b, c[0], d, a)
    with pytest.raises(ValueError, match="constraint values"):
        constrained_solve(matvec, b, c, d[:1], a)
    with pytest.raises(ValueError, match="initial guess"):
        constrained_solve(matvec, b, c, d, a, x0=b[:-1])
    with pytest.raises(DtypeError):
        constrained_solve(matvec, b, c.to(torch.int64), d, a)


def test_projected_cg_arguments() -> None:
    a, b, c, d = _inputs()
    rhs = torch.cat((b, d))
    bordered = lambda v, a: v  # noqa: E731
    with pytest.raises(ValueError, match="vdims=1"):
        projected_cg(bordered, rhs.unsqueeze(0), a, constraints=1, vdims=2)
    with pytest.raises(ValueError, match="number of constraints"):
        projected_cg(bordered, rhs, a, constraints=0)
    with pytest.raises(ValueError, match="number of constraints"):
        projected_cg(bordered, rhs, a, constraints=rhs.shape[-1])
