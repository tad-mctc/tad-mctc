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
Test implicit differentiation of roots, fixed points and minima.

Oracles: a fully converged Newton iteration (dense `inv`, whose derivatives
are exact in every nesting) differentiated by unrolling, and closed forms.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from functools import partial
from typing import Any

import pytest
import torch
from torch.func import jacfwd, jacrev, vmap

from tad_mctc._version import __tversion__
from tad_mctc.storch.implicit import (
    implicit_argmin,
    implicit_fixed_point,
    implicit_root,
)
from tad_mctc.storch.linsolve import gmres
from tad_mctc.typing import Tensor

from ..conftest import DEVICE

DD = {"dtype": torch.double, "device": DEVICE}


def _randn(*shape: int, seed: int) -> Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=gen, dtype=torch.double).to(DEVICE)


def _nest(f: Callable, modes: str) -> Callable:
    for mode in modes:
        f = (jacrev if mode == "r" else jacfwd)(f)
    return f


def assert_close(x: Tensor, y: Tensor, tol: float = 1e-9) -> None:
    assert x.shape == y.shape
    assert pytest.approx(y.detach().cpu(), abs=tol, rel=tol) == x.cpu()


########################################################################
# A nonlinear system F(z, p) = A z + z^3 - p (A symmetric positive
# definite, so F_z = A + 3 diag(z^2) is too)

N = 4
A = _randn(N, N, seed=0)
A = A @ A.mT + N * torch.eye(N, **DD)


def residual(z: Tensor, p: Tensor) -> Tensor:
    return z @ A.mT + z**3 - p


def newton(p: Tensor, steps: int = 30) -> Tensor:
    """Differentiable reference solve (unrolled Newton with `inv`)."""
    z = torch.zeros_like(p)
    for _ in range(steps):
        jac = A + torch.diag_embed(3 * z**2)
        z = z - (torch.linalg.inv(jac) @ residual(z, p).unsqueeze(-1)).squeeze(
            -1
        )
    return z


def root_solution(p: Tensor, **kwargs: Any) -> Tensor:
    return implicit_root(residual, newton(p.detach()), p, **kwargs)


P = _randn(N, seed=1)


@pytest.mark.parametrize("method", ["gmres", "bicgstab", "cg"])
def test_root_first_order(method: str) -> None:
    f = partial(root_solution, method=method)
    assert_close(f(P), newton(P))
    assert_close(jacrev(f)(P), jacrev(newton)(P))
    assert_close(jacfwd(f)(P), jacrev(newton)(P))


@pytest.mark.parametrize("order", [2, 3])
def test_root_all_nestings(order: int) -> None:
    def f(p: Tensor) -> Tensor:
        return root_solution(p).sum()

    def ref(p: Tensor) -> Tensor:
        return newton(p).sum()

    expected = _nest(ref, "r" * order)(P)
    for modes in itertools.product("rf", repeat=order):
        assert_close(_nest(f, "".join(modes))(P), expected)


@pytest.mark.parametrize("modes", ["r", "rr", "fr", "rf", "fff"])
def test_root_vmap(modes: str) -> None:
    p = torch.stack((P, 0.5 * P, -P))

    def f(p: Tensor) -> Tensor:
        return root_solution(p).sum()

    def ref(p: Tensor) -> Tensor:
        return newton(p).sum()

    batched = vmap(_nest(f, modes))(p)
    looped = torch.stack([_nest(ref, modes)(pi) for pi in p])
    assert_close(batched, looped)


def test_root_vdims_batch() -> None:
    """Independent systems in the leading dimension of the root."""
    p = torch.stack((P, 0.5 * P, -P))
    z = implicit_root(residual, newton(p.detach()), p, vdims=1)
    jac = jacrev(lambda p: implicit_root(residual, newton(p), p, vdims=1))(p)
    assert_close(z, newton(p))
    assert_close(jac, jacrev(newton)(p))


def test_root_static_params() -> None:
    def scaled(z: Tensor, p: Tensor, scale: float, name: str) -> Tensor:
        assert name == "static"
        return residual(z, scale * p)

    def f(p: Tensor) -> Tensor:
        return implicit_root(scaled, newton(2.0 * p), p, 2.0, "static")

    assert_close(
        jacfwd(jacrev(f))(P), jacfwd(jacrev(lambda p: newton(2 * p)))(P)
    )


def test_root_gradcheck() -> None:
    p = P.clone().requires_grad_(True)
    f = lambda p: implicit_root(residual, newton(p.detach()), p)  # noqa: E731
    assert torch.autograd.gradcheck(f, (p,))
    assert torch.autograd.gradgradcheck(f, (p,))
    assert torch.autograd.gradcheck(f, (p,), check_forward_ad=True)


def test_root_method_options() -> None:
    f = partial(root_solution, method=partial(gmres, restart=2), maxiter=200)
    assert_close(jacrev(f)(P), jacrev(newton)(P))


def test_root_preconditioner() -> None:
    def jacobi(r: Tensor, z: Tensor, p: Tensor) -> Tensor:
        return r / (A.diagonal() + 3 * z**2)

    f = partial(root_solution, preconditioner=jacobi)
    assert_close(
        jacfwd(jacrev(lambda p: f(p).sum()))(P),
        jacfwd(jacrev(lambda p: newton(p).sum()))(P),
    )


def test_root_scalar() -> None:
    def cube(z: Tensor, p: Tensor) -> Tensor:
        return z**3 - p

    p = torch.tensor(8.0, **DD)
    f = lambda p: implicit_root(cube, p.detach() ** (1 / 3), p)  # noqa: E731
    ref = lambda p: p ** (1 / 3)  # noqa: E731
    assert f(p).shape == ()
    for modes in ("r", "f", "rf", "fff", "rrr"):
        assert_close(_nest(f, modes)(p), _nest(ref, modes)(p))


def test_root_singular_jacobian_is_nan() -> None:
    """Fail-closed: a singular F_z gives NaN derivatives."""

    def square(z: Tensor, p: Tensor) -> Tensor:
        return z**2 - p

    p = torch.zeros(2, **DD)
    grad = jacrev(lambda p: implicit_root(square, torch.zeros(2, **DD), p))(p)
    assert torch.isnan(grad).all()


def test_root_no_differentiable_params() -> None:
    z = implicit_root(residual, newton(P), P.detach())
    assert not z.requires_grad
    assert_close(z, newton(P))


########################################################################
# Fixed points


def test_fixed_point_all_nestings() -> None:
    def g(x: Tensor, a: Tensor) -> Tensor:
        return torch.cos(a * x)

    def unrolled(a: Tensor) -> Tensor:
        x = torch.tensor(0.7, **DD)
        for _ in range(400):
            x = torch.cos(a * x)
        return x

    def f(a: Tensor) -> Tensor:
        return implicit_fixed_point(g, unrolled(a.detach()), a)

    a = torch.tensor(1.0, **DD)
    for order in (1, 2, 3):
        expected = _nest(unrolled, "r" * order)(a)
        for modes in itertools.product("rf", repeat=order):
            assert_close(_nest(f, "".join(modes))(a), expected)


def test_fixed_point_not_contractive() -> None:
    """Only I - G_x must be invertible: G_x = -2 is not a contraction."""

    def g(x: Tensor, b: Tensor) -> Tensor:  # x = -2 x + b, x* = b / 3
        return -2.0 * x + b

    b = torch.tensor([3.0, 6.0], **DD)
    jac = jacrev(lambda b: implicit_fixed_point(g, b.detach() / 3, b))(b)
    assert_close(jac, torch.eye(2, **DD) / 3)


########################################################################
# Minima: ridge regression, z*(lam, y) = (X^T X + lam I)^-1 X^T y

X = _randn(7, N, seed=2)
Y = _randn(7, seed=3)


def ridge_objective(z: Tensor, lam: Tensor, y: Tensor) -> Tensor:
    return 0.5 * ((X @ z - y) ** 2).sum() + 0.5 * lam * (z**2).sum()


def ridge_closed(lam: Tensor, y: Tensor) -> Tensor:
    h = X.mT @ X + lam * torch.eye(N, **DD)
    return torch.linalg.inv(h) @ (X.mT @ y)


@pytest.mark.parametrize("method", ["cg", "gmres"])
def test_argmin_all_nestings(method: str) -> None:
    lam = torch.tensor(0.3, **DD)

    def f(lam: Tensor) -> Tensor:
        z = implicit_argmin(
            ridge_objective,
            ridge_closed(lam, Y).detach(),
            lam,
            Y,
            method=method,
        )
        return z.sum()

    def ref(lam: Tensor) -> Tensor:
        return ridge_closed(lam, Y).sum()

    for order in (1, 2, 3):
        expected = _nest(ref, "r" * order)(lam)
        for modes in itertools.product("rf", repeat=order):
            assert_close(_nest(f, "".join(modes))(lam), expected)


def test_argmin_data_gradient() -> None:
    lam = torch.tensor(0.3, **DD)
    f = lambda y: implicit_argmin(
        ridge_objective, ridge_closed(lam, y).detach(), lam, y
    )  # noqa: E731
    assert_close(jacrev(f)(Y), jacrev(lambda y: ridge_closed(lam, y))(Y))


def test_argmin_vmap() -> None:
    lam = torch.tensor([0.1, 0.3, 1.0], **DD)

    def f(lam: Tensor) -> Tensor:
        z = implicit_argmin(
            ridge_objective, ridge_closed(lam, Y).detach(), lam, Y
        )
        return z.sum()

    batched = vmap(jacfwd(jacrev(f)))(lam)
    looped = torch.stack(
        [jacfwd(jacrev(lambda l: ridge_closed(l, Y).sum()))(l) for l in lam]
    )
    assert_close(batched, looped)


########################################################################
# torch.compile: first derivatives only (see `implicit_solve`)


@pytest.mark.xfail(
    raises=Exception,
    strict=True,
    reason="Dynamo cannot trace a Function applied in another Function's "
    "backward (the response solve); it raises (PyTorch 2.7-2.14)",
)
def test_compile_jacrev_raises() -> None:
    compiled = torch.compile(
        jacrev(lambda p: root_solution(p, maxiter=8)),
        fullgraph=True,
        backend="aot_eager",
    )
    compiled(P)


def test_compile_backward() -> None:
    p = P.clone().requires_grad_(True)
    compiled = torch.compile(
        lambda p: root_solution(p, maxiter=8).sum(), backend="aot_eager"
    )
    (grad,) = torch.autograd.grad(compiled(p), p)
    assert_close(grad, jacrev(lambda p: newton(p).sum())(P))


def test_root_vmap_unbatched_root() -> None:
    """A root shared by all `vmap` lanes (only an unrelated input varies)."""

    def shifted(z: Tensor, p: Tensor, q: Tensor) -> Tensor:
        return residual(z, p) + 0.0 * q

    z0 = newton(P)
    q = torch.tensor([1.0, 2.0, 3.0], **DD)
    out = vmap(lambda q: implicit_root(shifted, z0, P, q) * q)(q)
    assert_close(out, q.unsqueeze(-1) * z0)


def test_root_forward_without_parameter_tangent() -> None:
    z0 = newton(P)
    t = torch.tensor(2.0, **DD)
    jac = jacfwd(lambda t: implicit_root(residual, z0, P) * t)(t)
    assert_close(jac, z0)


def test_root_vmap_preconditioner() -> None:
    """
    The preconditioner sees one system at a time, also when the derivative
    is taken outside of `vmap` (the batched problem solves the response).
    """

    def jacobi(r: Tensor, z: Tensor, p: Tensor) -> Tensor:
        assert r.ndim == 1 and z.ndim == 1 and p.ndim == 1
        return r / (A.diagonal() + 3 * z**2)

    def solve(p: Tensor) -> Tensor:
        return implicit_root(
            residual, newton(p.detach()), p, preconditioner=jacobi
        )

    p = torch.stack((P, 0.5 * P, -P))
    for outer in (jacrev, jacfwd):
        jac = outer(lambda p: vmap(solve)(p).sum(-1))(p)
        assert_close(jac, jacrev(lambda p: newton(p).sum(-1))(p))

    leaf = p.clone().requires_grad_(True)
    vmap(solve)(leaf).sum().backward()
    assert leaf.grad is not None
    assert_close(leaf.grad, jacrev(lambda p: newton(p).sum())(p))


def test_package_namespace() -> None:
    """`storch.implicit` is this package, not `storch.linsolve.implicit`."""
    import tad_mctc.storch as storch
    import tad_mctc.storch.implicit.root as root

    assert storch.implicit.implicit_root is root.implicit_root
    assert storch.implicit_fixed_point is root.implicit_fixed_point
    assert not hasattr(storch, "krylov")
