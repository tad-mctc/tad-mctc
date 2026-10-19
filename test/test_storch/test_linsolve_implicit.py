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
Test the implicit derivatives of the iterative linear solve.

The derivatives are compared with dense direct solves differentiated by
autograd. The dense oracles are `cholesky_solve` (SPD) and `lu_solve`
(general), because the forward-mode rule of `torch.linalg.solve` is not
differentiable: derivatives taken over it (`jacrev(jacfwd(f))`,
`jacfwd(jacfwd(f))`) are wrong (PyTorch 2.7 to 2.14).

SPD matrices are parametrized as ``A = S S^T + n I``, so that a perturbation
of the parameters never leaves the domain of CG.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import torch
from torch.autograd import gradcheck, gradgradcheck
from torch.func import jacfwd, jacrev, jvp, vmap

from tad_mctc._version import __tversion__
from tad_mctc.storch.linsolve import (
    STATUS_CONVERGED,
    LinearSolveResult,
    bicgstab,
    cg,
    implicit_solve,
)
from tad_mctc.tools.testing import requires_compile
from tad_mctc.typing import DD, Tensor

from ..conftest import DEVICE
from ..utils import compile_fullgraph

METHODS = ["cg", "bicgstab", "gmres"]


def matvec(x: Tensor, a: Tensor) -> Tensor:
    return torch.einsum("...ij,...j->...i", a, x)


def jacobi(r: Tensor, a: Tensor) -> Tensor:
    return r / a.diagonal(dim1=-2, dim2=-1)


def _randn(*shape: int, seed: int) -> Tensor:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(*shape, generator=gen, dtype=torch.double).to(DEVICE)


def build(method: str, s: Tensor) -> Tensor:
    """Operator from free parameters: SPD for CG, general otherwise."""
    n = s.shape[-1]
    eye = torch.eye(n, dtype=s.dtype, device=s.device)
    if method == "cg":
        return s @ s.mT + n * eye
    return s + 2 * n**0.5 * eye


def oracle(method: str, a: Tensor, b: Tensor) -> Tensor:
    """Dense solve whose autograd derivatives are correct to any order."""
    if method == "cg":
        chol = torch.linalg.cholesky(a)
        return torch.cholesky_solve(b.unsqueeze(-1), chol).squeeze(-1)
    lu, pivots = torch.linalg.lu_factor(a)
    return torch.linalg.lu_solve(lu, pivots, b.unsqueeze(-1)).squeeze(-1)


def implicit(method: str, a: Tensor, b: Tensor, **kwargs: Any) -> Tensor:
    return implicit_solve(matvec, b, a, method=method, **kwargs).solution  # type: ignore[arg-type]


def data(n: int = 4, seed: int = 0) -> tuple[Tensor, Tensor]:
    return _randn(n, n, seed=seed), _randn(n, seed=seed + 1)


def assert_close(x: Any, y: Any, tol: float = 1e-10) -> None:
    if isinstance(x, tuple):
        for xi, yi in zip(x, y):
            assert_close(xi, yi, tol)
        return
    assert x.shape == y.shape
    assert pytest.approx(y.cpu(), abs=tol, rel=tol) == x.cpu()


########################################################################
# Derivatives against dense oracles


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("transform", [jacrev, jacfwd])
def test_first_order(method: str, transform: Callable) -> None:
    s, b = data()

    def f(solver: Callable) -> Callable:
        return lambda s, b: solver(method, build(method, s), b)

    jac = transform(f(implicit), argnums=(0, 1))(s, b)
    ref = transform(f(oracle), argnums=(0, 1))(s, b)
    assert_close(jac, ref)


@pytest.mark.parametrize("method", METHODS)
def test_first_order_matrix(method: str) -> None:
    """
    The gradient with respect to the matrix itself is ``-lambda x^T``,
    as for a dense solve (not symmetrized).
    """
    s, b = data()
    a = build(method, s)

    jac = jacrev(lambda a: implicit(method, a, b))(a)
    lu, pivots = torch.linalg.lu_factor(a)
    ref = jacrev(
        lambda a: torch.linalg.lu_solve(*torch.linalg.lu_factor(a), b[:, None])
    )(a).squeeze(1)
    assert_close(jac, ref)
    del lu, pivots


NESTED = {
    "rev-rev": lambda f: jacrev(jacrev(f)),
    "fwd-rev": lambda f: jacfwd(jacrev(f)),
    "rev-fwd": lambda f: jacrev(jacfwd(f)),
    "fwd-fwd": lambda f: jacfwd(jacfwd(f)),
}


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("nested", NESTED)
def test_second_order(method: str, nested: str) -> None:
    s, b = data()

    def f(solver: Callable) -> Callable:
        return lambda s: (solver(method, build(method, s), b) ** 2).sum()

    hess = NESTED[nested](f(implicit))(s)
    ref = NESTED[nested](f(oracle))(s)
    assert_close(hess, ref)


@pytest.mark.parametrize("method", METHODS)
def test_second_order_mixed(method: str) -> None:
    """Mixed matrix/rhs second derivatives."""
    s, b = data()

    def f(solver: Callable) -> Callable:
        return lambda s, b: (solver(method, build(method, s), b) ** 3).sum()

    for t in (jacrev, jacfwd):
        hess = t(jacrev(f(implicit), argnums=0), argnums=1)(s, b)
        ref = t(jacrev(f(oracle), argnums=0), argnums=1)(s, b)
        assert_close(hess, ref)


THIRD = {
    "rev-rev-rev": lambda f: jacrev(jacrev(jacrev(f))),
    "fwd-rev-rev": lambda f: jacfwd(jacrev(jacrev(f))),
    "rev-fwd-rev": lambda f: jacrev(jacfwd(jacrev(f))),
    "fwd-fwd-fwd": lambda f: jacfwd(jacfwd(jacfwd(f))),
}


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("nested", THIRD)
def test_third_order(method: str, nested: str) -> None:
    s, b = data(3)

    def f(solver: Callable) -> Callable:
        return lambda s: (solver(method, build(method, s), b) ** 2).sum()

    third = THIRD[nested](f(implicit))(s)
    ref = THIRD[nested](f(oracle))(s)
    assert_close(third, ref, tol=1e-9)


@pytest.mark.parametrize("method", METHODS)
def test_gradcheck(method: str) -> None:
    s, b = data(3)
    s.requires_grad_(True)
    b.requires_grad_(True)

    def f(s: Tensor, b: Tensor) -> Tensor:
        return implicit(method, build(method, s), b)

    assert gradcheck(f, (s, b))
    assert gradgradcheck(f, (s, b))
    assert gradcheck(f, (s, b), check_forward_ad=True, check_backward_ad=False)


@pytest.mark.parametrize("method", METHODS)
def test_autograd_double_backward(method: str) -> None:
    """Plain autograd with `create_graph=True`."""
    s, b = data()

    grads = []
    for solver in (implicit, oracle):
        _s = s.clone().requires_grad_(True)
        _b = b.clone().requires_grad_(True)
        x = solver(method, build(method, _s), _b)
        (g,) = torch.autograd.grad((x**2).sum(), _s, create_graph=True)
        (gg,) = torch.autograd.grad(g.sum(), _b)
        grads.append((g.detach(), gg))

    assert_close(grads[0], grads[1])


@pytest.mark.parametrize("method", METHODS)
def test_vdims(method: str) -> None:
    """Derivatives of a matrix-valued unknown."""
    s, b = data(6)
    b = b.reshape(2, 3)

    def op(x: Tensor, a: Tensor) -> Tensor:
        return matvec(x.flatten(-2), a).unflatten(-1, (2, 3))

    def f(s: Tensor) -> Tensor:
        a = build(method, s)
        return implicit_solve(op, b, a, method=method, vdims=2).solution  # type: ignore[arg-type]

    def ref(s: Tensor) -> Tensor:
        a = build(method, s)
        return oracle(method, a, b.flatten()).unflatten(-1, (2, 3))

    assert_close(jacrev(f)(s), jacrev(ref)(s))
    assert_close(jacfwd(f)(s), jacrev(ref)(s))


def test_matrix_free_parameters() -> None:
    """
    A matrix-free operator ``A(p) x = p * x + K x`` with a parameter vector
    and a fixed matrix passed as a parameter without derivatives.
    """
    k, b = data(5)
    k = k @ k.mT
    p = 1.0 + _randn(5, seed=7).abs()

    def op(x: Tensor, p: Tensor, k: Tensor) -> Tensor:
        return p * x + k @ x

    def f(p: Tensor, b: Tensor) -> Tensor:
        return implicit_solve(op, b, p, k).solution

    def ref(p: Tensor, b: Tensor) -> Tensor:
        return oracle("cg", torch.diag(p) + k, b)

    for t in (jacrev, jacfwd):
        assert_close(t(f, argnums=(0, 1))(p, b), t(ref, argnums=(0, 1))(p, b))
    assert_close(jacfwd(jacrev(f))(p, b), jacfwd(jacrev(ref))(p, b))


def test_static_parameters() -> None:
    """Non-tensor and integer parameters are passed through untouched."""
    s, b = data(4)
    mask = torch.tensor([1, 1, 0, 1], device=DEVICE)

    def op(x: Tensor, a: Tensor, scale: float, mask: Tensor) -> Tensor:
        return scale * matvec(x, a) + (1 - mask) * x

    def f(s: Tensor) -> Tensor:
        a = build("cg", s) * mask.unsqueeze(-1) * mask.unsqueeze(-2)
        return implicit_solve(op, b, a, 2.0, mask).solution

    def ref(s: Tensor) -> Tensor:
        a = build("cg", s) * mask.unsqueeze(-1) * mask.unsqueeze(-2)
        dense = 2.0 * a + torch.diag(1.0 - mask.double())
        return oracle("cg", dense, b)

    assert_close(f(s), ref(s))
    assert_close(jacrev(f)(s), jacrev(ref)(s))
    assert_close(jacfwd(f)(s), jacfwd(ref)(s))


def test_rmatvec() -> None:
    """An explicit transposed operator gives the same derivatives."""
    s, b = data()

    def rop(v: Tensor, a: Tensor) -> Tensor:
        return matvec(v, a.mT)

    def f(s: Tensor, rmatvec: Callable | None) -> Tensor:
        a = build("bicgstab", s)
        return implicit_solve(
            matvec, b, a, method="bicgstab", rmatvec=rmatvec
        ).solution

    assert_close(
        jacrev(lambda s: f(s, rop))(s), jacrev(lambda s: f(s, None))(s)
    )


def test_custom_method() -> None:
    """A solver function with the signature of `cg`."""
    s, b = data()

    def f(s: Tensor) -> Tensor:
        return implicit_solve(
            matvec, b, build("cg", s), method=bicgstab
        ).solution

    assert_close(
        jacrev(f)(s), jacrev(lambda s: oracle("cg", build("cg", s), b))(s)
    )


def test_unknown_method() -> None:
    s, b = data()
    with pytest.raises(ValueError, match="Unknown method"):
        implicit_solve(matvec, b, s, method="gmresx")  # type: ignore[arg-type]


########################################################################
# Independence of the numerical solve


@pytest.mark.parametrize("method", METHODS)
def test_independent_of_solver_settings(method: str) -> None:
    """
    Derivatives do not depend on the preconditioner (including one that
    depends on the differentiated matrix), on the initial guess, or on the
    number of iterations once the solves converge.
    """
    s, b = data(6)

    def f(**kwargs: Any) -> Callable:
        def g(s: Tensor) -> Tensor:
            return (implicit(method, build(method, s), b, **kwargs) ** 2).sum()

        return g

    x_ref = oracle(method, build(method, s), b)
    settings: list[dict[str, Any]] = [
        {},
        {"preconditioner": jacobi},
        {"x0": x_ref + 1e-6},  # one iteration
        {"rtol": 1e-14, "maxiter": 100},
    ]

    ref = jacrev(jacrev(f()))(s)
    for kwargs in settings:
        assert_close(jacrev(jacrev(f(**kwargs)))(s), ref)
        assert_close(jacfwd(jacrev(f(**kwargs)))(s), ref)


def test_independent_of_method() -> None:
    """CG and BiCGSTAB give the same derivatives for an SPD operator."""
    s, b = data(5)

    def f(method: str) -> Callable:
        return lambda s: (implicit(method, build("cg", s), b) ** 2).sum()

    assert_close(jacrev(jacrev(f("cg")))(s), jacrev(jacrev(f("bicgstab")))(s))


def test_unrolled_differs() -> None:
    """
    Differentiating a loosely converged CG trajectory gives a different
    derivative than the implicit rule, which is exact for the same solve.
    """
    s, b = data(12)

    def unrolled(s: Tensor) -> Tensor:
        return cg(matvec, b, build("cg", s), rtol=1e-3, atol=0).solution.sum()

    def implicit_loose(s: Tensor) -> Tensor:
        x = implicit_solve(matvec, b, build("cg", s), rtol=1e-3, atol=0)
        return x.solution.sum()

    ref = jacrev(lambda s: oracle("cg", build("cg", s), b).sum())(s)
    err_unrolled = (jacrev(unrolled)(s) - ref).abs().max()
    err_implicit = (jacrev(implicit_loose)(s) - ref).abs().max()
    assert err_implicit < err_unrolled


########################################################################
# Failures and transforms


@pytest.mark.parametrize("method", METHODS)
def test_vmap(method: str) -> None:
    """`vmap` over systems that need different numbers of iterations."""
    s = _randn(3, 5, 5, seed=3)
    s[1] = s[1] * 1e-3  # almost a multiple of the identity
    b = _randn(3, 5, seed=4)

    def f(solver: Callable) -> Callable:
        def g(s: Tensor, b: Tensor) -> Tensor:
            return (solver(method, build(method, s), b) ** 2).sum()

        return g

    def loop(transform: Callable) -> Any:
        # The oracle loops: `vmap(jacrev(f))` of `lu_solve` is wrong
        # (PyTorch 2.14).
        out = [transform(f(oracle))(s[i], b[i]) for i in range(3)]
        if isinstance(out[0], tuple):
            return tuple(torch.stack(o) for o in zip(*out))
        return torch.stack(out)

    for t in (jacrev, jacfwd):
        out = vmap(t(f(implicit), argnums=(0, 1)))(s, b)
        assert_close(out, loop(lambda g: t(g, argnums=(0, 1))))

    out = vmap(jacfwd(jacrev(f(implicit))))(s, b)
    assert_close(out, loop(lambda g: jacfwd(jacrev(g))))


def test_failed_derivative_is_nan() -> None:
    """
    A derivative solve that fails yields NaN instead of a finite partial
    iterate, also for a single lane under `vmap`; other lanes are intact.
    """
    a = torch.eye(3, dtype=torch.double, device=DEVICE).repeat(2, 1, 1)
    a[1, 2, 2] = -1.0  # indefinite: CG breaks down
    b = torch.ones(2, 3, dtype=torch.double, device=DEVICE)

    def f(a: Tensor, b: Tensor) -> Tensor:
        return implicit_solve(matvec, b, a).solution.sum()

    grad = vmap(jacrev(f, argnums=(0, 1)))(a, b)
    for g in grad:
        assert torch.isfinite(g[0]).all()
        assert torch.isnan(g[1]).all()

    tangent = vmap(lambda a, b: jvp(lambda b: f(a, b), (b,), (b,))[1])(a, b)
    assert torch.isfinite(tangent[0])
    assert torch.isnan(tangent[1])

    # eager
    _b = b.clone().requires_grad_(True)
    result = implicit_solve(matvec, _b, a)
    assert result.status.tolist()[0] == STATUS_CONVERGED
    assert result.status.tolist()[1] != STATUS_CONVERGED
    result.solution.sum().backward()
    assert _b.grad is not None
    assert torch.isfinite(_b.grad[0]).all()
    assert torch.isnan(_b.grad[1]).all()


def test_result() -> None:
    s, b = data()
    a = build("cg", s).requires_grad_(True)

    result = implicit_solve(matvec, b, a)

    assert isinstance(result, LinearSolveResult)
    assert result.solution.requires_grad
    assert not result.residual.requires_grad
    assert result.converged.item()

    result = implicit_solve(matvec, b, a.detach())
    assert not result.solution.requires_grad


@pytest.mark.parametrize("method", METHODS)
def test_float32(method: str) -> None:
    s, b = data(5)
    s, b = s.float(), b.float()

    def f(solver: Callable) -> Callable:
        return lambda s: (solver(method, build(method, s), b) ** 2).sum()

    assert_close(jacrev(f(implicit))(s), jacrev(f(oracle))(s), tol=1e-4)
    assert_close(jacfwd(f(implicit))(s), jacrev(f(oracle))(s), tol=1e-4)


@requires_compile
@pytest.mark.usefixtures("reset_dynamo")
@pytest.mark.parametrize("method", METHODS)
def test_compile(method: str) -> None:
    """
    Compiled forward and reverse mode. A fixed `maxiter` keeps the unrolled
    loop small.
    """
    s, b = data(4)

    def f(s: Tensor) -> tuple[Tensor, Tensor]:
        result = implicit_solve(
            matvec, b, build(method, s), method=method, maxiter=8  # type: ignore[arg-type]
        )
        return (result.solution**2).sum(), result.status

    def grad(s: Tensor) -> Tensor:
        return jacrev(lambda s: f(s)[0])(s)

    value, status = compile_fullgraph(f)(s)
    assert_close(value, f(s)[0])
    assert status.item() == STATUS_CONVERGED

    # Before 2.11, Dynamo rejects the Function under `jacrev` ("must override
    # the setup_context staticmethod"); checked with 2.7, 2.8 and 2.10.
    if __tversion__ >= (2, 11, 0):
        assert_close(compile_fullgraph(grad)(s), grad(s))

    _s = s.clone().requires_grad_(True)
    compile_fullgraph(lambda s: f(s)[0])(_s).backward()
    assert _s.grad is not None
    assert_close(_s.grad, grad(s))


########################################################################
# Linearization


def test_linearize() -> None:
    """
    A cached linearization replaces the operator in the iterations of all
    solves with that operator, but never in the derivatives.
    """
    s, b = data(5)
    calls = {"matvec": 0, "linearize": 0, "operator": 0}

    def counting_matvec(x: Tensor, a: Tensor) -> Tensor:
        calls["matvec"] += 1
        return matvec(x, a.mT)

    def rop(v: Tensor, a: Tensor) -> Tensor:
        return matvec(v, a)

    def linearize(a: Tensor) -> Callable[[Tensor], Tensor]:
        calls["linearize"] += 1
        at = a.mT.clone()

        def operator(x: Tensor) -> Tensor:
            calls["operator"] += 1
            return matvec(x, at)

        return operator

    def f(s: Tensor, **kwargs: Any) -> Tensor:
        a = build("bicgstab", s)
        x = implicit_solve(
            counting_matvec, b, a, method="bicgstab", rmatvec=rop, **kwargs
        ).solution
        return (x**2).sum()

    ref = jacfwd(jacrev(lambda s: f(s)))(s)
    calls.update(matvec=0, linearize=0, operator=0)

    hess = jacfwd(jacrev(lambda s: f(s, linearize=linearize)))(s)
    assert_close(hess, ref)
    assert calls["linearize"] > 0
    assert calls["operator"] > calls["matvec"]

    # the transposed linearization of a symmetric operator is the same
    out = implicit_solve(
        matvec, b, build("cg", s), linearize=lambda a: (lambda x: matvec(x, a))
    )
    assert_close(out.solution, oracle("cg", build("cg", s), b))


@requires_compile
@pytest.mark.usefixtures("reset_dynamo")
@pytest.mark.skipif(
    __tversion__ < (2, 11, 0), reason="Dynamo rejects `jacrev` before 2.11"
)
def test_compile_direct_inputs() -> None:
    """
    The matrix and the right-hand side as direct inputs of the compiled
    function: without a view in between, `jacrev` inside compiled code
    returns zero gradients for them (PyTorch 2.7 to 2.14).
    """
    s, b = data(4)
    a = build("cg", s)

    def f(a: Tensor, b: Tensor) -> Tensor:
        x = implicit_solve(matvec, b, a, maxiter=8).solution
        return (x**2).sum()

    grad = jacrev(f, argnums=(0, 1))
    compiled = compile_fullgraph(grad)(a, b)
    ref = grad(a, b)
    assert ref[0].abs().max() > 1e-3
    assert_close(compiled, ref)


@requires_compile
@pytest.mark.usefixtures("reset_dynamo")
@pytest.mark.skipif(
    __tversion__ < (2, 11, 0), reason="raises before 2.11 (documented)"
)
@pytest.mark.xfail(
    strict=True,
    reason="PyTorch 2.11-2.14: nested derivatives of a Function inside "
    "compiled code miss its backward's dependence on saved intermediates",
)
def test_compile_hessian_is_wrong() -> None:
    """Documents the limit to first derivatives under `torch.compile`."""
    s, b = data(4)

    def f(s: Tensor) -> Tensor:
        x = implicit_solve(matvec, b, build("cg", s), maxiter=8).solution
        return (x**2).sum()

    def ref(s: Tensor) -> Tensor:
        return (oracle("cg", build("cg", s), b) ** 2).sum()

    hessian = compile_fullgraph(jacrev(jacrev(f)))(s)
    assert_close(hessian, jacrev(jacrev(ref))(s))
