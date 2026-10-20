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
Test `dense_solve`: LU solve with exact derivatives of every order.

The oracle is `inv(A) @ B`, whose derivatives are exact in every nesting and
under `vmap` (unlike `torch.linalg.solve` and `lu_solve`).
"""

from __future__ import annotations

import itertools

import pytest
import torch
from torch.func import jacfwd, jacrev, vmap

from tad_mctc._version import __tversion__
from tad_mctc.exceptions import DtypeError
from tad_mctc.storch.linsolve import dense_solve
from tad_mctc.tools.testing import requires_compile
from tad_mctc.typing import Tensor

from ..conftest import DEVICE

DD = {"dtype": torch.double, "device": DEVICE}


def _system(n: int = 4, symmetric: bool = False, seed: int = 0):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    s = torch.randn(
        3, n, n, generator=gen, dtype=torch.double, device="cpu"
    ).to(DEVICE)
    c = torch.randn(
        3, n, n, generator=gen, dtype=torch.double, device="cpu"
    ).to(DEVICE)
    m = torch.randn(
        3, n, 2, generator=gen, dtype=torch.double, device="cpu"
    ).to(DEVICE)
    if symmetric:
        a0 = s @ s.mT + n * torch.eye(n, **DD)
        c = c + c.mT
    else:
        a0 = s + 2 * n**0.5 * torch.eye(n, **DD)
    return a0, c, m


def _f(solve, vector: bool):
    def f(t: Tensor, a0: Tensor, c: Tensor, m: Tensor) -> Tensor:
        a = a0 + t * c
        b = m[..., 0] * (1 + t) if vector else m * (1 + t)
        return (solve(a, b) ** 2).sum()

    return f


def _inv(a: Tensor, b: Tensor) -> Tensor:
    if b.ndim == a.ndim - 1:
        return (torch.linalg.inv(a) @ b.unsqueeze(-1)).squeeze(-1)
    return torch.linalg.inv(a) @ b


def _nest(f, modes: str):
    for mode in modes:
        f = (jacrev if mode == "r" else jacfwd)(f, argnums=0)
    return f


@pytest.mark.parametrize("symmetric", [False, True])
@pytest.mark.parametrize("vector", [False, True])
@pytest.mark.parametrize("order", [1, 2, 3])
def test_all_nestings(symmetric: bool, vector: bool, order: int) -> None:
    a0, c, m = _system(symmetric=symmetric)
    args = (a0[0], c[0], m[0])
    t = torch.tensor(0.1, **DD)
    for modes in itertools.product("rf", repeat=order):
        ref = _nest(_f(_inv, vector), modes)(t, *args)
        got = _nest(_f(dense_solve, vector), modes)(t, *args)
        assert pytest.approx(ref.cpu(), rel=1e-10, abs=1e-12) == got.cpu()


@pytest.mark.parametrize("modes", ["r", "rr", "fr", "rf", "ff", "fff", "rrr"])
def test_vmap(modes: str) -> None:
    a0, c, m = _system()
    t = torch.tensor(0.1, **DD)
    g = _nest(_f(dense_solve, False), modes)
    batched = vmap(g, in_dims=(None, 0, 0, 0))(t, a0, c, m)
    looped = torch.stack(
        [_nest(_f(_inv, False), modes)(t, a0[i], c[i], m[i]) for i in range(3)]
    )
    assert pytest.approx(looped.cpu(), rel=1e-10, abs=1e-12) == batched.cpu()


def test_value_and_shapes() -> None:
    a0, _, m = _system()
    # bitwise equal in PyTorch 2.14, roundoff-level differences in 2.7
    tol = {"rel": 1e-13, "abs": 1e-15}
    ref = torch.linalg.solve(a0, m).cpu()
    assert pytest.approx(ref, **tol) == dense_solve(a0, m).cpu()
    v = m[..., 0]
    ref = torch.linalg.solve(a0, v).cpu()
    assert pytest.approx(ref, **tol) == dense_solve(a0, v).cpu()
    # broadcast: one matrix, batch of right-hand sides and vice versa
    out = dense_solve(a0[0], m)
    assert out.shape == m.shape
    assert pytest.approx(torch.linalg.solve(a0[0], m).cpu()) == out.cpu()
    out = dense_solve(a0, m[0])
    assert out.shape == m.shape
    assert pytest.approx(torch.linalg.solve(a0, m[0]).cpu()) == out.cpu()


def test_ill_conditioned() -> None:
    """`tanh(Y) = (E + I)^-1 (E - I)` with `E = exp(2Y)`, `|E| ~ 1e7`."""
    gen = torch.Generator(device="cpu").manual_seed(1)
    q, _ = torch.linalg.qr(
        torch.randn(4, 4, generator=gen, dtype=torch.double, device="cpu")
    )
    q = q.to(DEVICE)
    y = torch.tensor([-1.0, 0.1, 5.0, 8.0], **DD)
    e = q @ torch.diag(torch.exp(2 * y)) @ q.mT
    eye = torch.eye(4, **DD)
    tanh = q @ torch.diag(torch.tanh(y)) @ q.mT

    got = dense_solve(e + eye, e - eye)
    assert (got - tanh).abs().max() < 1e-8


def test_gradcheck() -> None:
    a0, _, m = _system(n=3)
    a = a0[0].clone().requires_grad_(True)
    b = m[0].clone().requires_grad_(True)
    assert torch.autograd.gradcheck(dense_solve, (a, b))
    assert torch.autograd.gradgradcheck(dense_solve, (a, b))
    assert torch.autograd.gradcheck(dense_solve, (a, b), check_forward_ad=True)


def test_singular_raises() -> None:
    a = torch.zeros(3, 3, **DD)
    with pytest.raises(RuntimeError):
        dense_solve(a, torch.ones(3, **DD))


@pytest.mark.parametrize("lead", [False, True])
@pytest.mark.parametrize("outer", [jacrev, jacfwd])
def test_hessian_vector_parameter(lead: bool, outer) -> None:
    """
    Hessians in a vector parameter `vmap` the backward over cotangents; with a
    leading singleton batch dimension, `torch.linalg.solve` mixes the `vmap`
    dimension into the broadcast batch unless the batch shapes are equal.
    """
    a0, c, m = _system(symmetric=True)
    x = torch.tensor([0.1, -0.2, 0.05], **DD)

    def make(solve):
        def f(x: Tensor) -> Tensor:
            a = a0[0] + torch.einsum("i,ijk->jk", x, c)
            b = m[0] * (1 + x.sum())
            if lead:
                a, b = a[None], b[None]
            return (solve(a, b) ** 2).sum()

        return f

    ref = outer(jacrev(make(_inv)))(x)
    assert (
        pytest.approx(ref.cpu(), rel=1e-10, abs=1e-12)
        == outer(jacrev(make(dense_solve)))(x).cpu()
    )


def _compile(f):
    return torch.compile(f, fullgraph=True, backend="aot_eager")


@requires_compile
def test_compile_backward() -> None:
    a0, c, m = _system(n=3)
    t = torch.tensor(0.1, **DD, requires_grad=True)
    f = _f(dense_solve, False)
    (grad,) = torch.autograd.grad(_compile(f)(t, a0[0], c[0], m[0]), t)
    ref = jacrev(_f(_inv, False))(t.detach(), a0[0], c[0], m[0])
    assert pytest.approx(ref.cpu(), rel=1e-10) == grad.cpu()


@requires_compile
@pytest.mark.skipif(
    __tversion__ < (2, 11, 0), reason="jacrev under compile needs >= 2.11"
)
def test_compile_jacrev() -> None:
    a0, c, m = _system(n=3)
    t = torch.tensor(0.1, **DD)
    got = _compile(jacrev(_f(dense_solve, False)))(t, a0[0], c[0], m[0])
    ref = jacrev(_f(_inv, False))(t, a0[0], c[0], m[0])
    assert pytest.approx(ref.cpu(), rel=1e-10) == got.cpu()


@requires_compile
@pytest.mark.skipif(
    __tversion__ < (2, 11, 0), reason="raises before 2.11 (documented)"
)
@pytest.mark.xfail(
    strict=True,
    reason="PyTorch 2.11-2.14: nested derivatives of a Function inside "
    "compiled code miss its backward's dependence on saved intermediates",
)
def test_compile_hessian_is_wrong() -> None:
    a0, c, m = _system(n=3)
    t = torch.tensor(0.1, **DD)
    got = _compile(jacrev(jacrev(_f(dense_solve, False))))(t, a0[0], c[0], m[0])
    ref = jacrev(jacrev(_f(_inv, False)))(t, a0[0], c[0], m[0])
    assert pytest.approx(ref.cpu(), rel=1e-10) == got.cpu()


def _cholesky(a: Tensor, b: Tensor) -> Tensor:
    return dense_solve(a, b, method="cholesky")


@pytest.mark.parametrize("vector", [False, True])
@pytest.mark.parametrize("order", [1, 2, 3])
def test_cholesky_all_nestings(vector: bool, order: int) -> None:
    a0, c, m = _system(symmetric=True)
    args = (a0[0], c[0], m[0])
    t = torch.tensor(0.1, **DD)
    for modes in itertools.product("rf", repeat=order):
        ref = _nest(_f(_inv, vector), modes)(t, *args)
        got = _nest(_f(_cholesky, vector), modes)(t, *args)
        assert pytest.approx(ref.cpu(), rel=1e-10, abs=1e-12) == got.cpu()


@pytest.mark.parametrize("modes", ["r", "rr", "fr", "fff"])
def test_cholesky_vmap(modes: str) -> None:
    a0, c, m = _system(symmetric=True)
    t = torch.tensor(0.1, **DD)
    g = _nest(_f(_cholesky, False), modes)
    batched = vmap(g, in_dims=(None, 0, 0, 0))(t, a0, c, m)
    looped = torch.stack(
        [_nest(_f(_inv, False), modes)(t, a0[i], c[i], m[i]) for i in range(3)]
    )
    assert pytest.approx(looped.cpu(), rel=1e-10, abs=1e-12) == batched.cpu()


def test_cholesky_gradient_is_symmetric() -> None:
    """Only symmetric perturbations of `A` count, as for `cholesky`."""
    a0, _, m = _system(symmetric=True)
    a = a0[0].clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(_cholesky(a, m[0]).sum(), a)
    (ref,) = torch.autograd.grad(_inv(a, m[0]).sum(), a)
    assert pytest.approx(0.5 * (ref + ref.mT).cpu()) == grad.cpu()

    s = a0[0].clone().requires_grad_(True)
    sym = lambda s: _cholesky(0.5 * (s + s.mT), m[0])  # noqa: E731
    assert torch.autograd.gradcheck(sym, (s,))
    assert torch.autograd.gradgradcheck(sym, (s,))


def test_cholesky_value() -> None:
    a0, _, m = _system(symmetric=True)
    ref = torch.cholesky_solve(m, torch.linalg.cholesky(a0))
    assert torch.equal(_cholesky(a0, m), ref)


def test_cholesky_not_positive_definite() -> None:
    a = torch.diag(torch.tensor([1.0, -1.0], **DD))
    with pytest.raises(RuntimeError):
        _cholesky(a, torch.ones(2, **DD))


def test_unknown_method() -> None:
    a0, _, m = _system()
    with pytest.raises(ValueError, match="Unknown method"):
        dense_solve(a0, m, method="qr")  # type: ignore[arg-type]


# Shapes as `torch.linalg.solve`: (shape of a, shape of b); `None` raises
SHAPES = [
    ((3, 3), (3,)),
    ((3, 3), (3, 2)),
    ((2, 3, 3), (3,)),
    ((2, 3, 3), (2, 3)),
    ((2, 3, 3), (1, 3)),
    ((2, 3, 3), (3, 2)),
    ((2, 3, 3), (2, 3, 2)),
    ((2, 3, 3), (1, 3, 2)),
    ((1, 3, 3), (2, 3)),
    ((1, 3, 3), (4, 3, 2)),
    ((2, 1, 3, 3), (4, 3, 1)),
    ((2, 1, 3, 3), (3, 3)),
]


@pytest.mark.parametrize("method", ["lu", "cholesky"])
@pytest.mark.parametrize("shapes", SHAPES)
def test_shapes_as_linalg_solve(method, shapes) -> None:
    shape_a, shape_b = shapes
    gen = torch.Generator(device="cpu").manual_seed(1)
    s = torch.randn(
        *shape_a, generator=gen, dtype=torch.double, device="cpu"
    ).to(DEVICE)
    a = s @ s.mT + 3 * torch.eye(3, **DD)
    b = torch.randn(
        *shape_b, generator=gen, dtype=torch.double, device="cpu"
    ).to(DEVICE)
    try:
        ref = torch.linalg.solve(a, b)
    except RuntimeError:
        with pytest.raises(RuntimeError):
            dense_solve(a, b, method=method)
        return

    got = dense_solve(a, b, method=method)
    assert got.shape == ref.shape
    assert pytest.approx(ref.cpu(), rel=1e-12, abs=1e-12) == got.cpu()


def test_vmap_shared_vector() -> None:
    a0, c, m = _system()
    b = m[0, :, 0]
    got = vmap(dense_solve, in_dims=(0, None))(a0, b)
    assert pytest.approx(torch.linalg.solve(a0, b).cpu()) == got.cpu()

    # the matrix gradient through the shared vector
    f = lambda a: dense_solve(a, b).pow(2).sum()  # noqa: E731
    ref = jacrev(lambda a: _inv(a, b.expand(3, -1)).pow(2).sum())(a0)
    assert pytest.approx(ref.cpu()) == jacrev(f)(a0).cpu()


@pytest.mark.parametrize("dtype", [torch.cdouble, torch.int64])
def test_rejects_non_real(dtype: torch.dtype) -> None:
    a = torch.eye(2, dtype=dtype, device=DEVICE)
    with pytest.raises(DtypeError, match="matrix"):
        dense_solve(a, torch.ones(2, **DD))
    with pytest.raises(DtypeError, match="right-hand side"):
        dense_solve(torch.eye(2, **DD), torch.ones(2, dtype=dtype))


@requires_compile
def test_compile_forward_mode_raises() -> None:
    a0, c, m = _system(n=3)
    t = torch.tensor(0.1, **DD)
    compiled = _compile(jacfwd(_f(dense_solve, False)))
    with pytest.raises(
        Exception, match="Forward-mode derivatives|NotImplementedError"
    ):
        compiled(t, a0[0], c[0], m[0])


@pytest.mark.parametrize("method", ["lu", "cholesky"])
def test_rhs_only_derivatives(method) -> None:
    """Derivatives with respect to the right-hand side alone."""
    a0, _, m = _system(symmetric=True)
    a, b = a0[0], m[0]
    inv = torch.linalg.inv(a)
    ones = torch.ones_like(b)

    _b = b.clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(dense_solve(a, _b, method).sum(), _b)
    assert pytest.approx((inv.mT @ ones).cpu()) == grad.cpu()

    tangent = torch.func.jvp(lambda b: dense_solve(a, b, method), (b,), (ones,))
    assert pytest.approx((inv @ ones).cpu()) == tangent[1].cpu()
