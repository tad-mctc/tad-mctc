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
Test the tiled pair-kernel product against the dense kernel matrix.

The kernel is the Coulomb interaction of Gaussian charges, as in charge
models: ``erf(r / sqrt(s_i + s_j)) / r`` with per-atom ``s``.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import pytest
import torch
from torch.autograd import gradcheck, gradgradcheck
from torch.func import jacfwd, jacrev, vmap

from tad_mctc.storch import pairwise
from tad_mctc.storch.linsolve import (
    ConstrainedSolveResult,
    balancing,
    cell_index,
    constrained_solve,
    dense_solve,
    implicit_solve,
)
from tad_mctc.storch.pairwise import pair_cell_sums, pair_product
from tad_mctc.tools.testing import requires_compile
from tad_mctc.typing import Tensor

from ..conftest import DEVICE

DD = {"device": DEVICE, "dtype": torch.double}


def kernel(r: Tensor, pi: tuple[Tensor, ...], pj: tuple[Tensor, ...]) -> Tensor:
    return torch.erf(r * torch.rsqrt(pi[0] + pj[0])) / r


def dense(positions: Tensor, s: Tensor, mask: Tensor | None = None) -> Tensor:
    """The kernel matrix, zero on the diagonal and for masked pairs."""
    n = positions.shape[-2]
    diff = positions.unsqueeze(-2) - positions.unsqueeze(-3)
    valid = ~torch.eye(n, dtype=torch.bool, device=positions.device)
    if mask is not None:
        valid = valid & mask.unsqueeze(-1) & mask.unsqueeze(-2)
    r = torch.sqrt(torch.where(valid, (diff * diff).sum(-1), 1.0))
    k = kernel(r, (s.unsqueeze(-1),), (s.unsqueeze(-2),))
    return torch.where(valid, k, 0.0)


def _rand(*shape: int, seed: int, scale: float = 1.0) -> Tensor:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return scale * torch.rand(
        *shape, generator=gen, dtype=torch.double, device="cpu"
    ).to(DEVICE)


def system(n: int, batch: tuple[int, ...] = (), seed: int = 0):
    positions = _rand(*batch, n, 3, seed=seed, scale=2.0 * n ** (1 / 3))
    s = 0.5 + _rand(*batch, n, seed=seed + 1)
    x = _rand(*batch, n, seed=seed + 2) - 0.5
    return positions, s, x


def matmul(k: Tensor, x: Tensor) -> Tensor:
    return torch.einsum("...ij,...j->...i", k, x)


def assert_close(x: Tensor, y: Tensor, tol: float = 1e-10) -> None:
    assert x.shape == y.shape
    assert pytest.approx(y.detach().cpu(), abs=tol, rel=tol) == x.detach().cpu()


@pytest.mark.parametrize("tile", [None, 1, 3, 100])
def test_values(tile: int | None) -> None:
    positions, s, x = system(7)
    out = pair_product(kernel, positions, x, s, tile=tile)
    assert_close(out, matmul(dense(positions, s), x))


def test_batch_mask_broadcast() -> None:
    positions, s, x = system(6, (2,))
    mask = torch.ones(2, 6, dtype=torch.bool, device=DEVICE)
    mask[1, 4:] = False

    out = pair_product(kernel, positions, x, s, mask=mask, tile=2)
    assert_close(out, matmul(dense(positions, s, mask), x))
    assert (out[1, 4:] == 0).all()

    # several vectors for one structure
    xs = torch.stack((x[0], 2 * x[0], -x[0]))
    out = pair_product(kernel, positions[0], xs, s[0], tile=4)
    assert_close(out, matmul(dense(positions[0], s[0]), xs))


def test_broadcast_gradient() -> None:
    """The gradient of a vector broadcast over the batch is summed back."""
    positions, s, x = system(5, (2,))
    v = x[:1].clone().requires_grad_(True)  # (1, n), broadcast to (2, n)
    out = pair_product(kernel, positions, v, s, tile=2)
    (grad,) = torch.autograd.grad((out**2).sum(), v)
    (ref,) = torch.autograd.grad((matmul(dense(positions, s), v) ** 2).sum(), v)
    assert grad.shape == (1, 5)
    assert_close(grad, ref)

    v = x[0].clone().requires_grad_(True)  # (n,), broadcast to (2, n)
    out = pair_product(kernel, positions, v, s, tile=2)
    (grad,) = torch.autograd.grad((out**2).sum(), v)
    (ref,) = torch.autograd.grad((matmul(dense(positions, s), v) ** 2).sum(), v)
    assert grad.shape == (5,)
    assert_close(grad, ref)


def test_dtype() -> None:
    """float32 pair terms, float64 accumulation and gradients."""
    positions, s, x = system(8)
    positions.requires_grad_(True)
    out = pair_product(kernel, positions, x, s, dtype=torch.float)
    assert out.dtype == torch.double
    assert_close(out, matmul(dense(positions, s), x), 1e-6)

    (grad,) = torch.autograd.grad(out.sum(), positions)
    ref = torch.autograd.grad(matmul(dense(positions, s), x).sum(), positions)[
        0
    ]
    assert grad.dtype == torch.double
    assert_close(grad, ref, 1e-5)


@requires_compile
@pytest.mark.parametrize("dtype", [None, torch.float])
def test_fused(dtype) -> None:
    """The compiled tiles give the eager products and gradients."""
    positions, s, x = system(9, (2,))
    mask = torch.ones(2, 9, dtype=torch.bool, device=DEVICE)
    mask[1, 7:] = False
    positions.requires_grad_(True)

    def product(fused: bool) -> Tensor:
        return pair_product(
            kernel, positions, x, s, mask=mask, tile=4, dtype=dtype, fused=fused
        )

    fused, eager = product(True), product(False)
    assert_close(fused, eager, 1e-12 if dtype is None else 1e-6)
    grad = torch.autograd.grad((fused**2).sum(), positions)[0]
    ref = torch.autograd.grad((eager**2).sum(), positions)[0]
    assert_close(grad, ref, 1e-10 if dtype is None else 1e-5)

    # without a mask (all atoms interact)
    out = pair_product(kernel, positions, x, s, tile=4, dtype=dtype, fused=True)
    assert_close(out, _dense(positions, x, s), 1e-12 if dtype is None else 1e-6)

    # under a transform, the tiles are evaluated eagerly
    out = vmap(
        lambda v: pair_product(kernel, positions[0], v, s[0], fused=True)
    )(x)
    assert_close(
        out, _dense(positions[0].expand(2, 9, 3), x, s[0].expand(2, 9))
    )


@requires_compile
def test_fused_compiles_once() -> None:
    """Inputs that require gradients do not compile the tiles anew."""
    positions, s, x = system(13)

    def product(pos: Tensor, x: Tensor) -> Tensor:
        return pair_product(kernel, pos, x, s, tile=4, fused=True)

    first = product(positions, x)
    pos, v = positions.clone().requires_grad_(True), x.clone().requires_grad_()
    with torch._dynamo.config.patch(error_on_recompile=True):
        second = product(pos, v)
        torch.autograd.grad(second.sum(), (pos, v))
    assert_close(first, second, 1e-12)


def test_eager_tiles(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Whatever is evaluated eagerly stays within the eager tile budget, also
    with ``fused`` (whose tiles are much larger): the products without
    compiling, under transforms, beyond Dynamo's recompilation limit (which
    runs the compiled tile eagerly), and the derivatives.
    """
    n = 600
    positions, s, x = system(n)
    positions = positions.cpu().requires_grad_(True)
    s, x = s.cpu(), x.cpu()
    budget = pairwise._TILE_PAIRS_CPU // n
    ref = _dense(positions, x, s)
    ref_grad = torch.autograd.grad((ref**2).sum(), positions)[0]

    rows: list[int] = []
    block = pairwise._block

    def recorded(*args):
        rows.append(args[2].shape[-1])
        return block(*args)

    monkeypatch.setattr(pairwise, "_block", recorded)
    monkeypatch.setattr(pairwise, "is_compile_supported", lambda: False)

    def check(out: Tensor, expected: Tensor) -> None:
        assert_close(out, expected)
        assert len(rows) > 1 and max(rows) <= budget
        rows.clear()

    for tile in (None, 10 * n):
        out = pair_product(kernel, positions, x, s, tile=tile, fused=True)
        check(out, ref)

        grad = torch.autograd.grad((out**2).sum(), positions)[0]
        check(grad, ref_grad)

        out = jacrev(
            lambda v: pair_product(kernel, positions, v, s, fused=True).sum()
        )(x)
        check(out, dense(positions, s).sum(-2))

    # the compiled tile, run eagerly by Dynamo
    args = (
        kernel,
        None,
        *pairwise._tile_arguments(positions, (s,), None, 0, n),
    )
    out = pairwise._block_product_reduction(*args, x)  # pyright: ignore
    check(out, ref)


def test_cell_sums() -> None:
    positions, s, _ = system(9, (2,))
    mask = torch.ones(2, 9, dtype=torch.bool, device=DEVICE)
    mask[1, 7:] = False
    cells, k = cell_index(positions, 1.5, mask)

    out = pair_cell_sums(kernel, positions, cells, k, s, mask=mask, tile=4)
    z = torch.nn.functional.one_hot(cells.clamp(min=0), k).to(**DD)
    z = z * (cells >= 0).unsqueeze(-1)
    assert_close(out, dense(positions, s, mask) @ z)


def test_invalid() -> None:
    positions, s, x = system(5)
    with pytest.raises(ValueError, match="positions"):
        pair_product(kernel, positions[..., :2], x, s)
    with pytest.raises(ValueError, match="vector"):
        pair_product(kernel, positions, x[:-1], s)
    with pytest.raises(ValueError, match="parameter"):
        pair_product(kernel, positions, x, s[:-1])
    with pytest.raises(ValueError, match="tile"):
        pair_product(kernel, positions, x, s, tile=0)


########################################################################
# derivatives


@pytest.mark.grad
def test_gradcheck() -> None:
    positions, s, x = system(5)
    inputs = tuple(t.clone().requires_grad_(True) for t in (positions, x, s))

    def f(pos: Tensor, x: Tensor, s: Tensor) -> Tensor:
        return pair_product(kernel, pos, x, s, tile=2)

    assert gradcheck(f, inputs)
    assert gradgradcheck(f, inputs)


def _loss(product: Callable[..., Tensor]) -> Callable[..., Tensor]:
    w = _rand(6, seed=9)

    def f(pos: Tensor, x: Tensor, s: Tensor) -> Tensor:
        y = product(pos, x, s)
        return (w * y).sum() + (y * y).sum()

    return f


def _tiled(pos: Tensor, x: Tensor, s: Tensor) -> Tensor:
    return pair_product(kernel, pos, x, s, tile=4)


def _dense(pos: Tensor, x: Tensor, s: Tensor) -> Tensor:
    return matmul(dense(pos, s), x)


TRANSFORMS = {
    "jacrev": jacrev,
    "jacfwd": jacfwd,
    "jacrev-jacrev": lambda f, a: jacrev(jacrev(f, a), a),
    "jacfwd-jacrev": lambda f, a: jacfwd(jacrev(f, a), a),
    "jacrev-jacfwd": lambda f, a: jacrev(jacfwd(f, a), a),
    "jacfwd-jacfwd": lambda f, a: jacfwd(jacfwd(f, a), a),
}


@pytest.mark.parametrize("transform", TRANSFORMS)
@pytest.mark.parametrize("argnum", [0, 1, 2], ids=["positions", "x", "s"])
def test_transforms(transform: str, argnum: int) -> None:
    positions, s, x = system(6)
    inputs = (positions, x, s)
    t = TRANSFORMS[transform]
    assert_close(
        t(_loss(_tiled), argnum)(*inputs), t(_loss(_dense), argnum)(*inputs)
    )


def test_third_derivative() -> None:
    positions, s, x = system(4)

    def f(product: Callable[..., Tensor]):
        return lambda pos: product(pos, x, s).sum()

    out = jacfwd(jacrev(jacrev(f(_tiled))))(positions)
    assert_close(out, jacfwd(jacrev(jacrev(f(_dense))))(positions), 1e-9)


def test_vmap() -> None:
    positions, s, x = system(6, (3,))

    # batched positions, shared parameters, and the other way round
    out = vmap(_tiled, in_dims=(0, 0, None))(positions, x, s[0])
    assert_close(out, _dense(positions, x, s[0].expand(3, 6)))
    out = vmap(_tiled, in_dims=(None, 0, 0))(positions[0], x, s)
    assert_close(out, _dense(positions[0].expand(3, 6, 3), x, s))

    # a structure with extra batch dimensions under `vmap`
    out = vmap(_tiled, in_dims=(None, 0, None))(positions, x[0].expand(2, 6), s)
    assert_close(out, _dense(positions, x[0].expand(2, 3, 6), s))

    grad = vmap(jacrev(_loss(_tiled)))(positions, x, s)
    ref = vmap(jacrev(_loss(_dense)))(positions, x, s)
    assert_close(grad, ref)


########################################################################
# as the operator of a solve


def _operator(dtype=None):
    def matvec(x: Tensor, pos: Tensor, s: Tensor, diag: Tensor) -> Tensor:
        return diag * x + pair_product(kernel, pos, x, s, tile=5, dtype=dtype)

    return matvec


def _diag(s: Tensor) -> Tensor:
    return math.sqrt(2 / math.pi) / torch.sqrt(s) + 0.5


def test_implicit_solve() -> None:
    """Forces through the solve equal those of a dense Cholesky solve."""
    positions, s, b = system(12)

    def energy(pos: Tensor, solver: str) -> Tensor:
        if solver == "dense":
            a = dense(pos, s) + torch.diag(_diag(s))
            q = dense_solve(a, b, "cholesky")
        else:
            q = implicit_solve(
                _operator(), b, pos, s, _diag(s), method="cg", rtol=1e-13
            ).solution
        return (q * b).sum()

    for t in (jacrev, lambda f: jacfwd(jacrev(f))):
        out = t(lambda p: energy(p, "tiled"))(positions)
        ref = t(lambda p: energy(p, "dense"))(positions)
        assert_close(out, ref, 1e-8)


@pytest.mark.parametrize("dtype", [None, torch.float])
def test_constrained_balancing(dtype) -> None:
    """
    A charge model: constrained solve with the tiled operator and a
    balancing preconditioner from the cell sums, against a dense solve.
    """
    positions, s, b = system(40)
    diag = _diag(s)
    c = torch.ones(1, 40, **DD)
    total = torch.tensor([1.0], **DD)

    cells, k = cell_index(positions, 2.0)
    az = pair_cell_sums(kernel, positions, cells, k, s, dtype=dtype)
    az = az.scatter_add(
        -1, cells.unsqueeze(-1), diag.to(az.dtype).unsqueeze(-1)
    )
    prec = balancing(diag, cells, az)

    def charges(pos: Tensor, preconditioner) -> ConstrainedSolveResult:
        return constrained_solve(
            _operator(),
            b,
            c,
            total,
            pos,
            s,
            diag,
            preconditioner=preconditioner,
            rtol=1e-13,
        )

    result = charges(positions, prec)
    plain = charges(positions, None)
    assert result.converged.item()
    assert result.iterations < plain.iterations

    a = dense(positions, s) + torch.diag(diag)
    k_mat = torch.cat(
        (torch.cat((a, c.mT), -1), torch.cat((c, torch.zeros(1, 1, **DD)), -1))
    )
    ref = dense_solve(k_mat, torch.cat((b, total)), "lu")[:40]
    assert_close(result.solution, ref, 1e-9)

    grad = jacrev(lambda p: (charges(p, prec).solution ** 2).sum())(positions)

    def dense_charges(pos: Tensor) -> Tensor:
        a = dense(pos, s) + torch.diag(diag)
        k_mat = torch.cat(
            (
                torch.cat((a, c.mT), -1),
                torch.cat((c, torch.zeros(1, 1, **DD)), -1),
            )
        )
        return dense_solve(k_mat, torch.cat((b, total)), "lu")[:40]

    ref = jacrev(lambda p: (dense_charges(p) ** 2).sum())(positions)
    assert_close(grad, ref, 1e-8)
