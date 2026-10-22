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
Acceptance test of the matrix-free solve of a charge model (EEQ).

The model has the form of the EEQ: Gaussian charges of radii ``alpha``,
hardnesses ``eta`` and electronegativities ``chi`` (parameters of the
magnitude of the EEQ parameters for H, C, N and O), and a total charge.
The charges are the minimizer of

    E(q) = chi^T q + 1/2 q^T A q   subject to   sum(q) = Q,

with ``A_ii = eta_i + sqrt(2 / pi) / alpha_i`` and the pair terms
``erf(r_ij / sqrt(alpha_i^2 + alpha_j^2)) / r_ij``. The charges, the energy
and their derivatives with respect to the positions and the parameters
(forces, the Hessian, the parameter derivatives of the charges and of the
forces, and a third derivative) are compared with a dense solve of the
bordered system, for the matrix-free variants: projected CG, with the
balancing preconditioner, and refined from float32 products.

The model itself (parameters, coordination numbers, periodic systems) is
tested in tad-multicharge.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import pytest
import torch
from torch.func import jacfwd, jacrev

from tad_mctc.storch.linsolve import (
    balancing,
    cell_index,
    constrained_solve,
    require_convergence,
)
from tad_mctc.storch.pairwise import pair_cell_sums, pair_product
from tad_mctc.typing import Tensor

from ..conftest import DEVICE

DD = {"device": DEVICE, "dtype": torch.double}

# radius (bohr), hardness and electronegativity (hartree) of H, C, N, O
PARAMS = {
    "H": (0.84, 0.47, 1.24),
    "C": (1.39, 0.18, 1.26),
    "N": (1.54, 0.11, 1.32),
    "O": (1.24, 0.17, 1.54),
}
TOTAL = 1.0


def kernel(r: Tensor, pi: tuple[Tensor, ...], pj: tuple[Tensor, ...]) -> Tensor:
    return torch.erf(r * torch.rsqrt(pi[0] ** 2 + pj[0] ** 2)) / r


def diagonal(alpha: Tensor, eta: Tensor) -> Tensor:
    return eta + math.sqrt(2 / math.pi) / alpha


def structure(side: tuple[int, int, int]) -> tuple[Tensor, ...]:
    """Atoms on a jittered grid of 2.6 bohr, with H, C, N and O."""
    gen = torch.Generator(device="cpu").manual_seed(0)
    axes = [torch.arange(s, dtype=torch.double) for s in side]
    grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    positions = 2.6 * grid + 0.5 * torch.rand(
        grid.shape, generator=gen, dtype=torch.double
    )
    n = positions.shape[0]
    symbols = [list(PARAMS)[i % 4] for i in range(n)]
    alpha, eta, chi = (
        torch.tensor([PARAMS[s][k] for s in symbols], dtype=torch.double)
        for k in range(3)
    )
    return tuple(t.to(DEVICE) for t in (positions, alpha, eta, chi))


def matvec(q: Tensor, pos: Tensor, alpha: Tensor, eta: Tensor) -> Tensor:
    return diagonal(alpha, eta) * q + pair_product(kernel, pos, q, alpha)


def fast_matvec(q: Tensor, pos: Tensor, alpha: Tensor, eta: Tensor) -> Tensor:
    pairs = pair_product(kernel, pos, q, alpha, dtype=torch.float)
    return diagonal(alpha, eta) * q + pairs


def preconditioner(pos: Tensor, alpha: Tensor, eta: Tensor):
    cells, k = cell_index(pos, 3.0)
    products = pair_cell_sums(kernel, pos, cells, k, alpha)
    diag = diagonal(alpha, eta).detach()
    products = products.index_put(
        (torch.arange(pos.shape[0], device=DEVICE), cells),
        diag,
        accumulate=True,
    )
    return balancing(diag, cells, products)


Charges = Callable[[Tensor, Tensor, Tensor, Tensor], Tensor]


def dense_charges(pos: Tensor, alpha: Tensor, eta: Tensor, chi: Tensor):
    """The bordered system, solved densely."""
    n = pos.shape[0]
    diff = pos.unsqueeze(-2) - pos.unsqueeze(-3)
    eye = torch.eye(n, **DD).bool()
    r = torch.sqrt(torch.where(eye, 1.0, (diff * diff).sum(-1)))
    a = torch.where(eye, 0.0, kernel(r, (alpha[:, None],), (alpha[None],)))
    a = a + torch.diag(diagonal(alpha, eta))
    ones = torch.ones(n, 1, **DD)
    kkt = torch.cat(
        (
            torch.cat((a, ones), 1),
            torch.cat((ones.T, torch.zeros(1, 1, **DD)), 1),
        )
    )
    rhs = torch.cat((-chi, torch.tensor([TOTAL], **DD)))
    return torch.linalg.solve(kkt, rhs)[:n]


def matrix_free(variant: str, inputs: tuple[Tensor, ...]) -> Charges:
    """
    The balancing preconditioner is built once, for the structure (outside
    of the transforms), and reused for the perturbed operators of the
    derivatives, as in a calculation.
    """
    prec = preconditioner(*inputs[:3]) if variant == "bnn" else None

    def charges(pos: Tensor, alpha: Tensor, eta: Tensor, chi: Tensor):
        n = pos.shape[0]
        result = constrained_solve(
            matvec,
            -chi,
            torch.ones(1, n, **DD),
            torch.tensor([TOTAL], **DD),
            pos,
            alpha,
            eta,
            preconditioner=prec,
            fast_matvec=fast_matvec if variant == "float32" else None,
            rtol=1e-13,
        )
        return require_convergence(result)

    return charges


VARIANTS = ["cg", "bnn", "float32"]


def energy(charges: Charges) -> Callable[..., Tensor]:
    def f(pos: Tensor, alpha: Tensor, eta: Tensor, chi: Tensor) -> Tensor:
        q = charges(pos, alpha, eta, chi)
        return (chi * q).sum() + 0.5 * (q * matvec(q, pos, alpha, eta)).sum()

    return f


def assert_close(x: Tensor, y: Tensor, tol: float) -> None:
    assert x.shape == y.shape
    assert pytest.approx(y.detach().cpu(), abs=tol, rel=tol) == x.detach().cpu()


@pytest.mark.parametrize("variant", VARIANTS)
def test_charges_and_energy(variant: str) -> None:
    inputs = structure((3, 3, 2))
    q = matrix_free(variant, inputs)(*inputs)
    ref = dense_charges(*inputs)
    assert_close(q, ref, 1e-11)
    assert abs(q.sum().item() - TOTAL) < 1e-13
    assert_close(
        energy(matrix_free(variant, inputs))(*inputs),
        energy(dense_charges)(*inputs),
        1e-11,
    )


@pytest.mark.parametrize("variant", VARIANTS)
def test_forces_and_hessian(variant: str) -> None:
    inputs = structure((2, 2, 2))
    out, ref = energy(matrix_free(variant, inputs)), energy(dense_charges)

    assert_close(jacrev(out)(*inputs), jacrev(ref)(*inputs), 1e-10)
    assert_close(
        jacfwd(jacrev(out))(*inputs), jacfwd(jacrev(ref))(*inputs), 1e-9
    )


@pytest.mark.parametrize("variant", VARIANTS)
def test_parameter_derivatives(variant: str) -> None:
    """dq/dtheta and the parameter derivatives of the forces."""
    inputs = structure((2, 2, 2))
    params = (1, 2, 3)  # alpha, eta, chi

    out = jacrev(matrix_free(variant, inputs), argnums=params)(*inputs)
    ref = jacrev(dense_charges, argnums=params)(*inputs)
    for x, y in zip(out, ref):
        assert_close(x, y, 1e-10)
    # the total charge does not depend on the parameters
    for x in out:
        assert x.sum(0).abs().max() < 1e-12

    def forces(charges: Charges):
        return jacrev(energy(charges))

    out = jacrev(forces(matrix_free(variant, inputs)), argnums=params)(*inputs)
    ref = jacrev(forces(dense_charges), argnums=params)(*inputs)
    for x, y in zip(out, ref):
        assert_close(x, y, 1e-9)


def test_third_derivative() -> None:
    """The radius derivative of the Hessian (a fit to Hessians)."""
    inputs = structure((2, 2, 1))

    def hessian(charges: Charges):
        return jacfwd(jacrev(energy(charges)))

    out = jacrev(hessian(matrix_free("bnn", inputs)), argnums=1)(*inputs)
    ref = jacrev(hessian(dense_charges), argnums=1)(*inputs)
    assert_close(out, ref, 1e-8)
