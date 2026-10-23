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
Test the Jacobi and balancing preconditioners and the cell partition.

The test operator is the Coulomb matrix of Gaussian charges on a jittered
grid plus a positive diagonal, whose condition number grows with the size
through its smooth modes, as for charge models.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch.func import jacrev, vmap

from tad_mctc.storch.linsolve import (
    balancing,
    cell_index,
    cg,
    constrained_solve,
    jacobi,
)
from tad_mctc.typing import Tensor

from ..conftest import DEVICE

DD = {"device": DEVICE, "dtype": torch.double}


def matvec(x: Tensor, a: Tensor) -> Tensor:
    return torch.einsum("...ij,...j->...i", a, x)


def grid(side: int, seed: int = 0) -> Tensor:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    g = torch.arange(side, dtype=torch.double, device="cpu")
    pts = torch.stack(torch.meshgrid(g, g, g, indexing="ij"), -1)
    pts = 2.0 * pts.reshape(-1, 3)
    jitter = 0.4 * torch.rand(
        pts.shape, generator=gen, dtype=torch.double, device="cpu"
    )
    return (pts + jitter).to(DEVICE)


def coulomb(pos: Tensor) -> Tensor:
    """Gaussian charges of width 1 plus a hardness of 0.5."""
    r = torch.cdist(pos, pos)
    eye = torch.eye(pos.shape[-2], **DD)
    r = torch.where(eye.bool(), 1.0, r)
    a = torch.where(eye.bool(), 0.0, torch.erf(r / math.sqrt(2)) / r)
    return a + (math.sqrt(2 / math.pi) + 0.5) * eye


def cell_products(a: Tensor, cells: Tensor, k: int) -> Tensor:
    index = torch.where(cells < 0, k, cells)
    out = a.new_zeros(*a.shape[:-1], k + 1)
    return out.scatter_add(-1, index.unsqueeze(-2).expand_as(a), a)[..., :k]


def _bnn(a: Tensor, pos: Tensor, edge: float = 4.0, mask=None):
    cells, k = cell_index(pos, edge, mask)
    diag = a.diagonal(dim1=-2, dim2=-1)
    return balancing(diag, cells, cell_products(a, cells, k))


def test_iterations() -> None:
    """BNN cuts the CG iterations of a Coulomb matrix, Jacobi does not."""
    pos = grid(7)
    a = coulomb(pos)
    b = torch.sin(pos[:, 0])
    ref = torch.linalg.solve(a, b)

    results = {
        "plain": cg(matvec, b, a),
        "jacobi": cg(matvec, b, a, preconditioner=jacobi(a.diagonal())),
        "bnn": cg(matvec, b, a, preconditioner=_bnn(a, pos)),
    }
    for result in results.values():
        assert result.converged.item()
        assert pytest.approx(ref.cpu(), abs=1e-8) == result.solution.cpu()
    assert results["bnn"].iterations < 0.6 * results["plain"].iterations


def test_symmetric_positive() -> None:
    """x . M y == y . M x, and x . M x > 0."""
    pos = grid(4)
    a = coulomb(pos)
    m = _bnn(a, pos)
    gen = torch.Generator(device="cpu").manual_seed(1)
    x, y = torch.randn(
        2, pos.shape[0], generator=gen, dtype=torch.double, device="cpu"
    )
    x, y = x.to(DEVICE), y.to(DEVICE)

    assert pytest.approx((x * m(y)).sum().item(), rel=1e-12) == (
        (y * m(x)).sum().item()
    )
    assert (x * m(x)).sum() > 0


def test_coarse_space_exact() -> None:
    """
    For a residual of the form A Z u, the preconditioned residual is Z u:
    the coarse modes are solved exactly.
    """
    pos = grid(4)
    a = coulomb(pos)
    cells, k = cell_index(pos, 4.0)
    m = balancing(a.diagonal(), cells, cell_products(a, cells, k))
    z = torch.nn.functional.one_hot(cells, k).to(**DD)
    u = torch.linspace(-1, 1, k, **DD)
    assert pytest.approx((z @ u).cpu(), abs=1e-10) == m(a @ (z @ u)).cpu()


def test_float32_products() -> None:
    """Stored in float32, A Z gives a preconditioner symmetric to float32
    precision, with which CG converges to the default tolerance."""
    pos = grid(5)
    a = coulomb(pos)
    cells, k = cell_index(pos, 4.0)
    az = cell_products(a, cells, k).float()
    m = balancing(a.diagonal(), cells, az)
    b = torch.cos(pos[:, 1])

    result = cg(matvec, b, a, preconditioner=m)
    assert result.converged.item()
    x = torch.ones_like(b)
    assert pytest.approx((x * m(b)).sum().item(), rel=1e-6) == (
        (b * m(x)).sum().item()
    )


def test_batch_padding_and_empty_cells() -> None:
    """
    A padded batch: padding atoms (cell -1) get the diagonal only, and the
    smaller system has empty cells; both solve as they do alone.
    """
    big, small = grid(4), grid(3)
    n, ns = big.shape[0], small.shape[0]
    pos = torch.zeros(2, n, 3, **DD)
    pos[0], pos[1, :ns] = big, small
    mask = torch.zeros(2, n, dtype=torch.bool, device=DEVICE)
    mask[0], mask[1, :ns] = True, True

    a = torch.zeros(2, n, n, **DD)
    a[0], a[1, :ns, :ns] = coulomb(big), coulomb(small)
    a[1, ns:, ns:] = torch.eye(n - ns, **DD)
    b = torch.sin(pos[..., 0]) * mask

    cells, k = cell_index(pos, 3.0, mask)
    assert (cells[1, ns:] == -1).all()
    assert int(cells[1].max()) + 1 < k  # empty cells in the small system
    m = balancing(
        a.diagonal(dim1=-2, dim2=-1), cells, cell_products(a, cells, k)
    )

    result = cg(matvec, b, a, preconditioner=m)
    alone = cg(
        matvec,
        b[1, :ns],
        a[1, :ns, :ns],
        preconditioner=_bnn(a[1, :ns, :ns], small, 3.0),
    )
    assert result.converged.all()
    assert (
        pytest.approx(alone.solution.cpu(), abs=1e-9)
        == result.solution[1, :ns].cpu()
    )
    assert (result.solution[1, ns:] == 0).all()


def test_constrained_derivatives() -> None:
    """The derivative solves of a constrained solve reuse the preconditioner,
    also under `vmap` (`jacrev`)."""
    pos = grid(3)
    a = coulomb(pos)
    m = _bnn(a, pos)
    c = torch.ones(1, pos.shape[0], **DD)
    q = torch.tensor([0.5], **DD)

    def charges(b: Tensor, a: Tensor, precondition: bool) -> Tensor:
        return constrained_solve(
            matvec,
            b,
            c,
            q,
            a,
            preconditioner=m if precondition else None,
            rtol=1e-13,
        ).solution

    b = torch.sin(pos[:, 0])
    out = jacrev(charges, argnums=(0, 1))(b, a, True)
    ref = jacrev(charges, argnums=(0, 1))(b, a, False)
    for x, y in zip(out, ref):
        assert pytest.approx(y.cpu(), abs=1e-9) == x.cpu()

    batched = vmap(lambda b: charges(b, a, True))(torch.stack((b, 2 * b)))
    assert (
        pytest.approx(charges(2 * b, a, True).cpu(), abs=1e-10)
        == batched[1].cpu()
    )


def test_not_positive_definite() -> None:
    pos = grid(3)
    a = -coulomb(pos)
    cells, k = cell_index(pos, 4.0)
    with pytest.raises(torch.linalg.LinAlgError):
        balancing(-a.diagonal(), cells, cell_products(a, cells, k))


def test_invalid() -> None:
    pos = grid(3)
    a = coulomb(pos)
    cells, k = cell_index(pos, 4.0)
    with pytest.raises(ValueError, match="same number"):
        balancing(a.diagonal()[:-1], cells, cell_products(a, cells, k))
    with pytest.raises(ValueError, match="positive"):
        cell_index(pos, 0.0)


def test_cell_index() -> None:
    pos = torch.tensor(
        [
            [[0.0, 0, 0], [1.9, 0, 0], [2.1, 0, 0], [0.0, 0, 5.0]],
            [[10.0, 10, 10], [10.5, 10, 10], [0.0, 0, 0], [0.0, 0, 0]],
        ],
        **DD,
    )
    mask = torch.tensor([[True] * 4, [True, True, False, False]], device=DEVICE)
    cells, k = cell_index(pos, 2.0, mask)
    assert k == 3
    assert cells.tolist() == [[0, 0, 2, 1], [0, 0, -1, -1]]

    cells, k = cell_index(pos[0], 2.0)
    assert (k, cells.tolist()) == (3, [0, 0, 2, 1])

    # no points, and no points in the mask
    cells, k = cell_index(pos[..., :0, :], 2.0)
    assert (k, cells.shape) == (0, (2, 0))
    cells, k = cell_index(pos, 2.0, torch.zeros_like(mask))
    assert k == 0 and (cells == -1).all()


def test_max_cells() -> None:
    """A bound of the cells enlarges them; the preconditioner still works."""
    pos = grid(6)
    cells, k = cell_index(pos, 4.0)
    assert k == 27

    for bound in (8, 5, 1):
        bounded, kb = cell_index(pos, 4.0, max_cells=bound)
        assert 1 <= kb <= bound
        assert bounded.min() == 0 and int(bounded.max()) == kb - 1
    unchanged, kb = cell_index(pos, 4.0, max_cells=27)
    assert kb == 27 and unchanged.equal(cells)

    a = coulomb(pos)
    b = torch.sin(pos[:, 0])
    bounded, kb = cell_index(pos, 4.0, max_cells=8)
    m = balancing(a.diagonal(), bounded, cell_products(a, bounded, kb))
    result = cg(matvec, b, a, preconditioner=m)
    assert result.converged.item()

    with pytest.raises(ValueError, match="positive"):
        cell_index(pos, 4.0, max_cells=0)
