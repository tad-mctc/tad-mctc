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
r"""
Linear Solve: Preconditioners
=============================

Preconditioners ``M r ~ A^{-1} r`` for the Krylov solvers, as functions
``preconditioner(r, *params)`` (the operator arguments are ignored).

- :func:`jacobi`: the inverse diagonal.
- :func:`balancing`: a two-level balancing (BNN) preconditioner on a
  partition of the unknowns into cells, for operators whose largest
  eigenvalues belong to smooth modes, such as Coulomb matrices:

  .. math::

      M = P^\mathsf{T} D^{-1} P + Z E^{-1} Z^\mathsf{T}, \qquad
      P = I - A Z E^{-1} Z^\mathsf{T}, \qquad E = Z^\mathsf{T} A Z,

  with the cell indicators :math:`Z` and the diagonal :math:`D` of
  :math:`A`. It removes the modes spanned by :math:`Z` A-orthogonally.
  For a Coulomb matrix the condition number grows with the system size,
  but with cells of fixed size the iteration count barely does. An
  additive coarse correction (:math:`D^{-1} + Z E^{-1} Z^\mathsf{T}`)
  does not help there.
- :func:`cell_index`: a partition of points into cubic cells.

The preconditioners are built once from detached data, outside of
``vmap`` and ``torch.compile``, and are then applied under any transform
(the derivative solves of
:func:`~tad_mctc.storch.linsolve.implicit_solve` reuse them).
"""

from __future__ import annotations

import math
from typing import Any

import torch

from ...typing import Tensor
from .krylov import Operator

__all__ = ["balancing", "cell_index", "jacobi"]


def jacobi(diagonal: Tensor) -> Operator:
    """
    Jacobi preconditioner, ``M r = r / diag(A)``.

    Parameters
    ----------
    diagonal : Tensor
        Positive diagonal of :math:`A`, shape ``(..., n)``.

    Returns
    -------
    Callable[..., Tensor]
        The preconditioner ``(r, *params) -> M r``.
    """
    inverse = 1.0 / diagonal.detach()

    def apply(r: Tensor, *_: Any) -> Tensor:
        return inverse * r

    return apply


def balancing(diagonal: Tensor, cells: Tensor, products: Tensor) -> Operator:
    r"""
    Balancing (BNN) two-level preconditioner on a partition into cells.

    Parameters
    ----------
    diagonal : Tensor
        Positive diagonal of :math:`A`, shape ``(..., n)``.
    cells : Tensor
        Cell of each unknown, integers in ``[0, K)``, shape ``(..., n)``.
        ``-1`` leaves an unknown out of all cells (e.g. padding), so that it
        is preconditioned by its diagonal only. Empty cells are allowed.
    products : Tensor
        :math:`A Z`, the sums of the columns of :math:`A` over each cell,
        shape ``(..., n, K)``, e.g. from
        :func:`~tad_mctc.storch.pairwise.pair_cell_sums` plus the
        diagonal. It is stored as given; float32 halves the memory and
        keeps the preconditioner symmetric.

    Returns
    -------
    Callable[..., Tensor]
        The preconditioner ``(r, *params) -> M r``. It is symmetric, and
        positive definite if :math:`A` is.

    Raises
    ------
    ValueError
        The shapes do not match.
    torch.linalg.LinAlgError
        The coarse matrix :math:`E = Z^\mathsf{T} A Z` is not positive
        definite (:math:`A` is not).
    """
    n, k = products.shape[-2], products.shape[-1]
    if diagonal.shape[-1] != n or cells.shape[-1] != n:
        raise ValueError(
            f"The diagonal {tuple(diagonal.shape)}, the cells "
            f"{tuple(cells.shape)} and the cell products "
            f"{tuple(products.shape)} must have the same number of "
            "unknowns."
        )

    dtype = diagonal.dtype
    inverse = 1.0 / diagonal.detach()
    products = products.detach()

    # Unknowns outside of all cells go to an extra cell K, which is dropped.
    index = torch.where(cells < 0, k, cells).long()
    batch = torch.broadcast_shapes(index.shape[:-1], products.shape[:-2])
    index = index.expand(*batch, n)

    def to_cells(v: Tensor) -> Tensor:  # Z^T v, (..., n) -> (..., K)
        out = v.new_zeros(*v.shape[:-1], k + 1)
        return out.scatter_add(-1, index.expand_as(v), v)[..., :k]

    def from_cells(u: Tensor) -> Tensor:  # Z u, (..., K) -> (..., n)
        padded = torch.cat((u, u.new_zeros(*u.shape[:-1], 1)), dim=-1)
        return padded.gather(-1, index.expand(*u.shape[:-1], n))

    # E = Z^T A Z; an empty cell gets a unit diagonal, which decouples it.
    rows = index.unsqueeze(-1).expand(*batch, n, k)
    coarse = products.new_zeros(*batch, k + 1, k)
    coarse = coarse.scatter_add(-2, rows, products.expand(*batch, n, k))
    coarse = coarse[..., :k, :].to(dtype)
    count = to_cells(torch.ones(*batch, n, dtype=dtype, device=index.device))
    coarse = 0.5 * (coarse + coarse.mT) + torch.diag_embed(
        (count == 0).to(dtype)
    )
    factor = torch.linalg.cholesky(coarse)

    def coarse_solve(v: Tensor) -> Tensor:  # E^-1 v
        return torch.cholesky_solve(v.unsqueeze(-1), factor).squeeze(-1)

    def times_products(u: Tensor) -> Tensor:  # A Z u
        az = products.to(u.dtype) if products.dtype != u.dtype else products
        return torch.einsum("...nk,...k->...n", az, u)

    def times_products_t(v: Tensor) -> Tensor:  # (A Z)^T v
        az = products.to(v.dtype) if products.dtype != v.dtype else products
        return torch.einsum("...nk,...n->...k", az, v)

    def apply(r: Tensor, *_: Any) -> Tensor:
        cr = coarse_solve(to_cells(r))
        v = inverse * (r - times_products(cr))  # D^-1 P r
        return (
            v - from_cells(coarse_solve(times_products_t(v))) + from_cells(cr)
        )

    return apply


def cell_index(
    positions: Tensor, edge: float, mask: Tensor | None = None
) -> tuple[Tensor, int]:
    """
    Partition points into cubic cells of a given edge length.

    The cells of each system are numbered from zero, in the order of their
    integer coordinates; the batch shares the number of cells ``K`` (the
    largest of the batch), so some systems have empty cells. This reads the
    values of the positions and must run outside of ``vmap`` and
    ``torch.compile``.

    Parameters
    ----------
    positions : Tensor
        Cartesian coordinates, shape ``(..., n, 3)``.
    edge : float
        Edge length of the cells, in the units of the positions.
    mask : Tensor | None, optional
        Points to partition, shape ``(..., n)``; the others get cell
        ``-1``. Defaults to all.

    Returns
    -------
    tuple[Tensor, int]
        Cell of each point (``int64``, shape ``(..., n)``) and ``K``.

    Raises
    ------
    ValueError
        ``edge`` is not positive.

    Example
    -------
    >>> import torch
    >>> from tad_mctc.storch.linsolve import cell_index
    >>> positions = torch.tensor([[0.0, 0, 0], [0.5, 0, 0], [3.0, 0, 0]])
    >>> cell_index(positions, 2.0)
    (tensor([0, 0, 1]), 2)
    """
    if not edge > 0:
        raise ValueError(f"The cell edge ({edge}) must be positive.")

    positions = positions.detach()
    batch, n = positions.shape[:-2], positions.shape[-2]
    pos = positions.reshape(-1, n, 3)
    real = (
        torch.ones(pos.shape[:2], dtype=torch.bool, device=pos.device)
        if mask is None
        else mask.expand(*batch, n).reshape(-1, n)
    )

    # Integer cell coordinates from the lower corner of each system.
    lower = torch.where(real.unsqueeze(-1), pos, math.inf).amin(dim=-2)
    lower = torch.where(torch.isfinite(lower), lower, 0.0)
    coords = torch.floor((pos - lower.unsqueeze(-2)) / edge).long()
    coords = torch.where(real.unsqueeze(-1), coords, 0)
    size = int(coords.amax()) + 1 if coords.numel() > 0 else 1

    # One key per (system, cell); numbering them within each system.
    system = torch.arange(pos.shape[0], device=pos.device).unsqueeze(-1)
    key = ((system * size + coords[..., 0]) * size + coords[..., 1]) * size
    key = key + coords[..., 2]
    unique, inverse = torch.unique(key[real], return_inverse=True)
    owner = unique // size**3
    first = torch.searchsorted(
        owner, torch.arange(pos.shape[0], device=pos.device)
    )
    local = inverse - first[owner[inverse]]

    cells = torch.full(real.shape, -1, dtype=torch.long, device=pos.device)
    cells[real] = local
    counts = torch.bincount(owner, minlength=pos.shape[0])
    k = int(counts.max()) if counts.numel() > 0 else 0
    return cells.reshape(*batch, n), k
