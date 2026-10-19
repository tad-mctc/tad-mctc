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
SafeOps: Dense Linear Solve
===========================

Dense LU or Cholesky solve whose derivatives of every order are again
solves with the same factorization.

``torch.linalg.solve`` returns wrong second and higher derivatives taken
over its forward-mode rule (``jacfwd(jacfwd(f))``, ``jvp`` of ``jvp``) and
wrong nested derivatives under ``vmap``; ``lu_solve`` is wrong under
``vmap`` as well (PyTorch 2.7 to 2.14). An explicit inverse has correct
derivatives, but its products round differently from a solve.

:func:`dense_solve` factorizes once with ``torch.linalg.lu_factor`` (or
``torch.linalg.cholesky`` for symmetric positive definite matrices). The
solution and the solutions of all derivative systems,

- reverse: ``A^T lambda = x_bar``, ``A_bar = -lambda x^T``, ``B_bar = lambda``
- forward: ``A x_dot = B_dot - A_dot x``

come from ``torch.linalg.lu_solve`` (``torch.cholesky_solve``) with these
(detached) factors, inside one ``torch.autograd.Function`` that applies
itself recursively. PyTorch never differentiates the factorization or the
triangular solves. For Cholesky, this replaces PyTorch's Cholesky backward,
which dominates the cost of derivatives of large SPD solves.
"""

from __future__ import annotations

from typing import Any, Literal

import torch

from ...autograd.nested import nested_jvp
from ...tools.compile import is_compiling
from ...typing import Tensor

__all__ = ["dense_solve"]


Method = Literal["lu", "cholesky"]
"""Factorization used by :func:`dense_solve`."""


def _front(tensor: Any, dim: int | None, size: int) -> Any:
    if tensor is None:
        return None
    if dim is None:
        return tensor.expand(size, *tensor.shape)
    return tensor.movedim(dim, 0)


def _solve(
    factor: Tensor, pivots: Tensor | None, b: Tensor, transpose: bool
) -> Tensor:
    # The factors must be contiguous: `lu_solve` is wrong for factors with a
    # non-standard stride on a size-1 batch dimension, which `vmap` produces
    # (PyTorch 2.7-2.14).
    if pivots is None:
        return torch.cholesky_solve(b, factor.contiguous())
    return torch.linalg.lu_solve(
        factor.contiguous(), pivots.contiguous(), b, adjoint=transpose
    )


def _sym(a: Tensor) -> Tensor:
    return 0.5 * (a + a.mT)


class _DenseSolveReverse(torch.autograd.Function):
    """
    ``A X = B`` (or ``A^T X = B``) from the LU or Cholesky factors of ``A``
    (``pivots`` is ``None`` for Cholesky). The matrix ``A`` is an input for
    the derivatives only.
    """

    @staticmethod
    def forward(
        a: Tensor,
        b: Tensor,
        factor: Tensor,
        pivots: Tensor | None,
        transpose: bool,
    ) -> Tensor:
        return _solve(factor, pivots, b, transpose)

    @staticmethod
    def setup_context(ctx: Any, inputs: Any, output: Tensor) -> None:
        a, _, factor, pivots, transpose = inputs
        ctx.transpose = transpose
        # The output, not a recomputed solution, so that higher derivatives
        # see its dependence on the inputs.
        ctx.save_for_backward(a, output, factor, pivots)
        ctx.save_for_forward(a, output, factor, pivots)

    @staticmethod
    def backward(ctx: Any, grad: Tensor) -> tuple[Tensor | None, ...]:
        a, x, factor, pivots = ctx.saved_tensors
        adjoint = _apply(a, grad, factor, pivots, not ctx.transpose)
        grad_a = None
        if ctx.needs_input_grad[0]:
            grad_a = -(x @ adjoint.mT if ctx.transpose else adjoint @ x.mT)
            if pivots is None:
                # symmetric matrix: only symmetric perturbations
                grad_a = _sym(grad_a)
        return grad_a, adjoint, None, None, None

    @staticmethod
    def vmap(
        info: Any,
        in_dims: tuple[int | None, ...],
        a: Tensor,
        b: Tensor,
        factor: Tensor,
        pivots: Tensor | None,
        transpose: bool,
    ) -> tuple[Tensor, int]:
        size = info.batch_size
        a, b, factor = (
            _front(t, d, size) for t, d in zip((a, b, factor), in_dims)
        )
        pivots = _front(pivots, in_dims[3], size)
        return _apply(a, b, factor, pivots, transpose), 0


class _DenseSolve(_DenseSolveReverse):
    """Adds the forward-mode rule (not supported by Dynamo)."""

    @staticmethod
    @nested_jvp
    def jvp(
        ctx: Any,
        a_dot: Tensor | None,
        b_dot: Tensor | None,
        factor_dot: None,
        pivots_dot: None,
        transpose_dot: None,
    ) -> Tensor:
        a, x, factor, pivots = ctx.saved_tensors
        rhs = torch.zeros_like(x) if b_dot is None else b_dot.expand_as(x)
        if a_dot is not None:
            if pivots is None:
                a_dot = _sym(a_dot)
            rhs = rhs - (a_dot.mT if ctx.transpose else a_dot) @ x
        return _apply(a, rhs, factor, pivots, ctx.transpose)


def _apply(
    a: Tensor,
    b: Tensor,
    factor: Tensor,
    pivots: Tensor | None,
    transpose: bool,
) -> Tensor:
    if not is_compiling():
        return _DenseSolve.apply(a, b, factor, pivots, transpose)

    # Dynamo drops derivatives of graph inputs that reach a Function
    # unused by its `forward`; a view avoids it (see `implicit_solve`).
    return _DenseSolveReverse.apply(a.view_as(a), b, factor, pivots, transpose)


def dense_solve(a: Tensor, b: Tensor, method: Method = "lu") -> Tensor:
    """
    Solve ``A X = B`` by LU or Cholesky with exact derivatives of every order.

    Drop-in replacement for ``torch.linalg.solve`` (same shape rules) for
    code that is differentiated more than once in forward mode or under
    ``vmap``. The matrix is factorized once; all derivatives reuse the
    factors. Like ``torch.linalg.solve`` and ``torch.linalg.cholesky``, it
    raises for exactly singular (LU) or not positive definite (Cholesky)
    matrices in eager mode.

    Under ``torch.compile``, only first derivatives in reverse mode are
    available (see :func:`~tad_mctc.storch.linsolve.implicit_solve`).

    Parameters
    ----------
    a : Tensor
        Square matrices of shape ``(..., n, n)``.
    b : Tensor
        Right-hand sides, ``(..., n, k)``, or vectors ``(..., n)`` if
        ``b.shape == a.shape[:-1]`` (as in ``torch.linalg.solve``).
    method : {"lu", "cholesky"}, optional
        Factorization. ``"cholesky"`` requires symmetric positive definite
        matrices, reads only their lower triangle and is about twice as fast.
        Its derivatives treat ``A`` as symmetric, i.e., only the symmetric
        part of a perturbation of ``A`` contributes, as for
        ``torch.linalg.cholesky``. Defaults to ``"lu"``.

    Returns
    -------
    Tensor
        Solution with the shape of ``b`` (broadcast with ``a``).

    Raises
    ------
    ValueError
        Unknown method.

    Example
    -------
    >>> import torch
    >>> from torch.func import jacfwd
    >>> from tad_mctc.storch.linsolve import dense_solve
    >>> a = torch.tensor([[4.0, 1.0], [1.0, 3.0]], dtype=torch.double)
    >>> b = torch.tensor([1.0, 2.0], dtype=torch.double)
    >>> f = lambda t: dense_solve(a + t * torch.eye(2, dtype=a.dtype), b).sum()
    >>> t = torch.tensor(0.0, dtype=torch.double)
    >>> round(jacfwd(jacfwd(f))(t).item(), 4)  # 2 sum(A^-3 b)
    0.1097
    """
    if method not in ("lu", "cholesky"):
        raise ValueError(
            f"Unknown method '{method}'. Choose from 'lu' or 'cholesky'."
        )

    vector = b.ndim == a.ndim - 1 and b.shape == a.shape[:-1]
    if vector:
        b = b.unsqueeze(-1)

    # Equal batch shapes: under `vmap`, `torch.linalg.solve` mixes the
    # `vmap` dimension into a broadcast batch dimension (PyTorch 2.14).
    batch = torch.broadcast_shapes(a.shape[:-2], b.shape[:-2])
    a = a.expand(*batch, *a.shape[-2:])
    b = b.expand(*batch, *b.shape[-2:])

    pivots: Tensor | None = None
    with torch.no_grad():
        if method == "lu":
            factor, pivots = torch.linalg.lu_factor(a.detach())
        else:
            factor = torch.linalg.cholesky(a.detach())

    x = _apply(a, b, factor, pivots, False)
    return x.squeeze(-1) if vector else x
