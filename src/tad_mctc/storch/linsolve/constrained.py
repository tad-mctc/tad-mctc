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
Linear Solve: Equality Constraints
==================================

Minimizer of a quadratic form under linear equality constraints,

.. math::

    \min_x \tfrac12 x^\mathsf{T} A x - b^\mathsf{T} x
    \quad \text{subject to} \quad C x = d,

with implicit derivatives of every order. The solution and the Lagrange
multipliers :math:`\mu` solve the bordered (KKT) system

.. math::

    \begin{pmatrix} A & C^\mathsf{T} \\ C & 0 \end{pmatrix}
    \begin{pmatrix} x \\ \mu \end{pmatrix} =
    \begin{pmatrix} b \\ d \end{pmatrix},

which is symmetric but indefinite, so plain CG does not apply. It is solved
by projected CG (:func:`~tad_mctc.storch.linsolve.krylov.projected_cg`),
which needs :math:`A` to be symmetric positive definite only on the null
space of :math:`C`. The derivatives are those of the bordered system
(:func:`~tad_mctc.storch.linsolve.implicit_solve`), so they hold with
respect to :math:`A`, :math:`b`, :math:`C` and :math:`d`, and their
solves use the same method and preconditioner.

A typical use is a charge equilibration with a fixed total charge: one
constraint, ``C = 1^T`` and ``d = Q``.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import torch

from ...typing import Tensor
from .implicit import implicit_solve
from .krylov import Operator, projected_cg
from .result import ConstrainedSolveResult, require_real

__all__ = ["constrained_solve"]


def constrained_solve(
    matvec: Operator,
    rhs: Tensor,
    constraint: Tensor,
    value: Tensor,
    *params: Any,
    preconditioner: Operator | None = None,
    x0: Tensor | None = None,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    maxiter: int | None = None,
    early_exit: bool | None = None,
) -> ConstrainedSolveResult:
    r"""
    Solve :math:`\min_x \frac12 x^\mathsf{T} A x - b^\mathsf{T} x` subject
    to :math:`C x = d`, with implicit derivatives.

    Batched over the leading dimensions, which broadcast between the
    arguments. Works under ``torch.func`` transforms, like
    :func:`~tad_mctc.storch.linsolve.implicit_solve`.

    Parameters
    ----------
    matvec : Callable[..., Tensor]
        Operator ``matvec(x, *params) -> A x`` for ``x`` of shape
        ``(..., n)``. :math:`A` must be symmetric, and positive definite on
        the null space of :math:`C`.
    rhs : Tensor
        Right-hand side :math:`b`, shape ``(..., n)``.
    constraint : Tensor
        Constraint matrix :math:`C` of full rank, shape ``(..., m, n)``.
    value : Tensor
        Constraint values :math:`d`, shape ``(..., m)``.
    *params : Any
        Additional arguments of ``matvec`` and ``preconditioner``; the
        derivatives with respect to their floating point tensors are
        exact.
    preconditioner : Callable[..., Tensor] | None, optional
        Symmetric positive definite approximation of :math:`A^{-1}`,
        ``preconditioner(r, *params)`` for ``r`` of shape ``(..., n)``, e.g.
        :func:`~tad_mctc.storch.linsolve.jacobi` or
        :func:`~tad_mctc.storch.linsolve.balancing`. Defaults to none.
    x0 : Tensor | None, optional
        Initial guess for :math:`x`, moved onto the constraints. Defaults
        to zero.
    rtol : float, optional
        Relative tolerance of the bordered system. Defaults to ``1e-10``.
    atol : float, optional
        Absolute tolerance of the bordered system. Defaults to ``1e-12``.
    maxiter : int | None, optional
        Iteration limit of each solve, counting the checks of the true
        residual. Defaults to twice the size of the bordered system.
    early_exit : bool | None, optional
        Whether to stop once all systems are inactive. ``None`` (default)
        stops early unless a tensor is ``vmap``-batched or
        ``torch.compile`` is tracing.

    Returns
    -------
    ConstrainedSolveResult
        Solution :math:`x`, multipliers :math:`\mu` and the diagnostics of
        the bordered system. Convergence is not enforced: use
        :func:`~tad_mctc.storch.linsolve.require_convergence`. Derivative
        solves that do not converge give NaN.

    Raises
    ------
    ValueError
        The shapes of ``rhs``, ``constraint`` and ``value`` do not match.
    DtypeError
        A tensor is not real floating point (see
        :func:`~tad_mctc.storch.linsolve.implicit_solve`).

    Example
    -------
    >>> import torch
    >>> from tad_mctc.storch.linsolve import constrained_solve
    >>> a = torch.tensor([[2.0, 0.0], [0.0, 1.0]], dtype=torch.double)
    >>> b = torch.zeros(2, dtype=torch.double)
    >>> total = torch.tensor([1.0], dtype=torch.double, requires_grad=True)
    >>> result = constrained_solve(
    ...     lambda x, a: a @ x, b, torch.ones(1, 2, dtype=torch.double),
    ...     total, a,
    ... )
    >>> [round(v, 4) for v in result.solution.tolist()]
    [0.3333, 0.6667]
    >>> result.solution[1].backward()
    >>> round(total.grad.item(), 4)
    0.6667
    """
    require_real("constraint matrix", constraint)
    require_real("constraint value", value)
    if constraint.ndim < 2 or constraint.shape[-1] != rhs.shape[-1]:
        raise ValueError(
            f"The constraint matrix must have shape (..., m, n) with the "
            f"size n = {rhs.shape[-1]} of the right-hand side, not "
            f"{tuple(constraint.shape)}."
        )
    m, n = constraint.shape[-2], constraint.shape[-1]
    if value.ndim < 1 or value.shape[-1] != m:
        raise ValueError(
            f"The constraint values must have shape (..., {m}), not "
            f"{tuple(value.shape)}."
        )
    if x0 is not None and x0.shape[-1] != n:
        raise ValueError(
            f"The initial guess must have the size n = {n} of the "
            f"right-hand side, not {x0.shape[-1]}."
        )

    batch = torch.broadcast_shapes(
        rhs.shape[:-1], value.shape[:-1], constraint.shape[:-2]
    )
    bordered = torch.cat(
        (rhs.expand(*batch, n), value.expand(*batch, m)), dim=-1
    )
    if x0 is not None:
        x0 = torch.cat((x0.expand(*batch, n), x0.new_zeros(*batch, m)), dim=-1)

    def kkt(v: Tensor, c: Tensor, *args: Any) -> Tensor:
        x, mu = v[..., :n], v[..., n:]
        ax = matvec(x, *args) + torch.einsum("...kn,...k->...n", c, mu)
        return torch.cat((ax, torch.einsum("...kn,...n->...k", c, x)), -1)

    kkt_preconditioner = None
    if preconditioner is not None:
        prec = preconditioner

        def kkt_preconditioner(v: Tensor, c: Tensor, *args: Any) -> Tensor:
            return torch.cat((prec(v[..., :n], *args), v[..., n:]), dim=-1)

    result = implicit_solve(
        kkt,
        bordered,
        constraint,
        *params,
        method=partial(projected_cg, constraints=m),
        symmetric=True,
        preconditioner=kkt_preconditioner,
        x0=x0,
        rtol=rtol,
        atol=atol,
        maxiter=maxiter,
        early_exit=early_exit,
    )
    return ConstrainedSolveResult(
        solution=result.solution[..., :n],
        multiplier=result.solution[..., n:],
        residual=result.residual,
        target=result.target,
        iterations=result.iterations,
        status=result.status,
    )
