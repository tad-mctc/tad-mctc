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
Linear Solve: Krylov Methods
============================

Batched, matrix-free Krylov solvers for ``A x = b``.

The operator is a function ``matvec(x, *params)`` returning ``A x`` for a
tensor ``x`` of the shape of the right-hand side. The trailing ``vdims``
dimensions of ``x`` form the vector; all leading dimensions are independent
systems (a batch), each with its own convergence state. A dense, batched
matrix ``a`` is used as ``matvec = lambda x, a: (a @ x.unsqueeze(-1)).squeeze(-1)``.

Every system iterates under a tensor mask. Converged, failed or padding
systems are frozen, so the solvers are usable under ``torch.func.vmap`` and
``torch.compile(fullgraph=True)``, where the loop runs for the full
``maxiter`` trips, each with a product of the operator for all systems.
Outside these transforms, the loop exits as soon as all systems are
inactive. The default ``maxiter`` (twice the vector size) is a safety limit
for eager mode; under ``vmap`` or ``torch.compile``, and for large systems,
set it to the iteration count the (preconditioned) problem actually needs,
plus one per check of the true residual (see below).

A system converges when its true residual ``b - A x`` satisfies the
tolerance. When the recursive residual of CG or BiCGSTAB does, the next
operator application checks the true one, which drifts from it by roundoff
(mostly in single precision); a failed check restarts the recurrence from
the true residual. If that does not decrease from one check (or, for
GMRES, one restart cycle) to the next, the tolerance is below the
attainable accuracy, about :math:`\epsilon \|A\| \|x\|`, and the system
stops as :data:`~tad_mctc.storch.linsolve.STATUS_STAGNATED`. A true residual
that keeps decreasing slowly is not stagnation and runs until ``maxiter``.

Only real floating point tensors are supported.

The solvers are ordinary PyTorch code: differentiating through them
differentiates the iterations ("unrolling"). For derivatives of the
mathematical solution, which do not depend on the iterations, use
:func:`~tad_mctc.storch.linsolve.implicit_solve`.

The conjugate gradient iteration follows
`TorchOpt <https://github.com/metaopt/torchopt>`__ (``torchopt.linalg.cg``,
Apache-2.0) and the tolerance handling and BiCGSTAB recurrence follow
`dxtb <https://github.com/grimme-lab/dxtb>`__ (Apache-2.0), with breakdown
detection and batching added. GMRES is the restarted method of Saad and
Schultz (SIAM J. Sci. Stat. Comput. 7, 856 (1986)) with Givens rotations,
as in ``jax.scipy.sparse.linalg.gmres`` (``solve_method="incremental"``).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import torch

from ...autograd.checks import is_vmapped
from ...typing import Tensor
from .result import (
    LinearSolveResult,
    convergence_target,
    finalize,
    require_like,
    require_real,
    require_real_params,
    safe_divide,
    vdot,
    vnorm,
)

__all__ = ["bicgstab", "cg", "gmres"]


Operator = Callable[..., Tensor]
"""Linear operator ``matvec(x, *params) -> A x``."""


def _check_arguments(
    rhs: Tensor,
    params: tuple[Any, ...],
    x0: Tensor | None,
    maxiter: int | None,
    vdims: int,
) -> int:
    """Validate the arguments, and return the number of iterations."""
    require_real("right-hand side", rhs)
    require_real("initial guess", x0)
    require_real_params(params)
    if vdims < 1 or vdims > rhs.ndim:
        raise ValueError(
            f"The number of vector dimensions ({vdims}) must be between 1 "
            f"and the number of dimensions of the right-hand side "
            f"({rhs.ndim})."
        )

    if maxiter is None:
        maxiter = 2 * math.prod(rhs.shape[-vdims:])
    if maxiter < 0:
        raise ValueError(f"The iteration limit ({maxiter}) is negative.")

    if x0 is not None and x0.shape != rhs.shape:
        raise ValueError(
            f"Shape of the initial guess {tuple(x0.shape)} does not match "
            f"the right-hand side {tuple(rhs.shape)}."
        )
    return maxiter


def _check_image(image: Tensor, rhs: Tensor) -> None:
    """Validate the first product of the operator."""
    require_like("operator output", image, "right-hand side", rhs)
    if image.shape != rhs.shape:
        raise ValueError(
            f"The operator maps the right-hand side shape {tuple(rhs.shape)} "
            f"to {tuple(image.shape)}; it must preserve the shape."
        )


def _setup(
    matvec: Operator,
    rhs: Tensor,
    params: tuple[Any, ...],
    x0: Tensor | None,
    maxiter: int | None,
    vdims: int,
) -> tuple[Tensor, Tensor, int]:
    """
    Validate the arguments, and return the initial guess, its residual and
    the number of iterations.
    """
    maxiter = _check_arguments(rhs, params, x0, maxiter, vdims)
    if x0 is None:
        x0 = torch.zeros_like(rhs)

    image = matvec(x0, *params)
    _check_image(image, rhs)
    return x0, rhs - image, maxiter


def _exit_early(active: Tensor, early_exit: bool | None) -> bool:
    """
    Whether the loop may stop once all systems are inactive.

    Stopping requires inspecting a tensor in Python, which ``vmap`` forbids
    and which breaks the graph under ``torch.compile``.
    """
    if early_exit is None:
        return not is_vmapped(active)
    return early_exit


def _check(
    true_residual: Tensor,
    checking: Tensor,
    target: Tensor,
    norm: Tensor,
    checked: Tensor,
    stagnation: Tensor,
    vdims: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Check the true residual of the systems whose recursive residual met the
    target.

    The recursive residual of CG and BiCGSTAB drifts from the true one by
    roundoff, most in single precision. A system that fails the check
    restarts from its true residual, unless that did not decrease since its
    last check: the target is then below the attainable accuracy, and the
    system stops (:data:`~tad_mctc.storch.linsolve.STATUS_STAGNATED`)
    instead of iterating until ``maxiter``.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor, Tensor]
        Systems to restart, and the updated true residual norms, norms at
        the last check and stagnation flags.
    """
    true_norm = vnorm(true_residual, vdims)
    failed = checking & (true_norm > target)
    stagnated = failed & ~(true_norm < checked)
    norm = torch.where(checking, true_norm, norm)
    checked = torch.where(checking, true_norm, checked)
    return failed & ~stagnated, norm, checked, stagnation | stagnated


def _true_residual(
    matvec: Operator,
    rhs: Tensor,
    x: Tensor,
    params: tuple[Any, ...],
    norm: Tensor,
    stopped: bool,
    breakdown: Tensor,
    vdims: int,
) -> Tensor:
    """
    Norm of the true residual of the solution.

    If the loop stopped early without a breakdown, the solution of every
    system was checked (or is the initial guess), and ``norm`` holds its
    true residual; otherwise, the residual is recomputed.
    """
    if stopped and not bool(breakdown.any()):
        return norm
    return vnorm(rhs - matvec(x, *params), vdims)


def cg(
    matvec: Operator,
    rhs: Tensor,
    *params: Any,
    x0: Tensor | None = None,
    preconditioner: Operator | None = None,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    maxiter: int | None = None,
    vdims: int = 1,
    early_exit: bool | None = None,
) -> LinearSolveResult:
    r"""
    Preconditioned conjugate gradient method.

    The operator must be symmetric positive definite. A direction of
    non-positive curvature stops the system and is reported as
    :data:`~tad_mctc.storch.linsolve.STATUS_BREAKDOWN`, unless the true
    residual already satisfies the tolerance. Convergence is decided by the
    true residual (see the module documentation).

    Parameters
    ----------
    matvec : Callable[..., Tensor]
        Linear operator, ``matvec(x, *params) -> A x``.
    rhs : Tensor
        Right-hand side ``b``. The trailing ``vdims`` dimensions form the
        vector, leading dimensions are batch dimensions.
    *params : Any
        Additional arguments of ``matvec`` and ``preconditioner``.
    x0 : Tensor | None, optional
        Initial guess. Defaults to zero.
    preconditioner : Callable[..., Tensor] | None, optional
        Symmetric positive definite approximation of :math:`A^{-1}`,
        ``preconditioner(r, *params) -> M r``, e.g. ``r / diag(A)`` for the
        Jacobi preconditioner. Defaults to none.
    rtol : float, optional
        Relative tolerance. Defaults to ``1e-10``.
    atol : float, optional
        Absolute tolerance. Defaults to ``1e-12``. A system converged if
        :math:`\|b - Ax\| \leq \mathrm{atol} + \max(\mathrm{rtol},
        10\epsilon) \|b\|`.
    maxiter : int | None, optional
        Iteration limit, counting the checks of the true residual. Defaults
        to twice the vector size.
    vdims : int, optional
        Number of trailing dimensions forming the vector. Defaults to ``1``.
    early_exit : bool | None, optional
        Whether to stop once all systems are inactive, which needs a
        host-device synchronization per iteration. ``None`` (default) stops
        early unless a tensor is ``vmap``-batched or ``torch.compile`` is
        tracing.

    Returns
    -------
    LinearSolveResult
        Solution and diagnostics.

    Example
    -------
    >>> import torch
    >>> from tad_mctc.storch.linsolve import cg
    >>> a = torch.tensor([[4.0, 1.0], [1.0, 3.0]], dtype=torch.double)
    >>> b = torch.tensor([1.0, 2.0], dtype=torch.double)
    >>> result = cg(lambda x, a: a @ x, b, a)
    >>> [round(v, 4) for v in result.solution.tolist()]
    [0.0909, 0.6364]
    >>> result.converged.item(), result.iterations.item()
    (True, 2)
    """
    x, r, maxiter = _setup(matvec, rhs, params, x0, maxiter, vdims)

    def precondition(v: Tensor) -> Tensor:
        return v if preconditioner is None else preconditioner(v, *params)

    target = convergence_target(rhs, rtol, atol, vdims)
    z = precondition(r)
    p = z
    rz = vdot(r, z, vdims)

    norm = vnorm(r, vdims)
    active = norm > target
    checking = torch.zeros_like(active)
    checked = torch.full_like(norm, math.inf)
    breakdown = torch.zeros_like(active)
    stagnation = torch.zeros_like(active)
    iterations = torch.zeros_like(active, dtype=torch.int64)

    exit_early = _exit_early(active, early_exit)
    stopped = False
    for _ in range(maxiter):
        if exit_early and not bool((active | checking).any()):
            stopped = True
            break

        # One product per trip: of the search direction, or of the solution
        # of a system whose recursive residual met the target.
        ap = matvec(torch.where(checking, x, p), *params)
        pap = vdot(p, ap, vdims)

        # Non-positive curvature: the operator (or the preconditioner) is
        # not positive definite, and the step would be meaningless.
        bad = ~((pap > 0) & (rz > 0))
        breakdown = breakdown | (active & bad)
        update = active & ~bad

        alpha = safe_divide(rz, pap)
        x = torch.where(update, x + alpha * p, x)
        r = torch.where(update, r - alpha * ap, r)

        restart, norm, checked, stagnation = _check(
            rhs - ap, checking, target, norm, checked, stagnation, vdims
        )
        r = torch.where(restart, rhs - ap, r)

        z = precondition(r)
        rz_new = vdot(r, z, vdims)
        beta = safe_divide(rz_new, rz)
        p = torch.where(update, z + beta * p, torch.where(restart, z, p))
        rz = torch.where(update | restart, rz_new, rz)

        iterations = iterations + update.to(torch.int64)
        met = vnorm(r, vdims) <= target
        checking = update & met
        active = (update & ~met) | restart

    residual = _true_residual(
        matvec, rhs, x, params, norm, stopped, breakdown, vdims
    )
    return finalize(
        x, residual, target, iterations, breakdown, vdims, stagnation
    )


def bicgstab(
    matvec: Operator,
    rhs: Tensor,
    *params: Any,
    x0: Tensor | None = None,
    preconditioner: Operator | None = None,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    maxiter: int | None = None,
    vdims: int = 1,
    early_exit: bool | None = None,
) -> LinearSolveResult:
    r"""
    Right-preconditioned stabilized bi-conjugate gradient method (BiCGSTAB).

    The operator may be nonsymmetric. Each iteration applies the operator
    twice. A vanishing denominator stops the system and is reported as
    :data:`~tad_mctc.storch.linsolve.STATUS_BREAKDOWN`, unless the true
    residual already satisfies the tolerance. Convergence is decided by the
    true residual (see the module documentation).

    Parameters
    ----------
    matvec : Callable[..., Tensor]
        Linear operator, ``matvec(x, *params) -> A x``.
    rhs : Tensor
        Right-hand side ``b``. The trailing ``vdims`` dimensions form the
        vector, leading dimensions are batch dimensions.
    *params : Any
        Additional arguments of ``matvec`` and ``preconditioner``.
    x0 : Tensor | None, optional
        Initial guess. Defaults to zero.
    preconditioner : Callable[..., Tensor] | None, optional
        Approximation of :math:`A^{-1}`, ``preconditioner(r, *params)``.
        Defaults to none.
    rtol : float, optional
        Relative tolerance. Defaults to ``1e-10``.
    atol : float, optional
        Absolute tolerance. Defaults to ``1e-12``. A system converged if
        :math:`\|b - Ax\| \leq \mathrm{atol} + \max(\mathrm{rtol},
        10\epsilon) \|b\|`.
    maxiter : int | None, optional
        Iteration limit, counting the checks of the true residual. Defaults
        to twice the vector size.
    vdims : int, optional
        Number of trailing dimensions forming the vector. Defaults to ``1``.
    early_exit : bool | None, optional
        Whether to stop once all systems are inactive. ``None`` (default)
        stops early unless a tensor is ``vmap``-batched or
        ``torch.compile`` is tracing.

    Returns
    -------
    LinearSolveResult
        Solution and diagnostics.

    Example
    -------
    >>> import torch
    >>> from tad_mctc.storch.linsolve import bicgstab
    >>> a = torch.tensor([[4.0, 2.0], [-1.0, 3.0]], dtype=torch.double)
    >>> b = torch.tensor([1.0, 2.0], dtype=torch.double)
    >>> result = bicgstab(lambda x, a: a @ x, b, a)
    >>> [round(v, 4) for v in result.solution.tolist()]
    [-0.0714, 0.6429]
    >>> result.converged.item()
    True
    """
    x, residual, maxiter = _setup(matvec, rhs, params, x0, maxiter, vdims)

    def precondition(v: Tensor) -> Tensor:
        return v if preconditioner is None else preconditioner(v, *params)

    target = convergence_target(rhs, rtol, atol, vdims)
    shadow = residual
    direction = torch.zeros_like(rhs)
    image = torch.zeros_like(rhs)

    scalar = torch.ones_like(target)
    rho_previous = scalar
    alpha_previous = scalar
    omega_previous = scalar
    tiny = torch.finfo(rhs.dtype).tiny

    norm = vnorm(residual, vdims)
    active = norm > target
    checking = torch.zeros_like(active)
    checked = torch.full_like(norm, math.inf)
    breakdown = torch.zeros_like(active)
    stagnation = torch.zeros_like(active)
    iterations = torch.zeros_like(active, dtype=torch.int64)

    exit_early = _exit_early(active, early_exit)
    stopped = False
    for _ in range(maxiter):
        if exit_early and not bool((active | checking).any()):
            stopped = True
            break

        rho = vdot(shadow, residual, vdims)
        beta = safe_divide(rho * alpha_previous, rho_previous * omega_previous)
        trial_direction = residual + beta * (direction - omega_previous * image)
        direction = torch.where(active, trial_direction, direction)

        # The first product is of the solution for a system whose recursive
        # residual met the target.
        phat = precondition(direction)
        image_new = matvec(torch.where(checking, x, phat), *params)
        sv = vdot(shadow, image_new, vdims)
        alpha = safe_divide(rho, sv)

        intermediate = residual - alpha * image_new
        shat = precondition(intermediate)
        correction = matvec(shat, *params)
        omega = safe_divide(
            vdot(correction, intermediate, vdims),
            vdot(correction, correction, vdims),
        )

        trial_x = x + alpha * phat + omega * shat
        trial_residual = intermediate - omega * correction
        trial_active = vnorm(trial_residual, vdims) > target

        # A vanishing `rho` or `<shadow, A p>` makes the recurrence
        # undefined; a vanishing `omega` stagnates it (unless the
        # intermediate residual already converged).
        bad = (rho.abs() <= tiny) | (sv.abs() <= tiny)
        bad = bad | ((omega.abs() <= tiny) & trial_active)
        bad = bad | ~torch.isfinite(alpha * omega)
        breakdown = breakdown | (active & bad)
        update = active & ~bad

        x = torch.where(update, trial_x, x)
        residual = torch.where(update, trial_residual, residual)
        image = torch.where(update, image_new, image)
        rho_previous = torch.where(update, rho, rho_previous)
        alpha_previous = torch.where(update, alpha, alpha_previous)
        omega_previous = torch.where(update, omega, omega_previous)

        # A restart begins a new recurrence from the true residual.
        restart, norm, checked, stagnation = _check(
            rhs - image_new,
            checking,
            target,
            norm,
            checked,
            stagnation,
            vdims,
        )
        residual = torch.where(restart, rhs - image_new, residual)
        shadow = torch.where(restart, residual, shadow)
        direction = torch.where(restart, 0.0, direction)
        image = torch.where(restart, 0.0, image)
        rho_previous = torch.where(restart, 1.0, rho_previous)
        alpha_previous = torch.where(restart, 1.0, alpha_previous)
        omega_previous = torch.where(restart, 1.0, omega_previous)

        iterations = iterations + update.to(torch.int64)
        checking = update & ~trial_active
        active = (update & trial_active) | restart

    true_residual = _true_residual(
        matvec, rhs, x, params, norm, stopped, breakdown, vdims
    )
    return finalize(
        x, true_residual, target, iterations, breakdown, vdims, stagnation
    )


def gmres(
    matvec: Operator,
    rhs: Tensor,
    *params: Any,
    x0: Tensor | None = None,
    preconditioner: Operator | None = None,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    maxiter: int | None = None,
    vdims: int = 1,
    early_exit: bool | None = None,
    restart: int = 20,
) -> LinearSolveResult:
    r"""
    Restarted, right-preconditioned generalized minimal residual method.

    The operator may be nonsymmetric or indefinite. Each iteration applies
    the operator once and orthogonalizes against the current Krylov basis
    (modified Gram-Schmidt with one reorthogonalization), so the cost and
    memory grow with ``restart``. The residual norm never increases. A
    system whose true residual does not decrease over a full restart cycle
    stops and is reported as :data:`~tad_mctc.storch.linsolve.STATUS_STAGNATED`
    (e.g. a singular operator, or a tolerance below the attainable
    accuracy), unless it already satisfies the tolerance.

    Parameters
    ----------
    matvec : Callable[..., Tensor]
        Linear operator, ``matvec(x, *params) -> A x``.
    rhs : Tensor
        Right-hand side ``b``. The trailing ``vdims`` dimensions form the
        vector, leading dimensions are batch dimensions.
    *params : Any
        Additional arguments of ``matvec`` and ``preconditioner``.
    x0 : Tensor | None, optional
        Initial guess. Defaults to zero.
    preconditioner : Callable[..., Tensor] | None, optional
        Approximation of :math:`A^{-1}`, ``preconditioner(r, *params)``.
        Defaults to none.
    rtol : float, optional
        Relative tolerance. Defaults to ``1e-10``.
    atol : float, optional
        Absolute tolerance. Defaults to ``1e-12``. A system converged if
        :math:`\|b - Ax\| \leq \mathrm{atol} + \max(\mathrm{rtol},
        10\epsilon) \|b\|`.
    maxiter : int | None, optional
        Limit of the total number of iterations (operator applications in
        the Arnoldi process). Defaults to twice the vector size.
    vdims : int, optional
        Number of trailing dimensions forming the vector. Defaults to ``1``.
    early_exit : bool | None, optional
        Whether to stop once all systems are inactive. ``None`` (default)
        stops early unless a tensor is ``vmap``-batched or
        ``torch.compile`` is tracing.
    restart : int, optional
        Dimension of the Krylov subspace before a restart. Defaults to
        ``20``, capped at the vector size.

    Returns
    -------
    LinearSolveResult
        Solution and diagnostics.

    Example
    -------
    >>> import torch
    >>> from tad_mctc.storch.linsolve import gmres
    >>> a = torch.tensor([[1.0, 2.0], [2.0, -1.0]], dtype=torch.double)
    >>> b = torch.tensor([3.0, 1.0], dtype=torch.double)
    >>> result = gmres(lambda x, a: a @ x, b, a)  # symmetric indefinite
    >>> [round(v, 8) for v in result.solution.tolist()]
    [1.0, 1.0]
    >>> result.converged.item(), result.iterations.item()
    (True, 2)
    """
    x, residual, maxiter = _setup(matvec, rhs, params, x0, maxiter, vdims)
    if restart < 1:
        raise ValueError(f"The restart length ({restart}) must be positive.")
    size = math.prod(rhs.shape[-vdims:])
    restart = min(restart, size, max(maxiter, 1))

    def precondition(v: Tensor) -> Tensor:
        return v if preconditioner is None else preconditioner(v, *params)

    target = convergence_target(rhs, rtol, atol, vdims)
    norm = vnorm(residual, vdims)
    tiny = torch.finfo(rhs.dtype).tiny

    active = norm > target
    breakdown = torch.zeros_like(active)
    stagnation = torch.zeros_like(active)
    iterations = torch.zeros_like(active, dtype=torch.int64)

    exit_early = _exit_early(active, early_exit)
    remaining = maxiter
    while remaining > 0:
        if exit_early and not bool(active.any()):
            break
        steps = min(restart, remaining)
        remaining -= steps

        # Arnoldi process on A M with Givens rotations; `g` holds the
        # rotated right-hand side, whose last entry is the residual norm.
        basis = [safe_divide(residual, norm)]
        columns: list[list[Tensor]] = []
        cosines: list[Tensor] = []
        sines: list[Tensor] = []
        g = [norm]
        count = torch.zeros_like(iterations)
        stepping = active

        for j in range(steps):
            if exit_early and not bool(stepping.any()):
                break

            w = matvec(precondition(basis[j]), *params)
            h = [torch.zeros_like(norm) for _ in range(j + 2)]
            for _ in range(2):
                for i in range(j + 1):
                    projection = vdot(w, basis[i], vdims)
                    h[i] = h[i] + projection
                    w = w - projection * basis[i]
            h[j + 1] = vnorm(w, vdims)
            basis.append(safe_divide(w, h[j + 1]))

            for i in range(j):
                h[i], h[i + 1] = (
                    cosines[i] * h[i] + sines[i] * h[i + 1],
                    -sines[i] * h[i] + cosines[i] * h[i + 1],
                )
            radius = torch.sqrt(h[j] ** 2 + h[j + 1] ** 2)
            cosine = torch.where(radius > tiny, safe_divide(h[j], radius), 1.0)
            sine = safe_divide(h[j + 1], radius)
            cosines.append(cosine)
            sines.append(sine)
            h[j] = radius
            g.append(-sine * g[j])
            g[j] = cosine * g[j]
            columns.append(h[: j + 1])

            count = count + stepping.to(torch.int64)
            # The rotated residual norm; a vanishing new basis vector ("lucky
            # breakdown") makes it zero, since the Krylov space then
            # contains the solution.
            stepping = stepping & (g[j + 1].abs() > target)

        # Least-squares solution of the triangular system for the first
        # `count` columns of each system; the others are set to zero.
        k = len(columns)
        zero = torch.zeros_like(norm)
        r = torch.stack(
            [
                torch.stack(column + [zero] * (k - len(column)), dim=-1)
                for column in columns
            ],
            dim=-1,
        )
        index = torch.arange(k, device=rhs.device)
        used = index < count.unsqueeze(-1)
        diagonal = torch.where(used, r.diagonal(dim1=-2, dim2=-1), 1.0)
        diagonal = torch.where(diagonal.abs() > tiny, diagonal, 1.0)
        r = r * used.unsqueeze(-2) * used.unsqueeze(-1)
        r = r + torch.diag_embed(diagonal - r.diagonal(dim1=-2, dim2=-1))
        rhs_small = torch.where(used, torch.stack(g[:k], dim=-1), 0.0)
        y = torch.linalg.solve_triangular(
            r, rhs_small.unsqueeze(-1), upper=True
        ).squeeze(-1)

        correction = torch.zeros_like(rhs)
        for j in range(k):
            correction = correction + y[..., j] * basis[j]
        x = torch.where(active, x + precondition(correction), x)

        # restart from the true residual
        residual = rhs - matvec(x, *params)
        new_norm = vnorm(residual, vdims)
        stagnated = active & ~(new_norm < norm) & (new_norm > target)
        stagnation = stagnation | stagnated
        iterations = iterations + torch.where(active, count, 0)
        active = active & ~stagnated & (new_norm > target)
        norm = new_norm

    true_residual = vnorm(rhs - matvec(x, *params), vdims)
    return finalize(
        x, true_residual, target, iterations, breakdown, vdims, stagnation
    )


def projected_cg(
    matvec: Operator,
    rhs: Tensor,
    *params: Any,
    constraints: int,
    constraint: Operator | None = None,
    x0: Tensor | None = None,
    preconditioner: Operator | None = None,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    maxiter: int | None = None,
    vdims: int = 1,
    early_exit: bool | None = None,
) -> LinearSolveResult:
    r"""
    Projected preconditioned conjugate gradient method for a system with
    linear equality constraints,

    .. math::

        \begin{pmatrix} A & C^\mathsf{T} \\ C & 0 \end{pmatrix}
        \begin{pmatrix} x \\ \mu \end{pmatrix} =
        \begin{pmatrix} b \\ d \end{pmatrix},

    i.e. the minimizer of :math:`\frac12 x^\mathsf{T} A x - b^\mathsf{T} x`
    subject to :math:`C x = d` and its Lagrange multipliers :math:`\mu`
    (Gould, Hribar and Nocedal, SIAM J. Sci. Comput. 23, 1376 (2001), with
    their residual update). :math:`A` must be symmetric and positive
    definite on the null space of the ``m x n`` matrix :math:`C`, which
    must have full rank; :math:`A` itself may be indefinite. The iterates
    stay feasible, and each iteration applies the operator once.

    The operator is given in bordered form, as for
    :func:`~tad_mctc.storch.linsolve.implicit_solve` (whose ``method`` this
    can be, see :func:`~tad_mctc.storch.linsolve.constrained_solve`):
    ``matvec(v, *params)`` maps the bordered vector ``v = (x, mu)`` of size
    ``n + m`` to ``(A x + C^T mu, C x)``. The rows of :math:`C` are
    recovered from ``m`` products at the start, so the setup costs
    ``m + 1`` products, or one with ``constraint``. Convergence is decided by the true residual of the
    bordered system (see the module documentation).

    Parameters
    ----------
    matvec : Callable[..., Tensor]
        Bordered operator, see above.
    rhs : Tensor
        Bordered right-hand side ``(b, d)``, shape ``(..., n + m)``.
    *params : Any
        Additional arguments of ``matvec`` and ``preconditioner``.
    constraints : int
        Number of constraints ``m``.
    constraint : Callable[..., Tensor] | None, optional
        The constraint part of the bordered operator, ``constraint(v,
        *params) -> (C^T mu, C x)``, from which the rows of :math:`C` are
        recovered without products with :math:`A`. Defaults to
        ``matvec``.
    x0 : Tensor | None, optional
        Bordered initial guess. Its ``x`` block is moved onto the
        constraints (minimum-norm correction). Defaults to zero.
    preconditioner : Callable[..., Tensor] | None, optional
        Bordered preconditioner; only the ``x`` block of its output for
        ``(r, 0)`` is used, which must be a symmetric positive definite
        approximation of :math:`A^{-1} r`. Defaults to none.
    rtol : float, optional
        Relative tolerance. Defaults to ``1e-10``.
    atol : float, optional
        Absolute tolerance. Defaults to ``1e-12``.
    maxiter : int | None, optional
        Iteration limit, counting the checks of the true residual. Defaults
        to twice the vector size.
    vdims : int, optional
        Number of trailing dimensions forming the vector; must be ``1``.
    early_exit : bool | None, optional
        Whether to stop once all systems are inactive. ``None`` (default)
        stops early unless a tensor is ``vmap``-batched or
        ``torch.compile`` is tracing.

    Returns
    -------
    LinearSolveResult
        Bordered solution ``(x, mu)`` and diagnostics. A rank-deficient
        constraint matrix or a preconditioner that is not positive definite
        on it is reported as :data:`~tad_mctc.storch.linsolve.STATUS_BREAKDOWN`.

    Raises
    ------
    ValueError
        ``vdims`` is not ``1``, or ``constraints`` is not between ``1`` and
        the vector size minus one.

    Example
    -------
    >>> import torch
    >>> from tad_mctc.storch.linsolve.krylov import projected_cg
    >>> a = torch.tensor([[2.0, 0.0], [0.0, 1.0]], dtype=torch.double)
    >>> def bordered(v, a):  # constraint x_1 + x_2 = 1
    ...     x, mu = v[:2], v[2:]
    ...     return torch.cat((a @ x + mu, x.sum(-1, keepdim=True)))
    >>> rhs = torch.tensor([0.0, 0.0, 1.0], dtype=torch.double)
    >>> result = projected_cg(bordered, rhs, a, constraints=1)
    >>> [round(v, 4) for v in result.solution.tolist()]
    [0.3333, 0.6667, -0.6667]
    """
    maxiter = _check_arguments(rhs, params, x0, maxiter, vdims)
    if vdims != 1:
        raise ValueError(
            f"The projected CG method needs `vdims=1`, not {vdims}."
        )
    m = constraints
    n = rhs.shape[-1] - m
    if m < 1 or n < 1:
        raise ValueError(
            f"The number of constraints ({m}) must be between 1 and the "
            f"vector size minus one ({rhs.shape[-1] - 1})."
        )

    def join(x: Tensor, mu: Tensor) -> Tensor:
        return torch.cat((x, mu), dim=-1)

    b, d = rhs[..., :n], rhs[..., n:]
    zeros_m = torch.zeros_like(d)

    # The rows of C, from the products of the unit vectors of the
    # multipliers: (A 0 + C^T e_k, C 0) = (C_k, 0).
    unit = torch.eye(m, dtype=rhs.dtype, device=rhs.device)
    rows = []
    for k in range(m):
        e = join(torch.zeros_like(b), unit[k].expand_as(d))
        image = (matvec if constraint is None else constraint)(e, *params)
        if k == 0:
            _check_image(image, rhs)
        rows.append(image[..., :n])
    c = torch.stack(rows, dim=-2)  # (..., m, n)

    def times_c(x: Tensor) -> Tensor:  # C x
        return torch.einsum("...kn,...n->...k", c, x)

    def times_ct(mu: Tensor) -> Tensor:  # C^T mu
        return torch.einsum("...kn,...k->...n", c, mu)

    def precondition(r: Tensor) -> Tensor:
        if preconditioner is None:
            return r
        return preconditioner(join(r, zeros_m), *params)[..., :n]

    # Cholesky factors of C C^T (feasibility) and C M C^T (projection); a
    # failed factorization is a breakdown of the system.
    mc = torch.stack(
        [precondition(c[..., k, :]) for k in range(m)], dim=-2
    )  # rows of M C^T
    gram, info_gram = torch.linalg.cholesky_ex(c @ c.mT)
    schur, info_schur = torch.linalg.cholesky_ex(c @ mc.mT)
    singular = ((info_gram != 0) | (info_schur != 0)).unsqueeze(-1)
    eye = torch.eye(m, dtype=rhs.dtype, device=rhs.device)
    gram = torch.where(singular.unsqueeze(-1), eye, gram)
    schur = torch.where(singular.unsqueeze(-1), eye, schur)

    def solve(factor: Tensor, v: Tensor) -> Tensor:
        return torch.cholesky_solve(v.unsqueeze(-1), factor).squeeze(-1)

    def multiplier_step(r: Tensor) -> Tensor:
        # Least-squares multipliers of r in the M inner product; removing
        # C^T of them leaves the projected gradient unchanged but keeps the
        # residual small (residual update).
        return solve(schur, times_c(precondition(r)))

    def project(r: Tensor) -> Tensor:
        z = precondition(r)
        return z - torch.einsum(
            "...kn,...k->...n", mc, solve(schur, times_c(z))
        )

    # Feasible start: the minimum-norm correction onto C x = d.
    x = torch.zeros_like(b) if x0 is None else x0[..., :n]
    mu = zeros_m if x0 is None else x0[..., n:]
    x = x + times_ct(solve(gram, d - times_c(x)))

    image = matvec(join(x, mu), *params)
    r = image[..., :n] - b  # A x + C^T mu - b
    lam = multiplier_step(r)
    r, mu = r - times_ct(lam), mu - lam

    target = convergence_target(rhs, rtol, atol, vdims)
    g = project(r)
    p = -g
    rg = vdot(r, g, vdims)

    norm = vnorm(join(r, times_c(x) - d), vdims)
    breakdown = singular & (norm > target)
    active = (norm > target) & ~breakdown
    checking = torch.zeros_like(active)
    checked = torch.full_like(norm, math.inf)
    stagnation = torch.zeros_like(active)
    iterations = torch.zeros_like(active, dtype=torch.int64)

    exit_early = _exit_early(active, early_exit)
    stopped = False
    for _ in range(maxiter):
        if exit_early and not bool((active | checking).any()):
            stopped = True
            break

        # One product per trip: of the search direction, or of the solution
        # of a system whose recursive residual met the target.
        v = torch.where(checking, join(x, mu), join(p, zeros_m))
        image = matvec(v, *params)
        ap = image[..., :n]
        pap = vdot(p, ap, vdims)

        # Non-positive curvature on the constraints: A is not positive
        # definite on the null space of C.
        bad = ~((pap > 0) & (rg > 0))
        breakdown = breakdown | (active & bad)
        update = active & ~bad

        alpha = safe_divide(rg, pap)
        x = torch.where(update, x + alpha * p, x)
        r = torch.where(update, r + alpha * ap, r)

        restart, norm, checked, stagnation = _check(
            image - rhs, checking, target, norm, checked, stagnation, vdims
        )
        r = torch.where(restart, image[..., :n] - b, r)

        lam = multiplier_step(r)
        moved = update | restart
        r = torch.where(moved, r - times_ct(lam), r)
        mu = torch.where(moved, mu - lam, mu)

        g = project(r)
        rg_new = vdot(r, g, vdims)
        beta = safe_divide(rg_new, rg)
        p = torch.where(update, -g + beta * p, torch.where(restart, -g, p))
        rg = torch.where(moved, rg_new, rg)

        iterations = iterations + update.to(torch.int64)
        met = vnorm(join(r, times_c(x) - d), vdims) <= target
        checking = update & met
        active = (update & ~met) | restart

    solution = join(x, mu)
    residual = _true_residual(
        matvec, rhs, solution, params, norm, stopped, breakdown, vdims
    )
    return finalize(
        solution, residual, target, iterations, breakdown, vdims, stagnation
    )
