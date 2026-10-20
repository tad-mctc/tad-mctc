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
Linear Solve: Result
====================

Structured result of a linear solve and the status codes it reports.

All diagnostics are tensors with the batch shape of the right-hand side, so
that a result can be returned from ``torch.func.vmap`` and from code compiled
with ``torch.compile(fullgraph=True)``.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import torch

from ...autograd.checks import is_vmapped
from ...exceptions import ConvergenceError, DeviceError, DtypeError
from ...typing import Tensor

__all__ = [
    "LinearSolveResult",
    "require_convergence",
    "STATUS_CONVERGED",
    "STATUS_NOT_CONVERGED",
    "STATUS_BREAKDOWN",
    "STATUS_NONFINITE",
]


STATUS_CONVERGED = 0
"""The true residual satisfies the requested tolerance."""

STATUS_NOT_CONVERGED = 1
"""The iteration limit was reached without satisfying the tolerance."""

STATUS_BREAKDOWN = 2
"""
The method broke down before satisfying the tolerance, e.g. CG found a
direction of non-positive curvature (the operator is not positive definite)
or a BiCGSTAB denominator vanished.
"""

STATUS_NONFINITE = 3
"""The residual is not finite (NaN or infinite input or iterate)."""


class LinearSolveResult(NamedTuple):
    """
    Solution and diagnostics of a linear solve ``A x = b``.

    The batch shape is the shape of the right-hand side without its trailing
    vector dimensions (see ``vdims`` of the solvers).
    """

    solution: Tensor
    """Solution ``x``, same shape as the right-hand side."""

    residual: Tensor
    """Norm of the true residual ``b - A x``, recomputed after the
    iteration (batch shape)."""

    target: Tensor
    """Residual norm required for convergence (batch shape)."""

    iterations: Tensor
    """Number of iterations each system took (batch shape, ``int64``)."""

    status: Tensor
    """Status code of each system (batch shape, ``int64``), one of
    :data:`STATUS_CONVERGED`, :data:`STATUS_NOT_CONVERGED`,
    :data:`STATUS_BREAKDOWN` or :data:`STATUS_NONFINITE`."""

    @property
    def converged(self) -> Tensor:
        """
        Whether each system converged (batch shape, ``bool``).

        Under ``torch.compile`` with PyTorch < 2.8, compare
        :attr:`status` with :data:`STATUS_CONVERGED` instead, since Dynamo
        does not trace this property.
        """
        return self.status == STATUS_CONVERGED


_EPS_MULTIPLIER = 10.0
"""
A relative tolerance below roundoff rejects useful solutions in low
precision. The relative tolerance is therefore never smaller than this
multiple of the machine epsilon of the dtype.
"""


def require_real(name: str, tensor: Tensor | None) -> None:
    """
    Raise unless ``tensor`` is ``None`` or real floating point.

    The solvers use real inner products and transposes, not adjoints, so
    their values and derivatives are wrong for complex tensors.
    """
    if tensor is not None and not tensor.is_floating_point():
        raise DtypeError(
            f"The {name} must be a real floating point tensor, not "
            f"{tensor.dtype}."
        )


def require_real_params(params: tuple[Any, ...]) -> None:
    """
    Raise for complex tensors among the parameters of an operator.

    Integer and boolean tensors are allowed as structural arguments (e.g.
    indices); they receive no derivatives.
    """
    for i, param in enumerate(params):
        if isinstance(param, Tensor) and param.is_complex():
            raise DtypeError(
                f"Parameter {i} of the operator is complex ({param.dtype}); "
                "only real tensors are supported."
            )


def require_like(
    name: str, tensor: Tensor, reference_name: str, reference: Tensor
) -> None:
    """
    Raise unless ``tensor`` has the dtype and device of ``reference``.

    Catches operators that return complex or mixed-precision results, also
    from tensors captured in a closure, before any iteration.
    """
    if tensor.dtype != reference.dtype:
        raise DtypeError(
            f"The {name} has dtype {tensor.dtype}, but the {reference_name} "
            f"has dtype {reference.dtype}."
        )
    if tensor.device != reference.device:
        raise DeviceError(
            f"The {name} is on device {tensor.device}, but the "
            f"{reference_name} is on device {reference.device}."
        )


def vdot(a: Tensor, b: Tensor, vdims: int) -> Tensor:
    """
    Real inner product over the trailing ``vdims`` dimensions.

    The vector dimensions are kept (with size one) so that the result
    broadcasts against the vectors.
    """
    return torch.sum(a * b, dim=tuple(range(-vdims, 0)), keepdim=True)


def vnorm(a: Tensor, vdims: int) -> Tensor:
    """Euclidean norm over the trailing ``vdims`` dimensions (kept)."""
    return torch.linalg.vector_norm(
        a, dim=tuple(range(-vdims, 0)), keepdim=True
    )


def safe_divide(numerator: Tensor, denominator: Tensor) -> Tensor:
    """
    Division that replaces a vanishing denominator by one.

    The affected lanes are either inactive or reported as breakdown by the
    caller; the replacement only keeps their values finite.
    """
    tiny = torch.finfo(denominator.dtype).tiny
    safe = torch.where(
        denominator.abs() > tiny, denominator, torch.ones_like(denominator)
    )
    return numerator / safe


def convergence_target(
    rhs: Tensor, rtol: float, atol: float, vdims: int
) -> Tensor:
    r"""
    Residual norm required for convergence.

    .. math::

        \|b - A x\| \leq \mathrm{atol} +
        \max(\mathrm{rtol}, 10 \epsilon) \, \|b\|

    with the machine epsilon :math:`\epsilon` of the dtype of ``rhs``.
    """
    effective_rtol = max(rtol, _EPS_MULTIPLIER * torch.finfo(rhs.dtype).eps)
    return atol + effective_rtol * vnorm(rhs, vdims)


def finalize(
    solution: Tensor,
    residual: Tensor,
    target: Tensor,
    iterations: Tensor,
    breakdown: Tensor,
    vdims: int,
) -> LinearSolveResult:
    """
    Assemble the result from the true residual and the iteration state.

    The status is decided by the true residual: a lane that satisfies the
    tolerance is converged, even if the method flagged a breakdown on the
    way (e.g. a vanishing denominator after the residual already became
    zero).
    """
    dims = tuple(range(-vdims, 0))
    residual = residual.squeeze(dims) if vdims > 0 else residual
    target = target.squeeze(dims) if vdims > 0 else target
    breakdown = breakdown.squeeze(dims) if vdims > 0 else breakdown
    iterations = iterations.squeeze(dims) if vdims > 0 else iterations

    status = torch.where(
        residual <= target,
        STATUS_CONVERGED,
        torch.where(
            ~torch.isfinite(residual),
            STATUS_NONFINITE,
            torch.where(breakdown, STATUS_BREAKDOWN, STATUS_NOT_CONVERGED),
        ),
    )

    return LinearSolveResult(
        solution=solution,
        residual=residual,
        target=target,
        iterations=iterations,
        status=status,
    )


_STATUS_NAMES = {
    STATUS_NOT_CONVERGED: "iteration limit reached",
    STATUS_BREAKDOWN: "breakdown (operator not positive definite?)",
    STATUS_NONFINITE: "non-finite residual",
}


def require_convergence(
    result: LinearSolveResult, vdims: int = 1, name: str = "Linear solve"
) -> Tensor:
    """
    Solution of a linear solve that must have converged.

    Outside of ``torch.func.vmap`` and ``torch.compile``, a system that did
    not converge raises. Under these transforms, the status cannot be
    inspected in Python, and the solution of every such system is replaced
    by NaN instead (fail-closed), while the other systems are unaffected.

    Parameters
    ----------
    result : LinearSolveResult
        Result of the solve.
    vdims : int, optional
        Number of trailing dimensions of the solution forming the vector.
        Defaults to ``1``.
    name : str, optional
        Name of the solve for the error message.

    Returns
    -------
    Tensor
        The solution.

    Raises
    ------
    ConvergenceError
        A system did not converge (not under ``vmap`` or ``torch.compile``).
    """
    converged = result.status == STATUS_CONVERGED

    if not is_vmapped(converged):
        if not bool(converged.all()):
            failed = result.status[~converged]
            reasons = sorted({_STATUS_NAMES[int(c)] for c in failed.unique()})
            raise ConvergenceError(
                f"{name} did not converge for {failed.numel()} of "
                f"{converged.numel()} system(s): {', '.join(reasons)}. "
                f"Largest residual {result.residual.max().item():.3e}, "
                f"target {result.target.max().item():.3e}."
            )
        return result.solution

    for _ in range(vdims):
        converged = converged.unsqueeze(-1)
    nan = torch.full_like(result.solution, torch.nan)
    return torch.where(converged, result.solution, nan)
