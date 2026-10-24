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
"""Convergence masks with explicitly invalid derivatives for failed systems."""

from __future__ import annotations

from typing import Any

import torch

from ...autograd.nested import nested_jvp
from ...tools.compile import is_compiling, reject_forward_mode
from ...typing import Tensor

__all__ = ["mask_converged"]


def _front(value: Tensor, dim: int | None, size: int) -> Tensor:
    """Make a vmap dimension an ordinary leading batch dimension."""
    if dim is None:
        return value.expand(size, *value.shape)
    return value.movedim(dim, 0)


class _ConvergenceMaskReverse(torch.autograd.Function):
    """Identity for accepted values, NaNs for failed values and their VJPs."""

    @staticmethod
    def forward(solution: Tensor, converged: Tensor) -> Tensor:
        return torch.where(converged, solution, torch.nan)

    @staticmethod
    def setup_context(
        ctx: Any, inputs: tuple[Tensor, Tensor], output: Tensor
    ) -> None:
        # Save the output rather than a detached NaN factor: its dependence
        # on the solution keeps failed higher derivatives invalid, even for
        # an otherwise linear solution whose Hessian would be zero.
        ctx.save_for_backward(output, inputs[1])
        ctx.save_for_forward(output, inputs[1])

    @staticmethod
    def backward(ctx: Any, grad: Tensor) -> tuple[Tensor, None]:
        output, converged = ctx.saved_tensors
        factor = torch.where(converged, 1.0, output)
        # A joint Jacobian supplies zero cotangents for unrelated output
        # lanes. Preserve those zeros before they reach a shared input;
        # multiplying them by NaN would contaminate successful output rows.
        return torch.where(grad == 0, grad, grad * factor), None

    @staticmethod
    def vmap(
        info: Any,
        in_dims: tuple[int | None, ...],
        solution: Tensor,
        converged: Tensor,
    ) -> tuple[Tensor, int]:
        return (
            mask_converged(
                _front(solution, in_dims[0], info.batch_size),
                _front(converged, in_dims[1], info.batch_size),
            ),
            0,
        )


class _ConvergenceMask(_ConvergenceMaskReverse):
    """Add the forward-mode rule, including nested eager forward transforms."""

    @staticmethod
    @nested_jvp
    def jvp(ctx: Any, solution_dot: Tensor, converged_dot: None) -> Tensor:
        output, converged = ctx.saved_tensors
        # Keep this as an AD operation too: a plain product with NaN would
        # poison valid shared-input derivatives in reverse-over-forward.
        return _ConvergenceTangent.apply(solution_dot, output, converged)


class _ConvergenceTangent(torch.autograd.Function):
    """Invalid tangents with a primal dependency for further differentiation."""

    @staticmethod
    def forward(tangent: Tensor, primal: Tensor, converged: Tensor) -> Tensor:
        return torch.where(converged, tangent, torch.nan)

    @staticmethod
    def setup_context(
        ctx: Any, inputs: tuple[Tensor, Tensor, Tensor], output: Tensor
    ) -> None:
        _, primal, converged = inputs
        ctx.save_for_backward(primal, converged)
        ctx.save_for_forward(primal, converged)

    @staticmethod
    def backward(ctx: Any, grad: Tensor) -> tuple[Tensor, Tensor, None]:
        primal, converged = ctx.saved_tensors
        # Ignore inactive output cotangents in both input paths. The primal
        # path is zero for valid systems and invalid for active failed ones,
        # keeping higher derivatives invalid even when the true tangent is
        # constant in the primal (e.g. a linear solve with respect to b).
        accepted = converged | (grad == 0)
        return (
            mask_converged(grad, accepted),
            mask_converged(torch.zeros_like(primal), accepted),
            None,
        )

    @staticmethod
    @nested_jvp
    def jvp(
        ctx: Any, tangent_dot: Tensor, primal_dot: Tensor, converged_dot: None
    ) -> Tensor:
        primal, converged = ctx.saved_tensors
        return _ConvergenceTangent.apply(tangent_dot, primal, converged)

    @staticmethod
    def vmap(
        info: Any,
        in_dims: tuple[int | None, ...],
        tangent: Tensor,
        primal: Tensor,
        converged: Tensor,
    ) -> tuple[Tensor, int]:
        return (
            _ConvergenceTangent.apply(
                _front(tangent, in_dims[0], info.batch_size),
                _front(primal, in_dims[1], info.batch_size),
                _front(converged, in_dims[2], info.batch_size),
            ),
            0,
        )


def mask_converged(solution: Tensor, converged: Tensor) -> Tensor:
    """
    Preserve accepted values and invalidate failed values and derivatives.

    This is the mask used by
    :func:`~tad_mctc.storch.linsolve.require_convergence`.
    It can also mask an application's own residual checks, without creating
    a solver result. It never raises merely because a mask entry is false;
    applications can raise eagerly before calling it under transforms.

    Parameters
    ----------
    solution : Tensor
        Real floating-point values to mask.
    converged : Tensor
        Boolean acceptance mask, broadcastable to ``solution.shape`` on the
        same device. For one mask entry per system, append singleton vector
        dimensions, e.g. ``converged.unsqueeze(-1)`` for atomic charges.

    Returns
    -------
    Tensor
        Same shape as ``solution``. Accepted entries and their derivatives
        are unchanged. Failed entries and their active derivatives are NaN;
        unrelated zero cotangents remain zero.

    Raises
    ------
    TypeError
        The mask is not boolean.
    RuntimeError
        The mask cannot broadcast to the solution.

    Notes
    -----
    Eager ``vmap``, ``jacrev``, ``jacfwd`` and their tested nested
    compositions are supported. Compiled first reverse derivatives with
    ordinary batch dimensions are supported on PyTorch >= 2.11. Compiled
    ``vmap`` differentiation is known incorrect and unsupported; compiled
    forward mode and higher derivatives are unsupported too.

    Select accepted systems *before* applying nonlinear operations to a
    masked result. For example, ``masked[0].square()`` is safe when entry 0
    is accepted, whereas ``masked.square()[0]`` can have a NaN first
    derivative: the square's backward can produce ``0 * NaN`` before this
    mask receives its cotangent. Alternatively, compute finite intermediate
    values and apply the mask at the final output boundary. Joint higher
    derivatives sharing inputs can also propagate NaNs outside the mask;
    select accepted outputs first or differentiate per system with ``vmap``.

    Example
    -------
    >>> import torch
    >>> from tad_mctc.storch.linsolve import mask_converged
    >>> x = torch.tensor(2.0, requires_grad=True)
    >>> accepted = torch.tensor([True, False])
    >>> masked = mask_converged(torch.stack((x, 2 * x)), accepted)
    >>> torch.isnan(masked).tolist()
    [False, True]
    >>> loss = masked[0].square()  # select before the nonlinear operation
    >>> torch.autograd.grad(loss, x)[0].item()
    4.0
    """
    if converged.dtype != torch.bool:
        raise TypeError("The convergence mask must have boolean dtype.")
    # Prevent a mask from expanding the solution to an unintended shape.
    converged = converged.expand_as(solution)
    if is_compiling():
        reject_forward_mode("convergence mask", solution)
        # As for implicit_solve, the view preserves compiled jacrev's input
        # dependence when the Function's forward does not define its VJP.
        return _ConvergenceMaskReverse.apply(
            solution.view_as(solution), converged
        )
    return _ConvergenceMask.apply(solution, converged)
