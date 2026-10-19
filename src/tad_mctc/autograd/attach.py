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
Autograd Utility: Known Gradients
=================================

:func:`attach_gradient` evaluates a function whose gradient is known in
closed form as a differentiable tensor. Autograd then never differentiates
the evaluation of the function itself, which may be expensive to
differentiate (e.g. a quadrature) or not differentiable at all (e.g. via
eigenvalues at degeneracies). All derivatives of order ``k + 1`` come from
the derivatives of order ``k`` of the supplied gradient.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from ..tools.compile import is_compiling
from ..typing import Tensor
from .nested import nested_jvp

__all__ = ["attach_gradient"]


def _front(tensor: Tensor, dim: int | None, size: int) -> Tensor:
    if dim is None:
        return tensor.expand(size, *tensor.shape)
    return tensor.movedim(dim, 0)


class _AttachGradientReverse(torch.autograd.Function):
    """Value of ``function(x)`` with the supplied gradient (reverse mode)."""

    @staticmethod
    def forward(
        x: Tensor, gradient: Tensor, function: Callable[[Tensor], Tensor]
    ) -> Tensor:
        return function(x)

    @staticmethod
    def setup_context(ctx: Any, inputs: Any, output: Tensor) -> None:
        x, gradient, _ = inputs
        ctx.event_dims = x.ndim - output.ndim
        ctx.save_for_backward(gradient)
        ctx.save_for_forward(gradient)

    @staticmethod
    def backward(ctx: Any, grad: Tensor) -> tuple[Tensor, None, None]:
        (gradient,) = ctx.saved_tensors
        grad = grad.reshape(*grad.shape, *(1,) * ctx.event_dims)
        return grad * gradient, None, None

    @staticmethod
    def vmap(
        info: Any,
        in_dims: tuple[int | None, ...],
        x: Tensor,
        gradient: Tensor,
        function: Callable[[Tensor], Tensor],
    ) -> tuple[Tensor, int]:
        size = info.batch_size
        return (
            attach_gradient(
                function,
                _front(x, in_dims[0], size),
                _front(gradient, in_dims[1], size),
            ),
            0,
        )


class _AttachGradient(_AttachGradientReverse):
    """Adds the forward-mode rule (not supported by Dynamo)."""

    @staticmethod
    @nested_jvp
    def jvp(
        ctx: Any, x_dot: Tensor, gradient_dot: Tensor | None, _: None
    ) -> Tensor:
        (gradient,) = ctx.saved_tensors
        product = gradient * x_dot
        if ctx.event_dims == 0:
            return product
        return product.sum(tuple(range(-ctx.event_dims, 0)))


def attach_gradient(
    function: Callable[[Tensor], Tensor], x: Tensor, gradient: Tensor
) -> Tensor:
    """
    Evaluate ``function(x)`` with derivatives defined by a known gradient.

    ``function`` maps ``x`` of shape ``(..., *event)`` to one value per
    leading index, shape ``(...)``. Its own evaluation is never
    differentiated: the first derivative with respect to ``x`` is
    ``gradient``, and every higher derivative is the derivative of
    ``gradient``. The result is therefore exact to any order (reverse,
    forward and mixed modes, ``vmap``) if and only if ``gradient`` is the
    gradient of ``function`` computed from ``x`` with differentiable
    operations.

    Under ``torch.compile``, only first derivatives in reverse mode are
    correct: from PyTorch 2.11 on, nested derivatives of a Function inside
    compiled code silently miss the dependence of its backward on saved
    intermediate tensors (earlier versions raise). Take higher derivatives in
    eager mode.

    Parameters
    ----------
    function : Callable[[Tensor], Tensor]
        Value, ``function(x)``. It must accept leading batch dimensions.
    x : Tensor
        Input of shape ``(..., *event)``.
    gradient : Tensor
        Gradient of ``function`` at ``x``, same shape as ``x``.

    Returns
    -------
    Tensor
        ``function(x)``.

    Raises
    ------
    ValueError
        ``gradient`` and ``x`` differ in shape.

    Example
    -------
    >>> import torch
    >>> from torch.func import hessian
    >>> from tad_mctc.autograd import attach_gradient
    >>> lse = lambda x: torch.logsumexp(x, dim=-1)
    >>> f = lambda x: attach_gradient(lse, x, torch.softmax(x, dim=-1))
    >>> x = torch.tensor([0.0, 1.0], dtype=torch.double)
    >>> torch.allclose(hessian(f)(x), hessian(lse)(x))
    True
    """
    if gradient.shape != x.shape:
        raise ValueError(
            f"The gradient (shape {tuple(gradient.shape)}) must have the "
            f"shape of the input (shape {tuple(x.shape)})."
        )
    if is_compiling():
        # Dynamo drops the derivatives of Function inputs that are graph
        # inputs themselves (see `tad_mctc.storch.linsolve`); a view avoids it.
        return _AttachGradientReverse.apply(
            x.view_as(x), gradient.view_as(gradient), function
        )
    return _AttachGradient.apply(x, gradient, function)
