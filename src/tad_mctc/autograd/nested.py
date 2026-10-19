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
Autograd Utility: Nested Forward Mode
=====================================

PyTorch evaluates the ``jvp`` staticmethod of a ``torch.autograd.Function``
with forward-mode AD disabled. An enclosing forward-mode transform therefore
does not see the computation of the tangent: ``jacfwd(jacfwd(f))`` and
``jvp(jvp(f))`` silently miss every term from the dependence of the tangent
on the inputs, typically returning zero (PyTorch 2.7 to 2.14). Reverse mode
is not affected, so ``jacfwd(jacrev(f))`` and ``jacrev(jacfwd(f))`` are
correct.

:func:`nested_jvp` fixes this for a ``jvp`` rule built from differentiable
operations: as ``torch.func`` does for ``forward``, the rule is evaluated one
``torch.func`` level lower, with forward-mode AD enabled, so that only the
enclosing transforms track it.

With ``generate_vmap_rule = True``, ``torch.func`` evaluates the ``jvp`` rule
of a ``vmap``-ped Function inside a temporary ``vmap`` level, where it cannot
be lowered. This matters only for a third forward-mode level through
``vmap`` (``jacfwd(jacfwd(jacfwd(f)))``), which :func:`nested_jvp` then
rejects instead of returning wrong derivatives. A custom ``vmap``
staticmethod that re-applies the Function to batched inputs avoids it.

This relies on private ``torch.func`` interfaces.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

from torch._C._functorch import (  # pylint: disable=no-name-in-module
    TransformType,
    _unwrap_for_grad,
    _wrap_for_grad,
    get_interpreter_stack,
)
from torch._functorch.pyfunctorch import retrieve_current_functorch_interpreter
from torch.autograd.forward_ad import _set_fwd_grad_enabled
from torch.utils import _pytree as pytree

from ..typing import Tensor

__all__ = ["nested_jvp"]


class _LoweredContext:
    """
    View of a Function context whose saved tensors and tensor attributes are
    unwrapped from one ``torch.func`` level.
    """

    def __init__(self, ctx: Any, level: int) -> None:
        self._ctx = ctx
        self._level = level

    def _unwrap(self, value: Any) -> Any:
        if isinstance(value, Tensor):
            return _unwrap_for_grad(value, self._level)
        return value

    @property
    def saved_tensors(self) -> tuple[Any, ...]:
        return tuple(self._unwrap(t) for t in self._ctx.saved_tensors)

    def __getattr__(self, name: str) -> Any:
        return pytree.tree_map(self._unwrap, getattr(self._ctx, name))


def nested_jvp(rule: Callable[..., Any]) -> Callable[..., Any]:
    """
    Make the ``jvp`` rule of a ``torch.autograd.Function`` differentiable
    by enclosing forward-mode transforms (see module docstring).

    Apply it below ``@staticmethod``. The rule must compute the tangent
    from differentiable operations on the context and the input tangents.

    Parameters
    ----------
    rule : Callable[..., Any]
        The ``jvp`` staticmethod, ``rule(ctx, *tangents)``.

    Returns
    -------
    Callable[..., Any]
        The wrapped rule.

    Raises
    ------
    NotImplementedError
        The rule runs inside a temporary ``vmap`` level of a generated
        batching rule while an enclosing forward-mode transform needs its
        derivative (see module docstring).

    Example
    -------
    >>> import torch
    >>> from torch.func import jacfwd
    >>> from tad_mctc.autograd import nested_jvp
    >>> class Cube(torch.autograd.Function):
    ...     generate_vmap_rule = True
    ...     @staticmethod
    ...     def forward(x):
    ...         return x**3
    ...     @staticmethod
    ...     def setup_context(ctx, inputs, output):
    ...         ctx.save_for_forward(inputs[0])
    ...     @staticmethod
    ...     @nested_jvp
    ...     def jvp(ctx, x_dot):
    ...         (x,) = ctx.saved_tensors
    ...         return 3 * x**2 * x_dot
    >>> x = torch.tensor(2.0)
    >>> jacfwd(jacfwd(Cube.apply))(x)  # 6 x
    tensor(12.)
    """

    @functools.wraps(rule)
    def jvp(ctx: Any, *tangents: Any) -> Any:
        stack = get_interpreter_stack()
        if not stack:
            # `torch.autograd.forward_ad`, which cannot be nested
            return rule(ctx, *tangents)

        interpreter = retrieve_current_functorch_interpreter()
        if interpreter.key() != TransformType.Jvp:
            outer = [i for i in stack if i.key() == TransformType.Jvp]
            if len(outer) > 1:
                raise NotImplementedError(
                    f"The forward-mode rule of '{rule.__qualname__}' runs in "
                    "a generated `vmap` rule below another forward-mode "
                    "transform, which cannot see it (e.g. "
                    "`jacfwd(jacfwd(jacfwd(f)))`). Use reverse mode for "
                    "the outer derivatives, e.g. `jacfwd(jacrev(jacrev(f)))`."
                )
            return rule(ctx, *tangents)

        level = interpreter.level()

        def unwrap(value: Any) -> Any:
            if isinstance(value, Tensor):
                return _unwrap_for_grad(value, level)
            return value

        with interpreter.lower(), _set_fwd_grad_enabled(True):
            out = rule(
                _LoweredContext(ctx, level), *(unwrap(t) for t in tangents)
            )

        return pytree.tree_map_only(
            Tensor, lambda t: _wrap_for_grad(t, level), out
        )

    return jvp
