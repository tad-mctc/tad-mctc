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
Autograd Utility: Hessian
=========================

Utilities for calculating Hessians via autograd.

Note
----
Batched Hessians are not supported yet (via `vmap`).
"""

from __future__ import annotations

import torch

from ..typing import Any, Callable, Tensor
from .compat import jacrev_compat as jacrev

__all__ = ["hessian", "hess_fn_rev"]


def hessian(
    f: Callable[..., Tensor],
    inputs: tuple[Any, ...],
    argnums: int,
    is_batched: bool = False,
    **kwargs: Any,
) -> Tensor:
    """
    Wrapper for Hessian. The Hessian is the Jacobian of the gradient.

    PyTorch, however, suggests calculating the Jacobian of the Jacobian, which
    does not yield the correct shape in this case.

    Parameters
    ----------
    f : Callable[[Any], Tensor]
        The function whose result is differentiated.
    inputs : tuple[Any, ...]
        The input parameters of `f`.
    argnums : int, optional
        The variable w.r.t. which will be differentiated. Defaults to 0.

    Returns
    -------
    Tensor
        The Hessian.

    Raises
    ------
    RuntimeError
        The parameter selected for differentiation (via `argnums`) is not a
        tensor.
    """
    if not isinstance(inputs[argnums], Tensor):
        raise ValueError(
            f"The {argnums}'th input parameter must be a tensor but is of "
            f"type '{type(inputs[argnums])}'."
        )

    def _grad(*inps: tuple[Any, ...]) -> Tensor:
        e = f(*inps).sum()

        # catch missing gradients
        if e.grad_fn is None:
            return torch.zeros_like(inps[argnums])  # type: ignore

        (g,) = torch.autograd.grad(
            e,
            inps[argnums],
            create_graph=True,
        )
        return g

    _jac = jacrev(_grad, argnums=argnums, **kwargs)

    if is_batched:
        raise NotImplementedError("Batched Hessian not available.")
        # dims = tuple(None if x != argnums else 0 for x in range(len(inputs)))
        # _jac = torch.func.vmap(_jac, in_dims=dims)

    return _jac(*inputs)  # type: ignore


def hess_fn_rev(
    f: Callable[..., Tensor], argnums: tuple[int] | int = 0
) -> Callable:
    """
    Return the Hessian function using reverse-mode autodiff twice.
    (Functorch's `hessian` uses forward and backward mode, but forward is
    not implemented for our custom autograd functions.)

    Parameters
    ----------
    f : Callable[[Any], Tensor]
        The function whose result is differentiated.
    argnums : int or tuple[int], optional
        The variable w.r.t. which will be differentiated. Defaults to 0.

    Returns
    -------
    Callable
        A function that computes the Hessian of `f` with respect to the
        specified argument(s).
    """
    return torch.func.jacrev(
        torch.func.jacrev(f, argnums=argnums), argnums=argnums
    )
