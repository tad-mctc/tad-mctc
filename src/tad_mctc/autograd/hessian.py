# This file is part of tad-mctc.
#
# SPDX-Identifier: LGPL-3.0
# Copyright (C) 2023 Marvin Friede
#
# tad-mctc is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# tad_mctc is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with tad-mctc. If not, see <https://www.gnu.org/licenses/>.
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
from .internals import jacrev

__all__ = ["hessian"]


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
