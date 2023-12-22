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
SafeOps: Elementary Functions
=============================

Safe versions of elementary functions like `sqrt` or `abs`.
"""
from __future__ import annotations

import torch

from ..typing import Any, Tensor
from .utils import get_eps

__all__ = ["divide", "sqrt"]


def divide(
    x: Tensor, y: Tensor, *, eps: Tensor | float | int | None = None, **kwargs: Any
) -> Tensor:
    """
    Safe divide operation.

    Parameters
    ----------
    x : Tensor
        Input tensor (nominator).
    y : Tensor
        Input tensor (denominator).
    eps : Tensor | float | int | None, optional
        Value added to the denominator. Defaults to `None`, which resolves to
        `torch.finfo(x.dtype).eps`.

    Returns
    -------
    Tensor
        Square root of the input tensor.

    Raises
    ------
    TypeError
        Value for addition to denominator has wrong type.
    """
    if eps is None:
        eps = get_eps(x)
    elif isinstance(eps, (float, int)):
        eps = torch.tensor(eps, device=x.device, dtype=x.dtype)
    elif isinstance(eps, Tensor):
        eps = eps.to(device=x.device, dtype=x.dtype)
    else:
        raise TypeError(
            "Value for clamping must be None (default), Tensor, float, or int, "
            f"but {type(eps)} was given."
        )

    return torch.divide(x, (y + eps), **kwargs)


def sqrt(x: Tensor, eps: Tensor | float | int | None = None) -> Tensor:
    """
    Safe square root operation.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    eps : Tensor | float | int | None, optional
        Value for clamping. Defaults to `None`, which resolves to
        `torch.finfo(x.dtype).eps`.

    Returns
    -------
    Tensor
        Square root of the input tensor.

    Raises
    ------
    TypeError
        Value for clamping has wrong type.
    """
    if eps is None:
        eps = get_eps(x)
    elif isinstance(eps, (float, int)):
        eps = torch.tensor(eps, device=x.device, dtype=x.dtype)
    elif isinstance(eps, Tensor):
        eps = eps.to(device=x.device, dtype=x.dtype)
    else:
        raise TypeError(
            "Value for clamping must be None (default), Tensor, float, or int, "
            f"but {type(eps)} was given."
        )

    if eps < 0.0:
        raise ValueError(
            f"Value for clamping must be larger than 0.0, but {eps} was given."
        )

    return torch.sqrt(torch.clamp(x, min=eps))
