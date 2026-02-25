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
SafeOps: Elementary Functions
=============================

Safe versions of elementary functions like `sqrt` or `abs`.
"""

from __future__ import annotations

import torch

from ..typing import Any, Tensor
from .utils import get_eps

__all__ = ["divide", "pow", "reciprocal", "sqrt"]


def divide(
    x: Tensor,
    y: Tensor,
    *,
    eps: Tensor | float | int | None = None,
    **kwargs: Any,
) -> Tensor:
    """
    Safe divide operation.
    Only adds a small value to the denominator where it is zero.

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

    y_safe = torch.where(y != 0, y, eps)
    return torch.divide(x, y_safe, **kwargs)


def reciprocal(
    x: Tensor, *, eps: Tensor | float | int | None = None, **kwargs: Any
) -> Tensor:
    """
    Safe reciprocal operation.

    Parameters
    ----------
    x : Tensor
        Input tensor (denominator).
    eps : Tensor | float | int | None, optional
        Value added to the denominator. Defaults to `None`, which resolves to
        `torch.finfo(x.dtype).eps`.

    Returns
    -------
    Tensor
        Reciprocal of the input tensor.

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

    one = torch.tensor(1.0, device=x.device, dtype=x.dtype)
    return torch.divide(one, x + eps, **kwargs)


def pow(
    x: Tensor,
    exponent: Tensor | float | int,
    *,
    eps: Tensor | float | int | None = None,
) -> Tensor:
    """
    Takes the power of each element in input with exponent and returns a tensor with the result.

    This is a safer version of ``torch.pow`` (``out = x ** exponent``), which avoids:

    1. NaN/imaginary output when ``x < 0`` and exponent has a fractional part
        In this case, the function returns the signed (negative) magnitude of the complex number.

    2. NaN/infinite gradient at ``x = 0`` when exponent has a fractional part
        In this case, the positions of 0 are added by ``epsilon``,
        so the gradient is back-propagated as if ``x = epsilon``.

    However, this function doesn't deal with float overflow, such as 1e10000.

    Parameters
    ----------
    x : torch.Tensor or float
        The input base value.

    exponent : torch.Tensor or float
        The exponent value.

        (At least one of ``x`` and ``exponent`` must be a torch.Tensor)

    epsilon : float
        A small floating point value to avoid infinite gradient. Default: 1e-6

    Returns
    -------
    out : torch.Tensor
        The output tensor.
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

    if (eps == 0).any():
        raise ValueError(
            f"Value for clamping must be larger than 0.0, but {eps} was given."
        )

    def _int(x: Tensor, exponent: int) -> Tensor:
        # integer positive exponents are safe
        if exponent > 0:
            return torch.pow(x, exponent)

        # integer negative exponents fail for x = 0
        x = torch.where(x == 0, eps, x)
        return torch.pow(x, exponent)

    def _float(x: Tensor, exponent: float | Tensor) -> Tensor:
        # float positive exponents fail for x < 0
        if exponent > 0:
            x = torch.where(x < 0, eps, x)
            return torch.pow(x, exponent)

        # float negative exponents fail for x <= 0
        x = torch.where(x <= 0, eps, x)
        return torch.pow(x, exponent)

    if isinstance(exponent, int):
        return _int(x, exponent)

    if isinstance(exponent, float):
        if exponent.is_integer():
            return _int(x, int(exponent))

        return _float(x, exponent)

    if isinstance(exponent, Tensor):
        # integer positive exponents are safe
        if (exponent > 0).all() & (x >= 0).all():
            return torch.pow(x, exponent)

        # float negative exponents fail for x <= 0
        x = torch.where(x <= 0, eps, x)
        return torch.pow(x, exponent)

    raise ValueError(
        "Value for exponent must be integer, float, or Tensor, but "
        f"{type(exponent)} was given."
    )


def sqrt(x: Tensor, *, eps: Tensor | float | int | None = None) -> Tensor:
    """
    Safe square root operation.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    eps : Tensor | float | int | None, optional
        Value for clamping. Defaults to ``None``, which resolves to
        ``torch.finfo(x.dtype).eps``.

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
