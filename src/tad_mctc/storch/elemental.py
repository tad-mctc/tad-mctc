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

from typing import Any

import torch

from ..tools import is_compiling
from ..typing import Tensor
from .utils import get_eps

__all__ = ["safe_divide", "safe_pow", "safe_reciprocal", "safe_sqrt"]


def safe_divide(
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


def safe_reciprocal(
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


def safe_pow(
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
        # A Python number is a constant to Dynamo, so this check also runs
        # under `torch.compile`.
        if eps == 0:
            raise ValueError(
                "Value for clamping must be larger than 0.0, but "
                f"{eps} was given."
            )
        eps = torch.tensor(eps, device=x.device, dtype=x.dtype)
    elif isinstance(eps, Tensor):
        eps = eps.to(device=x.device, dtype=x.dtype)
    else:
        raise TypeError(
            "Value for clamping must be None (default), Tensor, float, or int, "
            f"but {type(eps)} was given."
        )

    # `(eps == 0).any()` reads a tensor's value, which is data-dependent
    # control flow that `torch.compile(fullgraph=True)` (Dynamo) rejects.
    # Skip the domain check of a `Tensor` eps while compiling, same as
    # `safe_sqrt`; eager mode still validates.
    if not is_compiling() and (eps == 0).any():
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
        # Branching on this predicate with an `if` would read a tensor's
        # value at trace time (data-dependent control flow, rejected by
        # `torch.compile(fullgraph=True)`). Unlike the `eps == 0`
        # check above, this is not a validation guard that can simply be
        # skipped while compiling: the two sides disagree at `x == 0` (fast
        # path leaves `0 ** positive_exponent == 0` untouched, slow path
        # would substitute `eps` for that `0` first). So the predicate is
        # kept -- as a tensor, not a Python `bool` -- and used to select the
        # *base* fed into a single `torch.pow` call instead of choosing
        # between two separate `torch.pow` calls. `torch.where` returns `x`
        # unchanged when the fast path applies (bit-identical to
        # `torch.pow(x, exponent)`) and the eps-clamped values otherwise
        # (bit-identical to `torch.pow` on the clamped base), so eager and
        # compiled results agree exactly. Selecting between the
        # two `torch.pow` *results* instead of the base would evaluate both,
        # and the discarded one is NaN for `x < 0` with a fractional
        # exponent, which would poison gradients through `torch.where`'s
        # backward (`0 * NaN = NaN`).
        fast_path_applies = (exponent > 0).all() & (x >= 0).all()
        x_clamped = torch.where(x <= 0, eps, x)
        x_safe = torch.where(fast_path_applies, x, x_clamped)
        return torch.pow(x_safe, exponent)

    raise ValueError(
        "Value for exponent must be integer, float, or Tensor, but "
        f"{type(exponent)} was given."
    )


def safe_sqrt(x: Tensor, *, eps: Tensor | float | int | None = None) -> Tensor:
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
        # A Python number is a constant to Dynamo, so this check also runs
        # under `torch.compile`.
        if eps < 0.0:
            raise ValueError(
                "Value for clamping must be larger than 0.0, but "
                f"{eps} was given."
            )
        eps = torch.tensor(eps, device=x.device, dtype=x.dtype)
    elif isinstance(eps, Tensor):
        eps = eps.to(device=x.device, dtype=x.dtype)
    else:
        raise TypeError(
            "Value for clamping must be None (default), Tensor, float, or int, "
            f"but {type(eps)} was given."
        )

    # `eps < 0.0` reads a tensor's value, which is data-dependent control
    # flow that `torch.compile(fullgraph=True)` (Dynamo) rejects. Skip the
    # domain check of a `Tensor` eps while compiling; eager mode still
    # validates.
    if not is_compiling() and eps < 0.0:
        raise ValueError(
            f"Value for clamping must be larger than 0.0, but {eps} was given."
        )

    return torch.sqrt(torch.clamp(x, min=eps))
