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
    Power ``x ** exponent`` that stays finite, with finite gradients, where
    ``torch.pow`` would give NaN or infinity.

    The result equals ``torch.pow(x, exponent)`` except at these points:

    - ``x == 0`` with a negative exponent, or a positive fractional
      exponent: the base is replaced by ``eps`` for the power (a negative
      exponent gives ``eps ** exponent``); for a positive fractional
      exponent the output is then set to exactly 0.
    - ``x < 0`` with a fractional exponent: the base is replaced by
      ``eps``, so the output is the constant ``eps ** exponent`` (not the
      signed magnitude ``-|x| ** exponent``).

    Negative bases with a whole-number exponent (also as a tensor) are
    exact, e.g. ``(-2) ** 3 == -8``.

    At every replaced position the base is chosen with ``torch.where``,
    so the gradient with respect to ``x`` there is 0 (it flows to ``eps``
    instead). Only exact zeros and negative values are guarded: tiny
    positive inputs still give huge or overflowing gradients. Overflow of
    the result is not handled either, e.g. ``0 ** -6`` is ``inf`` in
    float32 because ``eps ** -6`` exceeds the float32 range.

    Parameters
    ----------
    x : Tensor
        The base. Must be a tensor (its dtype and device are used).
    exponent : Tensor | float | int
        The exponent. A tensor exponent is handled without data-dependent
        branching, so it works under ``torch.compile(fullgraph=True)``.
    eps : Tensor | float | int | None, optional
        Replacement value, which must be larger than 0 (a zero, negative or
        NaN value raises ``ValueError``; a tensor is only checked in eager
        mode). Defaults to ``torch.finfo(x.dtype).eps`` (about 1.2e-7 for
        float32 and 2.2e-16 for float64).

    Returns
    -------
    Tensor
        The output tensor.

    Raises
    ------
    ValueError
        If ``eps`` is not larger than 0 or ``exponent`` has an unsupported
        type.
    TypeError
        If ``eps`` has an unsupported type.
    """
    if eps is None:
        eps = get_eps(x)
    elif isinstance(eps, (float, int)):
        # A Python number is a constant to Dynamo, so this check also runs
        # under `torch.compile`.
        if not eps > 0:
            raise ValueError(
                "Value for clamping must be larger than 0.0, but "
                f"{eps} was given."
            )
        eps = torch.tensor(eps, device=x.device, dtype=x.dtype)
    elif isinstance(eps, Tensor):
        eps = eps.to(device=x.device, dtype=x.dtype)
        # `(eps > 0).all()` reads a tensor's value, which is data-dependent
        # control flow that `torch.compile(fullgraph=True)` (Dynamo)
        # rejects. Skip the domain check of a `Tensor` eps while compiling,
        # same as `safe_sqrt`; eager mode still validates. Only a
        # user-supplied `eps` is checked, so the default path never syncs
        # with the device. Negating `> 0` also rejects NaN.
        if not is_compiling() and not (eps > 0).all():
            raise ValueError(
                "Value for clamping must be larger than 0.0, but "
                f"{eps} was given."
            )
    else:
        raise TypeError(
            "Value for clamping must be None (default), Tensor, float, or int, "
            f"but {type(eps)} was given."
        )

    def _int(x: Tensor, exponent: int) -> Tensor:
        # integer positive exponents are safe
        if exponent > 0:
            return torch.pow(x, exponent)

        # integer negative exponents fail for x = 0
        x = torch.where(x == 0, eps, x)
        return torch.pow(x, exponent)

    def _float(x: Tensor, exponent: float | Tensor) -> Tensor:
        # float positive exponents fail for x < 0, and their higher
        # derivatives are infinite at x = 0 (e.g. `x**1.5`): evaluate the
        # power at a safe base there and put the exact value 0 back, so the
        # derivatives at the masked point stay finite.
        if exponent > 0:
            safe = torch.where(x <= 0, eps, x)
            return torch.where(x == 0, 0.0, torch.pow(safe, exponent))

        # float negative exponents fail for x <= 0
        x = torch.where(x <= 0, eps, x)
        return torch.pow(x, exponent)

    if isinstance(exponent, int):
        return _int(x, exponent)

    if isinstance(exponent, float):
        # `exponent % 1`, not `float.is_integer`, which Dynamo cannot trace
        if exponent % 1 == 0:
            return _int(x, int(exponent))

        return _float(x, exponent)

    if isinstance(exponent, Tensor):
        # The sign of a tensor exponent cannot be branched on with an `if`:
        # that would read a tensor's value at trace time (data-dependent
        # control flow, rejected by `torch.compile(fullgraph=True)`). So
        # every case is selected with `torch.where` on the *base* fed into
        # a single `torch.pow` call: selecting between two `torch.pow`
        # *results* instead would evaluate both, and the discarded one is
        # NaN for `x < 0` with a fractional exponent, which would poison
        # gradients through `torch.where`'s backward (`0 * NaN = NaN`).
        #
        # Whole-number exponents are well defined for negative bases, so
        # they use `|x|` and restore the sign for odd exponents; this keeps
        # `safe_pow(x, 2)` and `safe_pow(x, torch.tensor(2.0))` equal. Other
        # negative bases are replaced by `eps`, as are zeros where `0 **
        # exponent` is singular. Positive whole-number exponents are smooth
        # at 0 (like `_int`), so `x == 0` is left alone for them; positive
        # fractional exponents get their exact value 0 put back afterwards.
        is_whole = exponent == torch.round(exponent)
        is_odd = is_whole & (torch.remainder(exponent, 2) == 1)
        smooth = is_whole & (exponent > 0)
        negative = x < 0

        magnitude = torch.where(negative & is_whole, -x, x)
        singular = ((x == 0) & ~smooth) | (negative & ~is_whole)
        power = torch.pow(torch.where(singular, eps, magnitude), exponent)

        is_zero = (x == 0) & (exponent > 0) & ~smooth
        power = torch.where(is_zero, 0.0, power)
        return torch.where(negative & is_odd, -power, power)

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
