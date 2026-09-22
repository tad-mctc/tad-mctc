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
Autograd Utility: Gradcheck
===========================

Collection of utility functions for testing.
"""

from __future__ import annotations

import itertools
from typing import Any

import torch
from torch.autograd.gradcheck import gradcheck, gradgradcheck

from ..io.structure import Structure
from ..typing import Callable, Tensor, TensorOrTensors
from .internals import jacrev, vmap

__all__ = [
    "dgradcheck",
    "dgradgradcheck",
    "jacrev_matches_finite_diff",
    "numgrad",
    "vmap_matches_loop",
]


FAST_MODE = True
"""Default for fast_mode argument (True)."""


def _wrap_gradcheck(
    gradcheck_func: Callable[..., bool],
    func: Callable[..., TensorOrTensors],
    diffvars: TensorOrTensors,
    **kwargs: Any,
) -> bool:
    fast_mode = kwargs.pop("fast_mode", FAST_MODE)
    try:
        assert gradcheck_func(func, diffvars, fast_mode=fast_mode, **kwargs)
    finally:
        if isinstance(diffvars, Tensor):
            diffvars.detach_()
        elif isinstance(diffvars, (list, tuple)):
            for diffvar in diffvars:
                if isinstance(diffvar, Tensor):
                    diffvar.detach_()

    return True


def dgradcheck(
    func: Callable[..., TensorOrTensors],
    diffvars: TensorOrTensors,
    **kwargs: Any,
) -> bool:
    """
    Wrapper for `torch.autograd.gradcheck` that detaches the differentiated
    variables after the check.

    Parameters
    ----------
    func : Callable[..., TensorOrTensors]
        Forward function.
    diffvars : TensorOrTensors
        Variables w.r.t. which we differentiate.

    Returns
    -------
    bool
        Status of check.
    """
    return _wrap_gradcheck(gradcheck, func, diffvars, **kwargs)


def dgradgradcheck(
    func: Callable[..., TensorOrTensors],
    diffvars: TensorOrTensors,
    **kwargs: Any,
) -> bool:
    """
    Wrapper for `torch.autograd.gradgradcheck` that detaches the differentiated
    variables after the check.

    Parameters
    ----------
    func : Callable[..., TensorOrTensors]
        Forward function.
    diffvars : TensorOrTensors
        Variables w.r.t. which we differentiate.

    Returns
    -------
    bool
        Status of check.
    """
    return _wrap_gradcheck(gradgradcheck, func, diffvars, **kwargs)


def jacrev_matches_finite_diff(
    f: Callable[[Tensor], Tensor],
    x: Tensor,
    *,
    eps: float = 1.0e-4,
    atol: float = 1e-5,
) -> bool:
    """
    Check a `jacrev` Jacobian against a central finite-difference Jacobian,
    perturbing `x` element-wise.

    A 0-d `x` (a scalar parameter) takes the single-iteration case; a
    higher-rank `x` (e.g. a 3x3 lattice) loops over every index in its
    shape.

    Parameters
    ----------
    f : Callable[[Tensor], Tensor]
        Differentiable function of a single tensor argument.
    x : Tensor
        Point at which the Jacobian is evaluated.
    eps : float, optional
        Finite-difference step size. Defaults to `1.0e-4`.
    atol : float, optional
        Absolute tolerance passed to `torch.allclose`. Defaults to `1e-5`.

    Returns
    -------
    bool
        Whether the two Jacobians agree within `atol`.
    """
    jacobian: Tensor = jacrev(f)(x)
    numeric = torch.zeros_like(jacobian)

    for idx in itertools.product(*(range(s) for s in x.shape)):
        shifted = x.clone()
        shifted[idx] += eps
        plus = f(shifted)
        shifted[idx] -= 2 * eps
        minus = f(shifted)
        numeric[(..., *idx)] = (plus - minus) / (2 * eps)  # type: ignore[index]

    return torch.allclose(jacobian, numeric, atol=atol)


def numgrad(
    function: Callable[[Structure], Tensor],
    structure: Structure,
) -> Tensor:
    """
    Central-difference Jacobian of a per-atom quantity w.r.t. positions.

    Only ``structure.positions`` is perturbed; every other field (e.g.
    ``lattice``, for a periodic structure) is carried through unchanged on
    each perturbed copy, built with `Structure.replace`.

    Parameters
    ----------
    function : Callable[[Structure], Tensor]
        Forward function of a `Structure`, returning a ``(..., nat)``-shaped
        per-atom quantity (e.g. a coordination number).
    structure : Structure
        Structure at which the Jacobian is evaluated.

    Returns
    -------
    Tensor
        Numerical gradient, ``(..., nat, nat, 3)``: the first ``nat`` axis
        indexes ``function``'s output, the second indexes the perturbed atom.
    """
    numbers = structure.numbers
    nat = numbers.shape[-1]
    pos = structure.positions.clone()

    gradient = torch.zeros(
        (*numbers.shape[:-1], nat, nat, 3), dtype=pos.dtype, device=pos.device
    )

    # Central-difference truncation error shrinks as `step**2`, rounding
    # error grows as `eps/step`; the classic balance of the two is
    # `step ~ eps**(1/3)` (Numerical Recipes, ch. 5.7). Deriving it from
    # `pos`'s own dtype -- rather than a single value tuned for
    # `float64` -- keeps a `float32` call from picking a step so far
    # below `float32`'s ~1e-7 relative precision that `pos + step`
    # silently rounds back to `pos`.
    step = torch.finfo(pos.dtype).eps ** (1.0 / 3.0)

    for i in range(nat):
        for j in range(3):
            pos[..., i, j] += step
            cnr = function(structure.replace(positions=pos))

            pos[..., i, j] -= 2 * step
            cnl = function(structure.replace(positions=pos))

            pos[..., i, j] += step
            gradient[..., :, i, j] = 0.5 * (cnr - cnl) / step

    return gradient


def vmap_matches_loop(
    f: Callable[[Tensor], Tensor],
    batch: Tensor,
    *,
    atol: float = 1e-12,
    rtol: float = 0.0,
) -> bool:
    """
    Check that `vmap(f)` over `batch` matches a plain Python loop.

    Parameters
    ----------
    f : Callable[[Tensor], Tensor]
        Function of a single tensor argument, applied per batch element.
    batch : Tensor
        Batch of inputs, stacked along the leading dimension.
    atol : float, optional
        Absolute tolerance passed to `torch.allclose`. Defaults to `1e-12`.
    rtol : float, optional
        Relative tolerance passed to `torch.allclose`. Defaults to `0.0`.

    Returns
    -------
    bool
        Whether the vmapped and looped results agree within tolerance.
    """
    batched: Tensor = vmap(f)(batch)
    looped = torch.stack([f(x) for x in batch])
    return torch.allclose(batched, looped, atol=atol, rtol=rtol)
