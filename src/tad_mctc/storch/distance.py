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
SafeOps: Distance
=================

Functions for calculating the cartesian distance of two vectors.
"""

from __future__ import annotations

import torch

from ..math import einsum
from ..typing import Tensor
from .elemental import safe_sqrt

__all__ = ["cdist"]


def euclidean_dist_quadratic_expansion(x: Tensor, y: Tensor) -> Tensor:
    """
    Computation of euclidean distance matrix via quadratic expansion,
    ``|x|^2 + |y|^2 - 2 x.y``.

    Only for euclidean (p=2) distances, and not the default of
    :func:`cdist`: the expansion cancels terms of size ``|x|^2``, so its
    rounding error grows with the distance of the points from the origin,
    not with the distances themselves. In float32, for points in a 30 Bohr
    box, a 2 Bohr distance comes out with a relative error of about
    ``1e-3``; shifted 1000 Bohr from the origin, close points collapse
    entirely. A distance of a point to itself is the square root of a
    rounding residual, between ``sqrt(eps)`` and a few times
    ``sqrt(eps * |x|^2)``, depending on the device.

    It is faster than :func:`cdist_direct_expansion` for large inputs on
    CPU (about 2x at 5000 points), and it needs half the memory, since it
    never forms the ``(..., n, m, 3)`` differences.

    For more information, see \
    `this Jupyter notebook <https://github.com/eth-cscs/PythonHPC/blob/master/\
    numpy/03-euclidean-distance-matrix-numpy.ipynb>`__ or \
    `this discussion thread in the PyTorch forum <https://discuss.pytorch.org/\
    t/efficient-distance-matrix-computation/9065>`__.

    Parameters
    ----------
    x : Tensor
        First tensor.
    y : Tensor
        Second tensor (with same shape as first tensor).

    Returns
    -------
    Tensor
        Pair-wise distance matrix.
    """
    eps = torch.tensor(
        torch.finfo(x.dtype).eps,
        device=x.device,
        dtype=x.dtype,
    )

    # using einsum is slightly faster than `torch.pow(x, 2).sum(-1)`
    xnorm = einsum("...ij,...ij->...i", x, x)
    ynorm = einsum("...ij,...ij->...i", y, y)

    n = xnorm.unsqueeze(-1) + ynorm.unsqueeze(-2)

    # "...ik,...jk->...ij"
    prod = x @ y.mT

    # important: remove negative values that give NaN in backward
    return safe_sqrt(n - 2.0 * prod, eps=eps)


def cdist_direct_expansion(x: Tensor, y: Tensor, p: int = 2) -> Tensor:
    """
    Computation of cartesian distance matrix from direct differences.

    Exact to rounding for every pair, wherever the points are. The distance
    of a point to itself is ``eps ** (1 / p)``: the sum is clamped to
    ``eps`` so that the gradient stays finite.

    Parameters
    ----------
    x : Tensor
        First tensor.
    y : Tensor
        Second tensor (with same shape as first tensor).
    p : int, optional
        Power used in the distance evaluation (p-norm). Defaults to 2.

    Returns
    -------
    Tensor
        Pair-wise distance matrix.
    """
    eps = torch.finfo(x.dtype).eps

    # unsqueeze different dimension to create matrix
    diff = x.unsqueeze(-2) - y.unsqueeze(-3)

    # `sqrt` rather than `pow(..., 0.5)`: a cheaper backward.
    if p == 2:
        return torch.sqrt(torch.clamp((diff * diff).sum(-1), min=eps))

    # An even power needs no absolute value.
    if p % 2 != 0:
        diff = torch.abs(diff)
    distances = torch.sum(torch.pow(diff, p), -1)
    return torch.pow(torch.clamp(distances, min=eps), 1.0 / p)


def cdist(x: Tensor, y: Tensor | None = None, p: int = 2) -> Tensor:
    """
    Wrapper for cartesian distance computation.

    This currently replaces the use of ``torch.cdist``, which does not handle
    zeros well and produces nan's in the backward pass.

    Additionally, ``torch.cdist`` does not return zero for distances between
    same vectors (see `here
    <https://github.com/pytorch/pytorch/issues/57690>`__).

    The distances come from direct differences
    (:func:`cdist_direct_expansion`), exact to rounding wherever the points
    are. The distance of a vector to itself is ``eps ** (1 / p)``, the same
    on every device. :func:`euclidean_dist_quadratic_expansion` is faster for
    large inputs on CPU, at the accuracy cost described there.

    Parameters
    ----------
    x : Tensor
        First tensor.
    y : Tensor | None, optional
        Second tensor. If no second tensor is given (default), the first tensor
        is used as the second tensor, too.
    p : int, optional
        Power used in the distance evaluation (p-norm). Defaults to 2.

    Returns
    -------
    Tensor
        Pair-wise distance matrix.
    """
    if y is None:
        y = x

    return cdist_direct_expansion(x, y, p=p)
