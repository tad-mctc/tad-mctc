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
Coordination number: Counting functions
=======================================

This module contains all the counting functions used throughout the projects of
the Grimme group. This includes the following counting functions:
- exponential (DFT-D3, EEQ)
- error function (DFT-D4)
- double exponential (GFN2-xTB)
"""

from __future__ import annotations

import torch

from .. import storch
from ..typing import Tensor
from . import defaults

__all__ = [
    "exp_count",
    "erf_count",
    "gfn2_count",
]


def exp_count(
    r: Tensor, r0: Tensor, kcn: Tensor | float | int = defaults.KCN_D3
) -> Tensor:
    """
    Exponential counting function for coordination number contributions.

    Parameters
    ----------
    r : Tensor
        Internuclear distances.
    r0 : Tensor
        Covalent atomic radii (R_AB = R_A + R_B).
    kcn : Tensor | float | int, optional
        Steepness of the counting function. Defaults to
        :data:`tad_mctc.ncoord.defaults.KCN_D3`.

    Returns
    -------
    Tensor
        Count of coordination number contribution.
    """
    # The counting function `1 / (1 + exp(-x))`, `x = kcn * (r0/r - 1)`,
    # is `sigmoid(x)`. `torch.sigmoid` computes it in one fused pass
    # instead of three full-array passes (`exp`, `+ 1.0`, `1.0 /`), and its
    # backward reuses the saved output (`grad * y * (1 - y)`). Measured
    # ~34% faster in float32 and float64 on a 188M-element array (the pair
    # count of a ~1.7M-atom system); the two forms differ only by rounding.
    return torch.sigmoid(kcn * (storch.safe_divide(r0, r) - 1.0))


def erf_count(
    r: Tensor,
    r0: Tensor,
    kcn: Tensor | float | int = defaults.KCN_D4,
    norm_exp: Tensor | float | int = 1.0,
) -> Tensor:
    """
    Error function counting function for coordination number contributions.

    Parameters
    ----------
    r : Tensor
        Internuclear distances.
    r0 : Tensor
        Covalent atomic radii (R_AB = R_A + R_B).
    kcn : Tensor | float | int, optional
        Steepness of the counting function. Defaults to
        :data:`tad_mctc.ncoord.defaults.KCN_D3`.
    norm_exp : Tensor | float | int, optional
        Exponent applied to ``r0`` in the normalization (used by the EEQBC
        coordination number). Defaults to ``1.0``, which recovers the
        standard error-function counting function used by DFT-D4 and EEQ.

    Returns
    -------
    Tensor
        Count of coordination number contribution.
    """
    # `norm_exp = 1.0` is the default (D4/EEQ; EEQBC overrides it to 0.75,
    # see `tad_mctc.ncoord.eeqbc`) and `r0**1.0` is not a no-op PyTorch
    # optimises away -- it is a full elementwise `pow` kernel over the
    # whole pair array for a value that never changes, measured at
    # ~0.38s/0.77s (float32/float64) wasted on a 188M-element array. The
    # `isinstance` check (mirroring `storch.safe_divide`'s own) keeps this
    # a static, non-data-dependent Python branch: a `Tensor`-valued
    # `norm_exp` (e.g. a fitted parameter) always takes the general branch
    # unchanged, so this never turns into a `.item()`-style sync under
    # `vmap`/`torch.compile`.
    #
    # `torch.special.ndtr` would fold `0.5 * (1.0 + erf(...))` into one
    # call, but its CPU kernel is slower than `erf` (2.07s vs 1.32s on a
    # 188M-element float32 array), so `erf` stays.
    rc = (
        r0
        if isinstance(norm_exp, (float, int)) and norm_exp == 1.0
        else r0**norm_exp
    )
    return 0.5 * (1.0 + torch.erf(-kcn * storch.safe_divide(r - r0, rc)))


def gfn2_count(
    r: Tensor,
    r0: Tensor,
    ka: Tensor | float | int = defaults.KA,
    kb: Tensor | float | int = defaults.KB,
    r_shift: Tensor | float | int = defaults.R_SHIFT,
) -> Tensor:
    """
    Exponential counting function for coordination number contributions as used
    in GFN2-xTB.

    Parameters
    ----------
    r : Tensor
        Internuclear distances.
    r0 : Tensor
        Covalent atomic radii (R_AB = R_A + R_B) or cutoff radius.
    ka : Tensor | float | int, optional
        Steepness of the first counting function. Defaults to
        :data:`tad_mctc.ncoord.defaults.KA`.
    kb : Tensor | float | int, optional
        Steepness of the second counting function. Defaults to
        :data:`tad_mctc.ncoord.defaults.KB`.
    r_shift : Tensor | float | int, optional
        Offset of the second counting function. Defaults to
        :data:`tad_mctc.ncoord.defaults.R_SHIFT`.

    Returns
    -------
    Tensor
        Count of coordination number contribution.
    """
    return exp_count(r, r0, ka) * exp_count(r, r0 + r_shift, kb)
