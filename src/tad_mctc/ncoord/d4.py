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
Coordination number: DFT-D4
===========================

Calculation of DFT-D4 coordination number. Includes electronegativity-
dependent term.
"""

from __future__ import annotations

from functools import partial

import torch

from ..typing import Tensor
from . import defaults
from .common import CNModel
from .count import erf_count

__all__ = ["cn_d4", "d4_en_weight"]


def d4_en_weight(
    en_i: Tensor,
    en_j: Tensor,
    k4: Tensor | float | int = defaults.D4_K4,
    k5: Tensor | float | int = defaults.D4_K5,
    k6: Tensor | float | int = defaults.D4_K6,
) -> Tensor:
    """
    Electronegativity pair weight used by the DFT-D4 coordination number.

    Parameters
    ----------
    en_i, en_j : Tensor
        Pauling electronegativities of the two atoms in a pair.
    k4, k5, k6 : Tensor | float | int, optional
        Parameters of the electronegativity scaling. Default to
        :data:`tad_mctc.ncoord.defaults.D4_K4`,
        :data:`tad_mctc.ncoord.defaults.D4_K5` and
        :data:`tad_mctc.ncoord.defaults.D4_K6`.

    Returns
    -------
    Tensor
        Elementwise pair weight, broadcastable over pairs.
    """
    en_diff = torch.abs(en_i - en_j)
    return k4 * torch.exp(-((en_diff + k5) ** 2) / k6)


cn_d4 = CNModel(
    count=partial(erf_count, kcn=defaults.KCN_D4),
    cutoff=defaults.CUTOFF_D4,
    pair_weight=d4_en_weight,
)
"""
The D4 fractional coordination number: the error-function counting
function, DFT-D4's steepness and cutoff, and the electronegativity pair
weight :func:`d4_en_weight` (:mod:`tad_mctc.ncoord.defaults`). Callable as
``cn_d4(structure)``: the molecular, all-pairs path when
``structure.lattice is None``, the periodic path (auto-building a shift
table every call) otherwise. ``cn_d4.with_precomputed_shifts(structure,
shifts=...)`` is the ``vmap``/``jacrev``-over-``lattice``-safe periodic
alternative, reusing a precomputed shift table instead of rebuilding one.
"""
