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
Coordination number: EEQ
========================

Calculation of coordination number for the EEQ model.
"""

from __future__ import annotations

from functools import partial

from ..typing import Tensor
from . import defaults
from .common import CNModel, cut_coordination_number
from .count import erf_count

__all__ = ["cn_eeq", "cn_eeq_en", "cut_coordination_number", "en_difference"]


def en_difference(en_i: Tensor, en_j: Tensor) -> Tensor:
    """
    Antisymmetric electronegativity pair weight used by the EN-weighted EEQ
    and EEQBC coordination numbers.

    Parameters
    ----------
    en_i, en_j : Tensor
        Pauling electronegativities of the two atoms in a pair.

    Returns
    -------
    Tensor
        ``en_j - en_i``: the weight on atom ``j``'s count as it is added to
        ``cn[i]``.
    """
    return en_j - en_i


cn_eeq = CNModel(
    count=partial(erf_count, kcn=defaults.KCN_EEQ),
    cutoff=defaults.CUTOFF_EEQ,
    cn_max=defaults.CUTOFF_EEQ_MAX,
)
"""
The EEQ coordination number: the error-function counting function with
EEQ's steepness, cutoff and CN cap (:mod:`tad_mctc.ncoord.defaults`).
Callable as ``cn_eeq(structure)``: the molecular, all-pairs path when
``structure.lattice is None``, the periodic path (auto-building a shift
table every call) otherwise. ``cn_eeq.with_precomputed_shifts(structure,
shifts=...)`` is the ``vmap``/``jacrev``-over-``lattice``-safe periodic
alternative, reusing a precomputed shift table instead of rebuilding one.
"""

cn_eeq_en = CNModel(
    count=partial(erf_count, kcn=defaults.KCN_EEQ_EN),
    cutoff=defaults.CUTOFF_EEQ,
    pair_weight=en_difference,
)
"""
The electronegativity-weighted EEQ coordination number: same steepness
family as :data:`cn_eeq` but with :data:`defaults.KCN_EEQ_EN`, no CN cap,
and the antisymmetric electronegativity pair weight :func:`en_difference`.
"""
