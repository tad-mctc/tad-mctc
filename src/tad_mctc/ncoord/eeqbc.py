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
Coordination number: EEQBC
===========================

Calculation of the coordination number for the EEQBC (EEQ bond-capacitor)
charge model of Froitzheim, Müller, Hansen and Grimme, J. Chem. Phys. 2025,
162, 214109 (DOI: 10.1063/5.0268978), as implemented in ``multicharge``.

The counting function is the same error function used for D4 and EEQ,
generalized by an extra exponent on the covalent-radii normalization (see
:func:`tad_mctc.ncoord.count.erf_count`'s ``norm_exp`` argument), evaluated
with EEQBC's own covalent radii (:func:`tad_mctc.data.radii.EEQBC_COV_RADII`)
and steepness/normalization parameters. Unlike :func:`tad_mctc.ncoord.eeq.
cn_eeq`, the EEQBC coordination number is not capped.
"""

from __future__ import annotations

from functools import partial

from ..data import radii
from . import defaults
from .common import CNModel
from .count import erf_count
from .eeq import en_difference

__all__ = ["cn_eeqbc", "cn_eeqbc_en"]


cn_eeqbc = CNModel(
    count=partial(
        erf_count, kcn=defaults.KCN_EEQBC, norm_exp=defaults.NORM_EXP_EEQBC
    ),
    cutoff=defaults.CUTOFF_EEQBC,
    rcov=radii.EEQBC_COV_RADII,
)
"""
The coordination number used by the EEQBC charge model: the error-function
counting function generalized by EEQBC's ``norm_exp``
(:data:`defaults.NORM_EXP_EEQBC`), EEQBC's own covalent radii
(:func:`tad_mctc.data.radii.EEQBC_COV_RADII`) and steepness
(:data:`defaults.KCN_EEQBC`), and no CN cap. Callable as
``cn_eeqbc(structure)``: the molecular, all-pairs path when
``structure.lattice is None``, the periodic path (auto-building a shift
table every call) otherwise. ``cn_eeqbc.with_precomputed_shifts(structure,
shifts=...)`` is the ``vmap``/``jacrev``-over-``lattice``-safe periodic
alternative, reusing a precomputed shift table instead of rebuilding one.
"""

cn_eeqbc_en = CNModel(
    count=partial(
        erf_count, kcn=defaults.KCN_EEQBC, norm_exp=defaults.NORM_EXP_EEQBC
    ),
    cutoff=defaults.CUTOFF_EEQBC,
    rcov=radii.EEQBC_COV_RADII,
    pair_weight=en_difference,
)
"""
The electronegativity-weighted coordination number used by the EEQBC charge
model to build its local charge contribution: same counting function as
:data:`cn_eeqbc`, weighted by the antisymmetric electronegativity pair
weight :func:`tad_mctc.ncoord.eeq.en_difference`.
"""
