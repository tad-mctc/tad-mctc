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
Coordination number: GFN2-xTB
=============================

Calculation of the double-exponential coordination number used in GFN2-xTB.
"""

from __future__ import annotations

from . import defaults
from .common import CNModel
from .count import gfn2_count

__all__ = ["cn_gfn2"]


cn_gfn2 = CNModel(count=gfn2_count, cutoff=defaults.CUTOFF_GFN2)
"""
The double-exponential (GFN2-xTB) coordination number
(:mod:`tad_mctc.ncoord.defaults`). Callable as ``cn_gfn2(structure)``:
the molecular, all-pairs path when ``structure.lattice is None``, the
periodic path (auto-building a shift table every call) otherwise.
``cn_gfn2.with_precomputed_shifts(structure, shifts=...)`` is the
``vmap``/``jacrev``-over-``lattice``-safe periodic alternative, reusing a
precomputed shift table instead of rebuilding one.
"""
