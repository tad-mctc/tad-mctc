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
Coordination number: DFT-D3
===========================

Calculation of coordination number for DFT-D3.
"""

from __future__ import annotations

from functools import partial

from . import defaults
from .common import CNModel
from .count import exp_count

__all__ = ["cn_d3"]


cn_d3 = CNModel(
    count=partial(exp_count, kcn=defaults.KCN_D3), cutoff=defaults.CUTOFF_D3
)
"""
The D3 fractional coordination number: the exponential counting function
with DFT-D3's steepness and cutoff (:mod:`tad_mctc.ncoord.defaults`).
Callable as ``cn_d3(structure)`` for a molecule or a cell.
See :meth:`.CNModel.__call__` for how pairs are enumerated.
"""
