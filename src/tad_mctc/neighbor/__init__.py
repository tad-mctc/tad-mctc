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
Neighbour search
================

Linear-scaling neighbour search, molecular or periodic: atoms are grouped
into spatially bounded tiles, tile pairs within a cutoff are found by an
exact bounding-box screen, and the result is compacted into a
fixed-capacity, padded neighbour list (:mod:`.list`) that is cheap to
consume under autograd, ``vmap`` and ``torch.compile``. Periodic boundary
conditions (:mod:`.images`) replicate atoms into a ghost pool ahead of
that same search, rather than changing it.
"""

from ._distance_kernels import (
    gather_index,
    pair_distance_squared,
    pair_distance_squared_from_columns,
    position_columns,
    split_lattice,
)
from .images import *
from .list import *
from .triples import *
