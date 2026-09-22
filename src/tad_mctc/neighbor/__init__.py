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

Periodic-image geometry (:mod:`.images`): integer lattice-translation
shifts and ghost-pool replication for a real-space cutoff sphere, and
folding positions back into the primary cell. Consumed directly by the
dense (all-pairs) periodic coordination-number path in
:mod:`tad_mctc.ncoord.common`.
"""

from .images import *
