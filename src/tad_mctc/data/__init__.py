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
Data
====

This module contains various arrays with atomic data:
 - Pauling electronegativities
 - masses
 - covalent radii
 - effective nuclear charges
 - periodic table

Note that the first element of all tensors is a dummy to allow indexing by the
atomic numbers.
"""

from .en import *
from .getters import *
from .hardness import *
from .mass import *
from .pse import *
from .radii import *
from .zeff import *
