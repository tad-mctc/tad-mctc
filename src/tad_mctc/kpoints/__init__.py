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
k-points
========

Sampling points of the Brillouin zone of a periodic structure: the
:class:`KPoints` container (:mod:`.points`) and uniform grids
(:mod:`.grid`).

k-points are Cartesian, in 1/Bohr, as in PySCF, with the Bloch phase
:math:`e^{i \\mathbf{k} \\cdot \\mathbf{T}}` of a lattice translation
:math:`\\mathbf{T}`. Their fractional (reduced) coordinates
:math:`\\boldsymbol{\\kappa}` refer to the reciprocal lattice vectors,
:math:`\\mathbf{k} = \\boldsymbol{\\kappa} \\mathbf{B}` with
:math:`\\mathbf{B} = 2\\pi \\mathbf{A}^{-T}` for the lattice vectors
:math:`\\mathbf{A}` as rows.
"""

from .grid import *
from .points import *
