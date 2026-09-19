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
I/O Read: Cell Parameters (internal)
=====================================

mctc-lib's ``cell_to_dlat``/``cell_to_dvol`` (duplicated verbatim across
``src/mctc/io/read/turbomole.f90`` and ``src/mctc/io/read/cjson.F90``):
build triclinic lattice vectors from the 6 standard cell parameters
(lengths ``a``/``b``/``c``, angles ``alpha``/``beta``/``gamma``). Not part
of the public API; each reader that needs it imports it directly.
"""

from __future__ import annotations

import math

import torch

from ...typing import DD, Tensor

__all__: list[str] = []


def cell_to_lattice(
    cellpar: tuple[float, float, float, float, float, float],
    dd: DD,
) -> Tensor:
    """
    Build lattice vectors (as rows) from cell parameters, following
    mctc-lib's ``cell_to_dlat``/``cell_to_dvol``. Angles are given in
    radians; lengths are already in the desired unit (bohr).
    """
    a, b, c, alpha, beta, gamma = cellpar
    cos_a, cos_b, cos_g = math.cos(alpha), math.cos(beta), math.cos(gamma)
    sin_g = math.sin(gamma)

    vol2 = 1.0 - cos_a**2 - cos_b**2 - cos_g**2 + 2.0 * cos_a * cos_b * cos_g
    dvol = math.sqrt(abs(vol2)) * a * b * c
    if vol2 < 0.0:
        # mctc-lib's own comment: "this should not happen, but who knows"
        dvol = -dvol

    v1 = (a, 0.0, 0.0)
    v2 = (b * cos_g, b * sin_g, 0.0)
    v3 = (
        c * cos_b,
        c * (cos_a - cos_b * cos_g) / sin_g,
        dvol / (a * b * sin_g),
    )
    return torch.tensor([v1, v2, v3], **dd)
