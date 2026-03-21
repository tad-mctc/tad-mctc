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

import torch

from ..data import radii
from ..typing import DD, CountingFunction, Tensor
from . import defaults
from .common import coordination_number
from .count import gfn2_count

__all__ = ["cn_gfn2"]


def cn_gfn2(
    numbers: Tensor,
    positions: Tensor,
    counting_function: CountingFunction = gfn2_count,
) -> Tensor:
    """
    Compute the double-exponential (GFN2-xTB) coordination number.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers for all atoms in the system of shape ``(..., nat)``.
    positions : Tensor
        Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
    counting_function : CountingFunction, optional
        Counting function used for the GFN2-xTB coordination number.
        Defaults to the GFN2-xTB counting function
        :func:`tad_mctc.ncoord.count.gfn2_count`.

    Returns
    -------
    Tensor
        Coordination numbers for all atoms (shape: ``(..., nat)``).
    """
    dd: DD = {"device": positions.device, "dtype": positions.dtype}
    cutoff = torch.tensor(defaults.CUTOFF_GFN2, **dd)
    rcov = radii.COV_D3(**dd)[numbers]

    return coordination_number(
        numbers,
        positions,
        counting_function=counting_function,
        rcov=rcov,
        cutoff=cutoff,
    )
