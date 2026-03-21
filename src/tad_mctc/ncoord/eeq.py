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

import torch

from ..data import en as eneg
from ..data import radii
from ..typing import DD, CountingFunction, Tensor
from . import defaults
from .common import coordination_number, cut_coordination_number
from .count import erf_count

__all__ = ["cn_eeq", "cn_eeq_en", "cut_coordination_number"]


def cn_eeq(
    numbers: Tensor,
    positions: Tensor,
    counting_function: CountingFunction = erf_count,
) -> Tensor:
    """
    Compute fractional coordination number using an exponential counting
    function.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers for all atoms in the system of shape ``(..., nat)``.
    positions : Tensor
        Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
    counting_function : CountingFunction, optional
        Counting function used for the EEQ coordination number.
        Defaults to the error function counting function
        :func:`tad_mctc.ncoord.count.erf_count`.

    Returns
    -------
    Tensor
        Coordination numbers for all atoms (shape: ``(..., nat)``).
    """
    dd: DD = {"device": positions.device, "dtype": positions.dtype}
    cutoff = torch.tensor(defaults.CUTOFF_EEQ, **dd)
    rcov = radii.COV_D3(**dd)[numbers]

    return coordination_number(
        numbers,
        positions,
        counting_function=counting_function,
        rcov=rcov,
        cutoff=cutoff,
        cn_max=defaults.CUTOFF_EEQ_MAX,
    )


def cn_eeq_en(
    numbers: Tensor,
    positions: Tensor,
    counting_function: CountingFunction = erf_count,
) -> Tensor:
    """
    Compute the electronegativity-weighted coordination number using the
    Pauling scale stored in :mod:`tad_mctc.data.en`.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers for all atoms in the system of shape ``(..., nat)``.
    positions : Tensor
        Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
    counting_function : CountingFunction, optional
        Counting function used for the EEQ coordination number.
        Defaults to the error function counting function
        :func:`tad_mctc.ncoord.count.erf_count`.

    Returns
    -------
    Tensor
        Electronegativity-weighted coordination numbers for all atoms
        (shape: ``(..., nat)``).
    """
    dd: DD = {"device": positions.device, "dtype": positions.dtype}

    cutoff = torch.tensor(defaults.CUTOFF_EEQ, **dd)
    rcov = radii.COV_D3(**dd)[numbers]
    en = eneg.PAULING(**dd)[numbers]
    kcn = torch.tensor(defaults.KCN_EEQ_EN, **dd)

    weight = en.unsqueeze(-2) - en.unsqueeze(-1)

    return coordination_number(
        numbers,
        positions,
        counting_function=counting_function,
        rcov=rcov,
        cutoff=cutoff,
        cn_max=None,
        pair_weight=weight,
        kcn=kcn,
    )
