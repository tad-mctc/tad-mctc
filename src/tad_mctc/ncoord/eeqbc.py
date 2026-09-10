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

import torch

from ..data import en as eneg
from ..data import radii
from ..typing import DD, CountingFunction, Tensor
from . import defaults
from .common import coordination_number
from .count import erf_count

__all__ = ["cn_eeqbc", "cn_eeqbc_en"]


def cn_eeqbc(
    numbers: Tensor,
    positions: Tensor,
    counting_function: CountingFunction = erf_count,
) -> Tensor:
    """
    Compute the coordination number used by the EEQBC charge model.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers for all atoms in the system of shape ``(..., nat)``.
    positions : Tensor
        Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
    counting_function : CountingFunction, optional
        Counting function used for the EEQBC coordination number. Defaults
        to the error function counting function
        :func:`tad_mctc.ncoord.count.erf_count`.

    Returns
    -------
    Tensor
        Coordination numbers for all atoms (shape: ``(..., nat)``).
    """
    dd: DD = {"device": positions.device, "dtype": positions.dtype}
    cutoff = torch.tensor(defaults.CUTOFF_EEQBC, **dd)
    rcov = radii.EEQBC_COV_RADII(**dd)[numbers]
    kcn = torch.tensor(defaults.KCN_EEQBC, **dd)
    norm_exp = torch.tensor(defaults.NORM_EXP_EEQBC, **dd)

    return coordination_number(
        numbers,
        positions,
        counting_function=counting_function,
        rcov=rcov,
        cutoff=cutoff,
        kcn=kcn,
        norm_exp=norm_exp,
    )


def cn_eeqbc_en(
    numbers: Tensor,
    positions: Tensor,
    counting_function: CountingFunction = erf_count,
) -> Tensor:
    """
    Compute the electronegativity-weighted coordination number used by the
    EEQBC charge model to build its local charge contribution.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers for all atoms in the system of shape ``(..., nat)``.
    positions : Tensor
        Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
    counting_function : CountingFunction, optional
        Counting function used for the EEQBC coordination number. Defaults
        to the error function counting function
        :func:`tad_mctc.ncoord.count.erf_count`.

    Returns
    -------
    Tensor
        Electronegativity-weighted coordination numbers for all atoms
        (shape: ``(..., nat)``).
    """
    dd: DD = {"device": positions.device, "dtype": positions.dtype}

    cutoff = torch.tensor(defaults.CUTOFF_EEQBC, **dd)
    rcov = radii.EEQBC_COV_RADII(**dd)[numbers]
    en = eneg.PAULING(**dd)[numbers]
    kcn = torch.tensor(defaults.KCN_EEQBC, **dd)
    norm_exp = torch.tensor(defaults.NORM_EXP_EEQBC, **dd)

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
        norm_exp=norm_exp,
    )
