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
Coordination number: DFT-D4
===========================

Calculation of DFT-D4 coordination number. Includes electronegativity-
dependent term.
"""

from __future__ import annotations

import torch

from .. import storch
from ..batch import real_pairs
from ..data import en as eneg
from ..data import radii
from ..typing import DD, Any, CountingFunction, Tensor
from . import defaults
from .count import erf_count

__all__ = ["cn_d4"]


def cn_d4(
    numbers: Tensor,
    positions: Tensor,
    *,
    counting_function: CountingFunction | None = None,
    rcov: Tensor | None = None,
    en: Tensor | None = None,
    cutoff: Tensor | None = None,
    **kwargs: Any,
) -> Tensor:
    """
    Compute the D4 fractional coordination number.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers for all atoms in the system of shape ``(..., nat)``.
    positions : Tensor
        Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
    counting_function : CountingFunction
        Calculate weight for pairs. Defaults to
        :func:`tad_mctc.ncoord.count.erf_count`.
    rcov : Tensor | None, optional
        Covalent radii for each species. Defaults to ``None``.
    en : Tensor | None, optional
        Electronegativities for all atoms. Defaults to ``None``.
    cutoff : Tensor | None, optional
        Real-space cutoff. Defaults to ``None``.
    kwargs : dict[str, Any]
        Pass-through arguments for counting function. For example, ``kcn``.
        the steepness of the counting function.

    Returns
    -------
    Tensor
        Coordination numbers for all atoms (shape: ``(..., nat)``).

    Raises
    ------
    ValueError
        If shape mismatch between ``numbers``, ``positions`` and
        ``rcov`` is detected.
    """
    dd: DD = {"device": positions.device, "dtype": positions.dtype}

    if cutoff is None:
        cutoff = torch.tensor(defaults.CUTOFF_D4, **dd)

    if rcov is None:
        rcov = radii.COV_D3(**dd)[numbers]
    else:
        rcov = rcov.to(**dd)

    if en is None:
        en = eneg.PAULING(**dd)[numbers]
    else:
        en = en.to(**dd)

    if counting_function is None:
        counting_function = erf_count

    if numbers.shape != rcov.shape:
        raise ValueError(
            f"Shape of covalent radii {rcov.shape} is not consistent with "
            f"({numbers.shape})."
        )
    if numbers.shape != positions.shape[:-1]:
        raise ValueError(
            f"Shape of positions ({positions.shape[:-1]}) is not consistent "
            f"with atomic numbers ({numbers.shape})."
        )

    eps = torch.tensor(torch.finfo(positions.dtype).eps, **dd)

    mask = real_pairs(numbers, mask_diagonal=True)
    distances = torch.where(mask, storch.cdist(positions, positions, p=2), eps)

    # Eq. 6
    endiff = torch.abs(en.unsqueeze(-2) - en.unsqueeze(-1))
    den = defaults.D4_K4 * torch.exp(
        -((endiff + defaults.D4_K5) ** 2.0) / defaults.D4_K6
    )

    rc = rcov.unsqueeze(-2) + rcov.unsqueeze(-1)
    cf = torch.where(
        mask * (distances <= cutoff),
        den * counting_function(distances, rc, **kwargs),
        torch.tensor(0.0, **dd),
    )
    return torch.sum(cf, dim=-1)
