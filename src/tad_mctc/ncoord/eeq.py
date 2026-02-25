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

from .. import storch
from ..batch import real_pairs
from ..data import radii
from ..typing import DD, Any, CountingFunction, Tensor
from . import defaults
from .count import erf_count

__all__ = ["cn_eeq"]


def cn_eeq(
    numbers: Tensor,
    positions: Tensor,
    *,
    counting_function: CountingFunction | None = None,
    rcov: Tensor | None = None,
    cutoff: Tensor | float | int | None = defaults.CUTOFF_EEQ,
    cn_max: Tensor | float | int | None = defaults.CUTOFF_EEQ_MAX,
    **kwargs: Any,
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
    counting_function : CountingFunction
        Calculate weight for pairs. Defaults to
        :func:`tad_mctc.ncoord.count.erf_count`.
    rcov : Tensor | None, optional
        Covalent radii for each species. Defaults to ``None``.
    cutoff : Tensor | float | int | None, optional
        Real-space cutoff. Defaults to
        :data:`tad_mctc.ncoord.defaults.CUTOFF_EEQ`.
    cn_max : Tensor | float | int | None, optional
        Maximum coordination number. Defaults to
        :data:`tad_mctc.ncoord.defaults.CUTOFF_EEQ_MAX`.
    kwargs : dict[str, Any]
        Pass-through arguments for counting function. For example, ``kcn``,
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
        cutoff = torch.tensor(defaults.CUTOFF_EEQ, **dd)

    if rcov is None:
        rcov = radii.COV_D3(**dd)[numbers]
    else:
        rcov = rcov.to(**dd)

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

    rc = rcov.unsqueeze(-2) + rcov.unsqueeze(-1)
    cf = torch.where(
        mask * (distances <= cutoff),
        counting_function(distances, rc, **kwargs),
        torch.tensor(0.0, **dd),
    )
    cn = torch.sum(cf, dim=-1)

    if cn_max is None:
        return cn

    return cut_coordination_number(cn, cn_max)


def cut_coordination_number(
    cn: Tensor, cn_max: Tensor | float | int = defaults.CUTOFF_EEQ_MAX
) -> Tensor:
    """
    Cut the coordination number at a maximum value.

    Parameters
    ----------
    cn : Tensor
        Coordination numbers.
    cn_max : Tensor | float | int, optional
        Maximum coordination number.
        Defaults to :data:`tad_mctc.ncoord.defaults.CUTOFF_EEQ_MAX`.

    Returns
    -------
    Tensor
        Cut coordination numbers.
    """
    if isinstance(cn_max, (float, int)):
        cn_max = torch.tensor(cn_max, device=cn.device, dtype=cn.dtype)

    if cn_max > 50:
        return cn

    return torch.log(1.0 + torch.exp(cn_max)) - torch.log(
        1.0 + torch.exp(cn_max - cn)
    )
