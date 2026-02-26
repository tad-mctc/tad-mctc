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
Coordination number: Common helpers
===================================

Generic building blocks that mirror the container-based design of mctc-lib.
"""

from __future__ import annotations

import torch

from .. import storch
from ..batch import real_pairs
from ..data import radii
from ..typing import DD, Any, CountingFunction, Tensor
from . import defaults

__all__ = ["coordination_number", "cut_coordination_number"]


def coordination_number(
    numbers: Tensor,
    positions: Tensor,
    *,
    counting_function: CountingFunction,
    rcov: Tensor | None = None,
    cutoff: Tensor | float | int | None = None,
    cn_max: Tensor | float | int | None = None,
    pair_weight: Tensor | None = None,
    **kwargs: Any,
) -> Tensor:
    """
    Generic coordination number evaluator.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers for all atoms in the system (shape: ``(..., nat)``).
    positions : Tensor
        Cartesian coordinates (shape: ``(..., nat, 3)``).
    counting_function : CountingFunction
        Pair counting function (exp, erf, dexp, ...).
    rcov : Tensor | None, optional
        Covalent radii for each species.
    cutoff : Tensor | float | int
        Real-space cutoff.
    cn_max : Tensor | float | int | None, optional
        Optional upper bound for the coordination number.
    pair_weight : Tensor | None, optional
        Optional per-pair weighting factor applied to the counting function.
    kwargs : dict[str, Any]
        Additional keyword arguments forwarded to the counting function.

    Returns
    -------
    Tensor
        Coordination numbers for all atoms (shape: ``(..., nat)``).
    """
    dd: DD = {"device": positions.device, "dtype": positions.dtype}

    if rcov is None:
        rcov = radii.COV_D3(**dd)[numbers]
    else:
        rcov = rcov.to(**dd)

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

    cutoff_tensor = (
        cutoff.to(**dd)
        if isinstance(cutoff, torch.Tensor)
        else torch.tensor(cutoff, **dd)
    )

    mask = real_pairs(numbers, mask_diagonal=True)
    eps = torch.tensor(torch.finfo(positions.dtype).eps, **dd)
    distances = torch.where(mask, storch.cdist(positions, positions, p=2), eps)

    rc = rcov.unsqueeze(-2) + rcov.unsqueeze(-1)
    counts = counting_function(distances, rc, **kwargs)

    if pair_weight is not None:
        counts = pair_weight.to(**dd) * counts

    valid = mask & (distances <= cutoff_tensor)
    zero = torch.tensor(0.0, **dd)
    cf = torch.where(valid, counts, zero)
    cn = torch.sum(cf, dim=-1)

    if cn_max is None:
        return cn

    return cut_coordination_number(cn, cn_max)


def cut_coordination_number(
    cn: Tensor, cn_max: Tensor | float | int = defaults.CUTOFF_EEQ_MAX
) -> Tensor:
    """
    Apply the smooth logarithmic cutoff used throughout mctc projects.

    Parameters
    ----------
    cn : Tensor
        Coordination numbers.
    cn_max : Tensor | float | int, optional
        Maximum coordination number. Large values disable the cutoff.

    Returns
    -------
    Tensor
        Cut coordination numbers.
    """
    if isinstance(cn_max, torch.Tensor):
        cn_max_tensor = cn_max.to(device=cn.device, dtype=cn.dtype)
    else:
        cn_max_tensor = torch.tensor(cn_max, device=cn.device, dtype=cn.dtype)

    if torch.all(cn_max_tensor > 50):
        return cn

    return torch.log1p(torch.exp(cn_max_tensor)) - torch.log1p(
        torch.exp(cn_max_tensor - cn)
    )
