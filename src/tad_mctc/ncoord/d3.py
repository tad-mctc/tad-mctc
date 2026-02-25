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
Coordination number: DFT-D3
===========================

Calculation of coordination number for DFT-D3.
"""

from __future__ import annotations

import torch

from .. import storch
from ..batch import real_pairs
from ..data import radii
from ..typing import DD, Any, CountingFunction, Tensor
from . import defaults
from .common import coordination_number
from .count import dexp_count, exp_count

__all__ = ["cn_d3", "cn_d3_gradient"]


def cn_d3(numbers: Tensor, positions: Tensor) -> Tensor:
    """
    Compute the D3 fractional coordination number using the default parameters
    described in :mod:`tad_mctc.ncoord.defaults`.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers for all atoms in the system of shape ``(..., nat)``.
    positions : Tensor
        Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).

    Returns
    -------
    Tensor
        Coordination numbers for all atoms (shape: ``(..., nat)``).
    """
    dd: DD = {"device": positions.device, "dtype": positions.dtype}
    cutoff = torch.tensor(defaults.CUTOFF_D3, **dd)
    rcov = radii.COV_D3(**dd)[numbers]

    return coordination_number(
        numbers,
        positions,
        counting_function=exp_count,
        rcov=rcov,
        cutoff=cutoff,
    )


def cn_d3_gradient(
    numbers: Tensor,
    positions: Tensor,
    *,
    dcounting_function: CountingFunction = dexp_count,
    rcov: Tensor | None = None,
    cutoff: Tensor | None = None,
    **kwargs: Any,
) -> Tensor:
    """
    Compute the derivative of the fractional coordination number with respect
    to atomic positions.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers for all atoms in the system of shape ``(..., nat)``.
    positions : Tensor
        Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
    dcounting_function : CountingFunction, optional
        Derivative of the counting function. Defaults to
        :func:`tad_mctc.ncoord.count.dexp_count`.
    rcov : Tensor | None, optional
        Covalent radii for each species. Defaults to ``None``.
    cutoff : Tensor | None, optional
        Real-space cutoff. Defaults to ``None``.
    kwargs : dict[str, Any]
        Pass-through arguments for counting function. For example, ``kcn``,
        the steepness of the counting function, which defaults to
        :data:`tad_mctc.ncoord.defaults.KCN_D3`.

    Returns
    -------
    Tensor
        Coordination numbers for all atoms (shape: ``(..., nat, nat, 3)``).

    Raises
    ------
    ValueError
        If shape mismatch between ``numbers``, ``positions`` and
        ``rcov`` is detected.
    """
    dd: DD = {"device": positions.device, "dtype": positions.dtype}

    if cutoff is None:
        cutoff = torch.tensor(defaults.CUTOFF_D3, **dd)

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

    eps = torch.tensor(torch.finfo(positions.dtype).eps, **dd)

    mask = real_pairs(numbers, mask_diagonal=True)
    distances = torch.where(mask, storch.cdist(positions, positions, p=2), eps)

    rc = rcov.unsqueeze(-2) + rcov.unsqueeze(-1)
    dcf = torch.where(
        mask * (distances <= cutoff),
        dcounting_function(distances, rc, **kwargs),
        torch.tensor(0.0, **dd),
    )

    # (..., nat, nat, 3)
    rij = positions.unsqueeze(-3) - positions.unsqueeze(-2)

    # (..., nat, nat, 1) * (..., nat, nat, 3)
    return (dcf / distances).unsqueeze(-1) * rij  # "...ij,...ijx->...ijx"
