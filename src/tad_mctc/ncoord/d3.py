# This file is part of tad_mctc.
#
# SPDX-Identifier: LGPL-3.0
# Copyright (C) 2023 Marvin Friede
#
# tad_mctc is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# tad_mctc is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with tad_mctc. If not, see <https://www.gnu.org/licenses/>.
"""
Coordination number: EEQ
========================

Calculation of coordination number for the EEQ model.
"""
from __future__ import annotations

import torch

from .. import storch
from .._typing import Any, CountingFunction, Tensor
from ..batch import real_pairs
from ..data import radii
from . import defaults
from .count import exp_count

__all__ = ["cn_d3"]


def cn_d3(
    numbers: Tensor,
    positions: Tensor,
    counting_function: CountingFunction = exp_count,
    rcov: Tensor | None = None,
    cutoff: Tensor | None = None,
    kcn: float = defaults.KCN_D3,
    **kwargs: Any,
) -> Tensor:
    """
    Compute the D3 fractional coordination (exponential counting function).

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers for all atoms in the system.
    positions : Tensor
        Atomic positions of molecular structure.
    counting_function : CountingFunction
        Calculate weight for pairs.
    rcov : Tensor | None, optional
        Covalent radii for each species. Defaults to `None`.
    cutoff : Tensor | None, optional
        Real-space cutoff. Defaults to `None`.
    kcn : float, optional
        Steepness of the counting function.
    kwargs : dict[str, Any]
        Pass-through arguments for counting function.

    Returns
    -------
    Tensor
        Coordination numbers for all atoms.

    Raises
    ------
    ValueError
        If shape mismatch between `numbers`, `positions` and `rcov` is detected.
    """
    dd = {"device": positions.device, "dtype": positions.dtype}

    if cutoff is None:
        cutoff = torch.tensor(defaults.CUTOFF_D3, **dd)

    if rcov is None:
        rcov = radii.COV_D3.to(**dd)[numbers]
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

    mask = real_pairs(numbers, diagonal=True)
    distances = torch.where(mask, storch.cdist(positions, positions, p=2), eps)

    rc = rcov.unsqueeze(-2) + rcov.unsqueeze(-1)
    cf = torch.where(
        mask * (distances <= cutoff),
        counting_function(distances, rc, kcn, **kwargs),
        torch.tensor(0.0, **dd),
    )
    return torch.sum(cf, dim=-1)
