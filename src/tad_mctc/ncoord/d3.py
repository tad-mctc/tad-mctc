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

from functools import partial
from typing import Any

import torch

from .. import storch
from ..batch import real_pairs
from ..data import radii
from ..typing import DD, CountingFunction, TableFunction, Tensor
from . import defaults
from .common import CNModel, _resolve_table, _species
from .count import dexp_count, exp_count

__all__ = ["cn_d3", "cn_d3_gradient"]


cn_d3 = CNModel(
    count=partial(exp_count, kcn=defaults.KCN_D3), cutoff=defaults.CUTOFF_D3
)
"""
The D3 fractional coordination number: the exponential counting function
with DFT-D3's steepness and cutoff (:mod:`tad_mctc.ncoord.defaults`).
Callable as ``cn_d3(structure)``: the molecular, all-pairs path when
``structure.lattice is None``, the periodic path (auto-building a shift
table every call) otherwise. ``cn_d3.with_precomputed_shifts(structure,
shifts=...)`` is the ``vmap``/``jacrev``-over-``lattice``-safe periodic
alternative, reusing a precomputed shift table instead of rebuilding one.
"""


def cn_d3_gradient(
    numbers: Tensor,
    positions: Tensor,
    *,
    dcounting_function: CountingFunction = dexp_count,
    rcov: Tensor | TableFunction | None = None,
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
    rcov : Tensor | TableFunction | None, optional
        Covalent radii, per element and indexed by atomic number (entry 0
        is the dummy) — the same convention as :attr:`.CNModel.rcov`.
        Either a :class:`~tad_mctc.typing.TableFunction` such as
        :func:`tad_mctc.data.radii.COV_D3` or a :class:`Tensor`, resolved
        on ``positions``' device and dtype. Defaults to
        :func:`tad_mctc.data.radii.COV_D3`.
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
        If shape mismatch between ``numbers`` and ``positions`` is detected.
    """
    dd: DD = {"device": positions.device, "dtype": positions.dtype}

    if cutoff is None:
        cutoff = torch.tensor(defaults.CUTOFF_D3, **dd)

    # Per-element table, indexed by atomic number — the same convention as
    # `CNModel.rcov`/`_resolve_table` (common.py), resolved here rather than
    # re-implemented so the two never drift apart again. Padding atoms are
    # mapped to element 1 via `_species` before the gather, exactly as
    # `_cn_dense_mol` does, so this analytical gradient agrees with the
    # autograd-derived one on batched/padded input instead of gathering
    # `rcov[0] == 0` and risking a non-finite `r0**norm_exp` term.
    rcov_table = _resolve_table(
        radii.COV_D3 if rcov is None else rcov, positions
    )
    rcov_atoms = rcov_table[_species(numbers)]

    if numbers.shape != positions.shape[:-1]:
        raise ValueError(
            f"Shape of positions ({positions.shape[:-1]}) is not consistent "
            f"with atomic numbers ({numbers.shape})."
        )

    eps = torch.tensor(torch.finfo(positions.dtype).eps, **dd)

    mask = real_pairs(numbers, mask_diagonal=True)
    distances = torch.where(mask, storch.cdist(positions, positions, p=2), eps)

    rc = rcov_atoms.unsqueeze(-2) + rcov_atoms.unsqueeze(-1)
    dcf = torch.where(
        mask * (distances <= cutoff),
        dcounting_function(distances, rc, **kwargs),
        torch.tensor(0.0, **dd),
    )

    # (..., nat, nat, 3)
    rij = positions.unsqueeze(-3) - positions.unsqueeze(-2)

    # (..., nat, nat, 1) * (..., nat, nat, 3)
    return (dcf / distances).unsqueeze(-1) * rij  # "...ij,...ijx->...ijx"
