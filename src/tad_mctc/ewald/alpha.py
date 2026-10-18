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
Ewald: Splitting parameter
==========================

The splitting parameter :math:`\\alpha` of a 3D Ewald summation, chosen
such that the real-space and reciprocal-space terms decline equally fast
(:func:`get_alpha`).

Port of ``get_alpha`` and ``search_alpha`` of multicharge
(``src/multicharge/ewald.f90``), which are the same as tblite's
(``src/tblite/coulomb/ewald.f90``). The search is a bisection, so the
result is a step function of the lattice and carries no gradient. A
converged Ewald sum does not depend on :math:`\\alpha`, and the Fortran
derivatives treat it as a constant as well.

Example
-------
>>> import torch
>>> from tad_mctc.ewald import get_alpha
>>>
>>> lattice = 10.0 * torch.eye(3, dtype=torch.double)
>>> get_alpha(lattice)
tensor(0.2936, dtype=torch.float64)
"""

from __future__ import annotations

import math

import torch

from ..kpoints import reciprocal_lattice
from ..typing import Tensor

__all__ = ["get_alpha"]


_ALPHA0 = 1.0e-8
"""Start of the search."""

_DOUBLINGS = 64
"""Doublings of :data:`_ALPHA0` searched for the bracket. The Fortran code
doubles until overflow; 64 doublings reach about ``1e11``, i.e., a cell
of about ``1e-10`` Bohr."""

_BISECTIONS = 30
"""Bisection steps (``niter`` of the Fortran code), see :func:`get_alpha`."""

_TOLERANCE = math.sqrt(torch.finfo(torch.float64).eps)
"""Tolerance on the difference of the two declines."""

_FALLBACK = 0.25
"""Splitting parameter if the search fails."""


def _real_term(r: Tensor, alpha: Tensor) -> Tensor:
    """Largest real-space term at distance ``r``"""
    return torch.erfc(alpha * r) / r


def _reciprocal_term(g: Tensor, alpha: Tensor, volume: Tensor) -> Tensor:
    """Largest reciprocal-space term at reciprocal vector length ``g``"""
    return (
        4.0 * math.pi * (torch.exp(-0.25 * g * g / alpha**2) / (volume * g * g))
    )


def _decline_difference(
    alpha: Tensor, rlen: Tensor, dlen: Tensor, volume: Tensor
) -> Tensor:
    """
    Difference of the decline of the reciprocal-space terms (from four to
    five times the shortest reciprocal vector) and the real-space terms
    (from two to three times the shortest lattice vector). The real-space
    sum is the shorter one.
    """
    rec = _reciprocal_term(4 * rlen, alpha, volume) - _reciprocal_term(
        5 * rlen, alpha, volume
    )
    real = _real_term(2 * dlen, alpha) - _real_term(3 * dlen, alpha)
    return rec - real


def _first(condition: Tensor) -> Tensor:
    """Index of the first ``True`` along the last axis, its size if none"""
    size = condition.shape[-1]
    index = torch.arange(size, device=condition.device)
    return torch.where(condition, index, size).amin(-1)


@torch.no_grad()
def get_alpha(lattice: Tensor) -> Tensor:
    """
    Splitting parameter of a 3D Ewald summation for the lattice.

    The search doubles :math:`\\alpha` from ``1e-8`` until the declines of
    the real-space and reciprocal-space terms (see
    :func:`_decline_difference`) bracket zero, and then bisects the
    bracket to a tolerance of :math:`\\sqrt{\\epsilon}`. As in the Fortran
    code, a search that does not converge within 30 bisection steps, or
    that has no bracket, returns ``0.25``.

    The search runs in double precision whatever the dtype of ``lattice``,
    with a fixed number of steps, so that it works under ``vmap`` and
    ``torch.compile`` and on a batch of lattices.

    Parameters
    ----------
    lattice : Tensor
        Lattice vectors as rows, shape ``(..., 3, 3)``, in Bohr.

    Returns
    -------
    Tensor
        Splitting parameter, shape ``(...)``, in 1/Bohr, with the dtype of
        ``lattice``.
    """
    lat = lattice.detach().to(torch.float64)
    volume = torch.linalg.det(lat).abs()
    rlen = reciprocal_lattice(lat).norm(dim=-1).amin(-1)
    dlen = lat.norm(dim=-1).amin(-1)

    def decline_difference(alpha: Tensor) -> Tensor:
        return _decline_difference(alpha, rlen, dlen, volume)

    # The doublings are exact, so all of them are evaluated at once: the
    # `k`-th is `_ALPHA0 * 2**k`, as after `k` steps of the Fortran loop.
    steps = torch.arange(_DOUBLINGS, dtype=torch.float64, device=lat.device)
    ladder = _ALPHA0 * 2.0**steps  # (n,)
    diffs = _decline_difference(
        ladder, rlen[..., None], dlen[..., None], volume[..., None]
    )  # (..., n)

    # First loop: double while the real-space terms decline faster.
    lower = _first(diffs >= -_TOLERANCE)

    # Second loop: continue doubling until the reciprocal-space terms
    # decline faster.
    upper = _first((diffs >= _TOLERANCE) & (steps >= lower[..., None]))

    # No doubling at all, or overflow (`stat` 2, 1 and 3 in Fortran)
    failed = (lower == 0) | (lower == _DOUBLINGS) | (upper == _DOUBLINGS)

    def at(index: Tensor) -> Tensor:
        # the same exact power of two as `ladder`, without indexing it by
        # a tensor (not batchable under `vmap`)
        step = index.clamp(0, _DOUBLINGS - 1).to(torch.float64)
        return _ALPHA0 * 2.0**step

    alpl = at(lower - 1)
    alpr = at(upper)
    alpha = (alpl + alpr) * 0.5
    diff = decline_difference(alpha)

    # Bisection, frozen once converged. The Fortran loop runs while
    # `ibs <= niter`, i.e., up to `niter + 1` steps, and fails (`stat` 4)
    # if it takes all of them.
    taken = torch.zeros_like(alpha, dtype=torch.long)
    for _ in range(_BISECTIONS + 1):
        active = diff.abs() > _TOLERANCE
        alpl = torch.where(active & (diff < 0), alpha, alpl)
        alpr = torch.where(active & (diff >= 0), alpha, alpr)
        alpha = torch.where(active, (alpl + alpr) * 0.5, alpha)
        diff = torch.where(active, decline_difference(alpha), diff)
        taken = taken + active.long()
    failed = failed | (taken > _BISECTIONS)

    alpha = torch.where(failed, _FALLBACK, alpha)
    return alpha.to(lattice.dtype)
