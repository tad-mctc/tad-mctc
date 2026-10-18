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
Neighbour search: Wigner-Seitz cell
===================================

The nearest periodic image(s) of every pair of atoms, i.e., the images
inside the Wigner-Seitz cell around the first atom of the pair. A lattice
sum truncated at a fixed number of cells (as the Ewald sums of
multicharge and tblite) is centred on these images, so that the
truncation is the same for every pair. An image on the boundary of the
Wigner-Seitz cell is as near as another one, and all of them are averaged
with equal weights.

Port of ``new_wignerseitz_cell`` of tblite (``src/tblite/wignerseitz.f90``).
The image search is discrete: the weights are piecewise constant in the
positions and the lattice and carry no gradient, as in the Fortran code.

Example
-------
>>> import torch
>>> from tad_mctc.neighbor.wignerseitz import wignerseitz_cell
>>>
>>> lattice = 4.0 * torch.eye(3, dtype=torch.double)
>>> periodic = torch.tensor([True, True, True])
>>> positions = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
>>> wsc = wignerseitz_cell(positions.double(), lattice, periodic)
>>> wsc.shifts[wsc.weights[0, 1] > 0]
tensor([[ 0,  0,  0],
        [-1,  0,  0]])
>>> (wsc.weights[0, 0] > 0).sum()  # six equally near images of the atom
tensor(6)
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch

from ..typing import Tensor
from .images import lattice_shifts_mctclib

__all__ = ["WignerSeitzCell", "wignerseitz_cell"]


_THRESHOLD = math.sqrt(torch.finfo(torch.float64).eps)
"""Squared distance below which an image is the atom itself, and not a
neighbour (``thr`` of tblite's ``wignerseitz.f90``)."""

_TOLERANCE = 0.01
"""Squared-distance tolerance within which two images are equally near
(``tol`` of tblite's ``wignerseitz.f90``)."""


class WignerSeitzCell(NamedTuple):
    """The nearest periodic images of all pairs of atoms."""

    shifts: Tensor
    """Integer lattice shifts searched, shape ``(27, 3)``, ``torch.long``,
    in the order of :func:`.lattice_shifts_mctclib`."""

    weights: Tensor
    """Weight of each shift for pair ``(i, j)``, shape
    ``(..., nat, nat, 27)``: ``1 / n`` for each of the ``n`` nearest
    images ``x_j - x_i + shift @ lattice``, zero for the others."""


@torch.no_grad()
def wignerseitz_cell(
    positions: Tensor, lattice: Tensor, periodic: Tensor
) -> WignerSeitzCell:
    """
    Find the nearest periodic images of every pair of atoms.

    The images are searched among the cell and its neighbouring cells,
    shifts ``-1`` to ``1`` along every axis, so ``positions`` should lie
    in or near the cell (see
    :func:`~tad_mctc.neighbor.images.wrap_to_central_cell`). An atom is not
    its own neighbour, but its images at a non-zero shift are. A shift
    along an axis that is not periodic is never an image.

    Padding atoms are treated as any other atom, the consumer masks their
    pairs.

    Parameters
    ----------
    positions : Tensor
        Cartesian coordinates, shape ``(..., nat, 3)``.
    lattice : Tensor
        Lattice vectors as rows, shape ``(..., 3, 3)``, in Bohr.
    periodic : Tensor
        Boolean mask, shape ``(3,)`` or ``(..., 3)``, which axes are
        periodic.

    Returns
    -------
    WignerSeitzCell
        The shifts searched and the weight of each for every pair. A pair
        without an image (an atom with itself in a molecule) has zero
        weights.
    """
    shifts = lattice_shifts_mctclib((1, 1, 1), device=positions.device)

    # (..., 27, 3)
    translations = shifts.to(positions.dtype) @ lattice

    # entry `(i, j, k)` points from atom `i` to image `k` of atom `j`
    vectors = positions[..., None, :, :] - positions[..., :, None, :]
    images = vectors[..., None, :] + translations[..., None, None, :, :]
    distance_squared = (images * images).sum(-1)  # (..., nat, nat, 27)

    # A shift along an axis that is not periodic does not exist.
    open_axis = (shifts != 0) & ~periodic[..., None, :]  # (..., 27, 3)
    exists = ~open_axis.any(-1)[..., None, None, :]  # (..., 1, 1, 27)

    candidate = exists & (distance_squared >= _THRESHOLD)
    nearest = torch.where(candidate, distance_squared, math.inf).amin(
        -1, keepdim=True
    )
    selected = candidate & (distance_squared - nearest <= _TOLERANCE)

    count = selected.sum(-1, keepdim=True).clamp(min=1)
    weights = selected.to(positions.dtype) / count
    return WignerSeitzCell(shifts=shifts, weights=weights)
