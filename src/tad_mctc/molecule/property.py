# This file is part of tad-mctc.
#
# SPDX-Identifier: LGPL-3.0
# Copyright (C) 2023 Marvin Friede
#
# tad-mctc is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# tad_mctc is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with tad-mctc. If not, see <https://www.gnu.org/licenses/>.
"""
Molecule: Property
==================

Collection of functions for the calculation of molecular properties.
"""

from __future__ import annotations

import torch

from .. import storch
from ..math import einsum
from ..typing import Tensor

__all__ = ["mass_center"]


def mass_center(masses: Tensor, positions: Tensor) -> Tensor:
    """
    Calculate the center of mass from the atomic coordinates and masses.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers for all atoms in the system of shape `(..., nat)`.
    positions : Tensor
        Cartesian coordinates of all atoms of shape `(..., nat, 3)`.

    Returns
    -------
    Tensor
        Cartesian coordinates of center of mass of shape `(..., 3)`.
    """
    s = storch.reciprocal(torch.sum(masses, dim=-1))
    return einsum("...z,...zx,...->...x", masses, positions, s)
