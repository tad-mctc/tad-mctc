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
Molecule: Geometry
==================

Collection of utility functions for calculations related to the geometry of a
molecule. This includes:
- Bond angles
- Linear molecule detection
"""

from __future__ import annotations

import torch

from .. import storch
from ..batch.mask import real_triples
from ..math import einsum
from ..typing import DD, Tensor

__all__ = ["bond_angles", "is_linear"]


def bond_angles(numbers: Tensor, positions: Tensor) -> Tensor:
    """
    Calculate all bond angles. Also works for batched systems.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers for all atoms in the system of shape ``(..., nat)``.
    positions : Tensor
        Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).

    Returns
    -------
    Tensor
        Tensor of bond angles of shape `(..., nat, nat, nat)`.
    """
    dd: DD = {"device": positions.device, "dtype": positions.dtype}

    # masking utility to avoid NaN's
    zero = torch.tensor(0.0, **dd)
    mask = real_triples(numbers, mask_diagonal=True, mask_self=True)

    # Expanding dimensions to compute vectors for all combinations
    p1 = positions.unsqueeze(-2).unsqueeze(-2)  # Shape: [..., N, 1, 1, 3]
    p2 = positions.unsqueeze(-3).unsqueeze(-2)  # Shape: [..., 1, N, 1, 3]
    p3 = positions.unsqueeze(-3).unsqueeze(-3)  # Shape: [..., 1, 1, N, 3]

    vector1 = p1 - p2  # Shape: [..., N, N, 1, 3]
    vector2 = p3 - p2  # Shape: [..., 1, N, N, 3]

    # Compute dot product across the last dimension
    dot_product = einsum("...i,...i->...", vector1, vector2)

    # Compute norms of the vectors
    norm1 = torch.norm(vector1, dim=-1)
    norm2 = torch.norm(vector2, dim=-1)

    # Compute cos(theta) and handle potential numerical issues
    cos_theta = storch.divide(dot_product, norm1 * norm2)
    cos_theta = torch.where(mask, cos_theta, zero)
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)

    # Calculate bond angles in degrees
    deg = torch.rad2deg(torch.acos(cos_theta))
    return torch.where(mask, deg, -zero)


def is_linear(
    numbers: Tensor, positions: Tensor, atol: float = 1e-8, rtol: float = 1e-5
) -> Tensor:
    """
    Check if a molecule is linear based on bond angles.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers for all atoms in the system of shape ``(..., nat)``.
    positions : Tensor
        Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
    atol : float, optional
        Absolute tolerance for the comparison. Defaults to 1e-8.
    rtol : float, optional
        Relative tolerance for the comparison. Defaults to 1e-5.

    Returns
    -------
    Tensor
        Boolean tensor of shape `(..., )` indicating if the molecule is linear.
    """
    angles = bond_angles(numbers, positions)

    # mask for values close to 0 or 180 degrees
    close_to_zero = torch.isclose(
        angles, torch.zeros_like(angles), atol=atol, rtol=rtol
    )
    close_to_180 = torch.isclose(
        angles, torch.full_like(angles, 180.0), atol=atol, rtol=rtol
    )

    # combined mask for values that are NOT close to either 0 or 180 degrees
    not_linear_mask = ~(close_to_zero | close_to_180)

    # Use summation instead of torch.any() to handle batch dimension.
    # Only if the whole mask is False, the molecule is linear.
    return not_linear_mask.sum((-1, -2, -3)) == 0
