# This file is part of tad-multicharge.
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
Batch Utility: Masks
====================

Functions for creating masks that discern between padding and actual values.
"""
from __future__ import annotations

import torch

from ..typing import Tensor

__all__ = ["real_atoms", "real_pairs", "real_triples"]


def real_atoms(numbers: Tensor) -> Tensor:
    """
    Create a mask for atoms, discerning padding and actual atoms.
    Padding value is zero.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers for all atoms.

    Returns
    -------
    Tensor
        Mask for atoms that discerns padding and real atoms.
    """
    return numbers != 0


def real_pairs(numbers: Tensor, mask_diagonal: bool = True) -> Tensor:
    """
    Create a mask for pairs of atoms from atomic numbers, discerning padding
    and actual atoms. Padding value is zero.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers for all atoms.
    mask_diagonal : bool, optional
        Flag for also masking the diagonal, i.e., all pairs with the same
        indices. Defaults to `True`, i.e., writing False to the diagonal.

    Returns
    -------
    Tensor
        Mask for atom pairs that discerns padding and real atoms.
    """
    real = real_atoms(numbers)
    mask = real.unsqueeze(-2) * real.unsqueeze(-1)

    if mask_diagonal is True:
        mask *= ~torch.diag_embed(torch.ones_like(real))
    return mask


def real_triples(
    numbers: Tensor, mask_diagonal: bool = True, mask_self: bool = True
) -> Tensor:
    """
    Create a mask for triples from atomic numbers. Padding value is zero.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers for all atoms.
    mask_diagonal : bool, optional
        Flag for also masking the diagonal, i.e., all pairs with the same
        indices. Defaults to `True`, i.e., writing False to the diagonal.
    mask_self : bool, optional
        Flag for also masking all triples where at least two indices are
        identical. Defaults to `True`, i.e., writing `False`.

    Returns
    -------
    Tensor
        Mask for triples.
    """
    real = real_pairs(numbers, mask_diagonal=False)
    mask = real.unsqueeze(-3) * real.unsqueeze(-2) * real.unsqueeze(-1)

    if mask_diagonal is True:
        mask *= ~torch.diag_embed(torch.ones_like(real))

    if mask_self is True:
        mask *= ~torch.diag_embed(torch.ones_like(real), offset=0, dim1=-3, dim2=-2)
        mask *= ~torch.diag_embed(torch.ones_like(real), offset=0, dim1=-3, dim2=-1)
        mask *= ~torch.diag_embed(torch.ones_like(real), offset=0, dim1=-2, dim2=-1)

    return mask
