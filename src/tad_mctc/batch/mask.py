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
Batch Utility: Masks
====================

Functions for creating masks that discern between padding and actual values.
"""
from __future__ import annotations

import torch

from .._typing import Tensor

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


def real_pairs(numbers: Tensor, diagonal: bool = False) -> Tensor:
    """
    Create a mask for pairs of atoms from atomic numbers, discerning padding
    and actual atoms. Padding value is zero.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers for all atoms.
    diagonal : bool, optional
        Flag for also writing `False` to the diagonal, i.e., to all pairs
        with the same indices. Defaults to `False`, i.e., writing False
        to the diagonal.

    Returns
    -------
    Tensor
        Mask for atom pairs that discerns padding and real atoms.
    """
    real = real_atoms(numbers)
    mask = real.unsqueeze(-2) * real.unsqueeze(-1)
    if diagonal is False:
        mask *= ~torch.diag_embed(torch.ones_like(real))
    return mask


def real_triples(
    numbers: torch.Tensor, diagonal: bool = False, self: bool = True
) -> Tensor:
    """
    Create a mask for triples from atomic numbers. Padding value is zero.

    Parameters
    ----------
    numbers : torch.Tensor
        Atomic numbers for all atoms.
    diagonal : bool, optional
        Flag for also writing `False` to the space diagonal, i.e., to all
        triples with the same indices. Defaults to `False`, i.e., writing False
        to the diagonal.
    self : bool, optional
        Flag for also writing `False` to all triples where at least two indices
        are identical. Defaults to `True`, i.e., not writing `False`.

    Returns
    -------
    Tensor
        Mask for triples.
    """
    real = real_pairs(numbers, diagonal=True)
    mask = real.unsqueeze(-3) * real.unsqueeze(-2) * real.unsqueeze(-1)

    if diagonal is False:
        mask *= ~torch.diag_embed(torch.ones_like(real))

    if self is False:
        mask *= ~torch.diag_embed(torch.ones_like(real), offset=0, dim1=-3, dim2=-2)
        mask *= ~torch.diag_embed(torch.ones_like(real), offset=0, dim1=-3, dim2=-1)
        mask *= ~torch.diag_embed(torch.ones_like(real), offset=0, dim1=-2, dim2=-1)

    return mask
