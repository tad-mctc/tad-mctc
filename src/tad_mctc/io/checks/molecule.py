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
Checks: Molecule
================

This module contains various checkers for the molecule that is read/written.
"""
from __future__ import annotations

import torch

from ... import storch
from ...batch import real_pairs
from ...data import pse
from ...exceptions import MoleculeError
from ...typing import DD, Tensor

__all__ = ["coldfusion_check", "content_checks"]


def coldfusion_check(
    numbers: Tensor, positions: Tensor, threshold: Tensor | float | int | None = None
) -> bool:
    """
    Check if interatomic distances are large enough (no fusion of atoms).

    Parameters
    ----------
    numbers : Tensor
        A 1D tensor containing atomic numbers or symbols.
    positions : Tensor
        A 2D tensor of shape (n_atoms, 3) containing atomic positions.
    threshold : Tensor | float | int | None, optional
        Threshold for acceptable interatomic distances. Defaults to `None`, which resolves to `torch.tensor(torch.finfo(dtype).eps ** 0.75, **dd)`.

    Returns
    -------
    bool
        True of atoms are not too close.

    Raises
    ------
    MoleculeError
        Interatomic distances are too close.
    """
    dd: DD = {"device": positions.device, "dtype": positions.dtype}

    mask = real_pairs(numbers, mask_diagonal=True)
    distances = torch.where(
        mask,
        storch.cdist(positions, positions),
        torch.tensor(1e100, **dd),
    )

    if threshold is None:
        threshold = torch.tensor(torch.finfo(dd["dtype"]).eps ** 0.75, **dd)

    # Check if any distance below the threshold is found
    if torch.any((distances < threshold) & mask):
        raise MoleculeError("Too close interatomic distances found")

    return True


def content_checks(numbers: Tensor, positions: Tensor) -> bool:
    """
    Check the content of the numbers and positions tensors.

    This function should be asserted as it returns `True` on success and raises
    an error on failure.

    Parameters
    ----------
    numbers : Tensor
        A 1D tensor containing atomic numbers or symbols.
    positions : Tensor
        A 2D tensor of shape (n_atoms, 3) containing atomic positions.

    Returns
    -------
    bool
        True if content is correct.

    Raises
    ------
    ValueError
        Atomic number too large or too small.
    """
    if numbers.max() > pse.MAX_ELEMENT:
        raise MoleculeError(f"Atomic number larger than {pse.MAX_ELEMENT} found.")
    if numbers.min() < 1:
        raise MoleculeError(
            "Atomic number smaller than 1 found. This may indicate residual "
            "padding. Remove before writing to file."
        )

    assert coldfusion_check(numbers, positions)

    return True
