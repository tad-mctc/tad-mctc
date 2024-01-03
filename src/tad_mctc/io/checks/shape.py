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
I/O Checks: Shape
=================

This module contains shape checkers for the inputs passed to the reader/writer.
"""
from __future__ import annotations

from ...typing import Tensor

__all__ = ["shape_checks"]


def shape_checks(numbers: Tensor, positions: Tensor) -> bool:
    """
    Check the shapes of the numbers and positions tensors. This explicitly
    checks for non-batched tensor shapes (batched tensors throw errors).

    Parameters
    ----------
    numbers : Tensor
        A 1D tensor containing atomic numbers or symbols.
    positions : Tensor
        A 2D tensor of shape (n_atoms, 3) containing atomic positions.

    Returns
    -------
    bool
        True if the shapes are correct.

    Raises
    ------
    ValueError
        If the shapes of both tensors are inconsistent, the last dimension of
        the positions tensor is not 3 (cartesian directions), the numbers
        tensor has not one dimension, or the positions tensor has not two
        dimensions.
    """
    if numbers.shape != positions.shape[:-1]:
        raise ValueError(
            f"Shape of positions ({positions.shape[:-1]}) is not consistent "
            f"with atomic numbers ({numbers.shape})."
        )

    if positions.shape[-1] != 3:
        raise ValueError(
            f"The last dimension of the position tensor must present the cartesian directions, i.e., it must be size 3 (but is {positions.shape[-1]}"
        )

    if len(numbers.shape) != 1 or len(positions.shape) != 2:
        raise ValueError("Invalid shape for tensors (batched tensors not allowed).")

    return True
