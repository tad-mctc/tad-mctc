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
I/O Checks: Shape
=================

This module contains shape checkers for the inputs passed to the reader/writer.
"""

from __future__ import annotations

from ...typing import Tensor

__all__ = ["shape_checks"]


def shape_checks(
    numbers: Tensor, positions: Tensor, allow_batched: bool = True
) -> bool:
    """
    Check the shapes of the numbers and positions tensors. This explicitly
    checks for non-batched tensor shapes (batched tensors throw errors).

    Parameters
    ----------
    numbers : Tensor
        A 1D tensor containing atomic numbers or symbols.
    positions : Tensor
        A 2D tensor of shape ``(nat, 3)`` containing atomic positions.

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
            "The last dimension of the position tensor must present the "
            "cartesian directions, i.e., it must be size 3 (but is "
            f"{positions.shape[-1]}"
        )

    if allow_batched is False:
        if len(numbers.shape) != 1 or len(positions.shape) != 2:
            raise ValueError(
                "Invalid shape for tensors (batched tensors not allowed)."
            )

    return True
