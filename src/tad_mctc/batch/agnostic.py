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
Batch: Agnostic Ops
===================

Batch-agnostic versions of PyTorch operations that do not work with batched and
non-batched tensors out of the box.
"""
from __future__ import annotations

import torch

from ..typing import Tensor

__all__ = ["eye"]


def eye(
    tensor: Tensor,
    value: float = 1.0,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """
    Create an identity tensor. This version handles a possible batch
    dimension.

    Parameters
    ----------
    tensor : Tensor
        Tensor shape to create the identity tensor from.
    value : float, optional
        Value to fill the diagonal with. Defaults to `1.0`.
    device : torch.device | None, optional
        Device to create the tensor on. If `None`, the device of the input
    dtype : torch.dtype | None, optional
        Data type of the tensor. If `None`, the data type of the input.

    Returns
    -------
    Tensor
        Possibly batched identity tensor.
    """
    identity = torch.zeros(
        tensor.shape,
        device=device if device is not None else tensor.device,
        dtype=dtype if dtype is not None else tensor.dtype,
    )
    identity.diagonal(dim1=-2, dim2=-1).fill_(value)
    return identity
