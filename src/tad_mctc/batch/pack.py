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
Batch Utility: Packing
======================

Pad a list of variable length tensors with zeros, or some other value, and
pack them into a single tensor.
"""
from __future__ import annotations

import torch

from .._typing import Size, Tensor, TensorOrTensors

__all__ = ["pack"]


def pack(
    tensors: TensorOrTensors,
    axis: int = 0,
    value: int | float = 0,
    size: Size | None = None,
) -> Tensor:
    """
    Pad a list of variable length tensors with zeros, or some other value, and
    pack them into a single tensor.

    Parameters
    ----------
    tensors : list[Tensor] | tuple[Tensor] | Tensor
        List of tensors to be packed, all with identical dtypes.
    axis : int
        Axis along which tensors should be packed; 0 for first axis -1
        for the last axis, etc. This will be a new dimension.
    value : int | float
        The value with which the tensor is to be padded.
    size :
        Size of each dimension to which tensors should be padded.
        This to the largest size encountered along each dimension.

    Returns
    -------
    padded : Tensor
        Input tensors padded and packed into a single tensor.
    """
    if isinstance(tensors, Tensor):
        return tensors

    _count = len(tensors)
    _device = tensors[0].device
    _dtype = tensors[0].dtype

    if size is None:
        size = torch.tensor([i.shape for i in tensors]).max(0).values.tolist()

    padded = torch.full((_count, *size), value, dtype=_dtype, device=_device)

    for n, source in enumerate(tensors):
        padded[(n, *[slice(0, s) for s in source.shape])] = source

    if axis != 0:
        axis = padded.dim() + 1 + axis if axis < 0 else axis
        order = list(range(1, padded.dim()))
        order.insert(axis, 0)
        padded = padded.permute(order)

    return padded
