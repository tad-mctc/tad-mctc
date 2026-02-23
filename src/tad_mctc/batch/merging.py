# This file is part of tad-mctc, modified from tbmalt/tbmalt.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2024 Grimme Group
#
# Original file licensed under the LGPL-3.0 License by tbmalt/tbmalt.
# Modifications made by Grimme Group.
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
Batch Utility: Merging
======================

Pad a list of variable length tensors with zeros, or some other value, and
pack them into a single tensor.
"""

from __future__ import annotations

import torch

from ..typing import Any, Tensor, TensorOrTensors

__all__ = ["merge"]


def merge(tensors: TensorOrTensors, value: Any = 0, axis: int = 0) -> Tensor:
    """
    Merge two or more packed tensors into a single packed tensor.

    Parameters
    ----------
    tensors : TensorOrTensors
        Packed tensors which are to be merged.
    value : Any, optional
        Value with which the tensor were/are to be padded. Default is 0.
    axis : int, optional
        Axis along which ``tensors`` are to be stacked. Default is 0.

    Returns
    -------
    Tensor
        The tensors ``tensors`` merged along the axis ``axis``.

    Warnings
    --------
    Care must be taken to ensure the correct padding value is specified as
    erroneous behavior may otherwise ensue. As the correct padding value
    cannot be reliably detected in situ, it defaults to zero.
    """
    # Merging is performed along the 0'th axis internally. If a non-zero axis
    # is requested then tensors must be reshaped during input and output.
    if axis != 0:
        tensors = [t.transpose(0, axis) for t in tensors]

    # Get sizes
    shapes = torch.tensor([i.shape for i in tensors], dtype=torch.int64)
    size = (shapes.sum(0)[0], *shapes.max(0).values[1:].tolist())
    size = tuple(int(i) for i in size)  # only for typing

    # Tensor to merge into, filled with padding value.
    merged = torch.full(
        size,
        value,
        dtype=tensors[0].dtype,
        device=tensors[0].device,
    )

    n = 0  # <- batch dimension offset
    for src, size in zip(tensors, shapes):  # Assign values to tensor
        merged[(slice(n, size[0] + n), *[slice(0, s) for s in size[1:]])] = src
        n += size[0]

    # Return the merged tensor, transposing back as required
    return merged if axis == 0 else merged.transpose(0, axis)
