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
