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
Tools: Memory
-------------

Functions for estimating the memory requirements of tensors.

Example
-------
>>> import torch
>>> from tad_mctc.tools import memory_tensor
>>> size = (100, 100)
>>> dtype = torch.float32
>>> memory_tensor(size, dtype)
0.03814697265625
"""

from __future__ import annotations

import torch

from ..typing import Size

__all__ = ["memory_tensor"]


def memory_tensor(size: Size, dtype: torch.dtype) -> float:
    """
    Estimate the memory usage of a tensor.

    Parameters
    ----------
    size : Size
        Shape of the tensor.
    dtype : :class:`torch.dtype`
        Data type of the tensor.

    Returns
    -------
    float
        Estimated memory usage in MB.

    Raises
    ------
    ValueError
        If the tensor data type is not supported.
    """
    # Product of elements in the size tuple gives the total number of elements
    num_elements = 1
    for dimension in size:
        num_elements *= dimension

    # Determine the size of each element based on the tensor's dtype
    if dtype == torch.float64 or dtype == torch.double:
        element_size = 8
    elif dtype == torch.float32:
        element_size = 4
    elif dtype == torch.float16:
        element_size = 2
    elif dtype == torch.int64 or dtype == torch.long:
        element_size = 8
    elif dtype == torch.int32:
        element_size = 4
    elif dtype == torch.int16:
        element_size = 2
    elif dtype == torch.int8 or dtype == torch.uint8:
        element_size = 1
    else:
        raise ValueError(f"Unsupported tensor dtype: {dtype}")

    return num_elements * element_size / (1024**2)
