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
Autograd Utility: Checks
========================

Utility functions for checking properties of tensors in the context of
automatic differentiation, such as whether a tensor is a grad tracking tensor,
batched tensor, or a both (i.e., a "functorch" tensor).
"""

from __future__ import annotations

import torch

from .._version import __tversion__
from ..typing import Tensor

__all__ = ["is_gradtracking", "is_batched", "is_functorch_tensor"]


def is_gradtracking(x: Tensor) -> bool:
    """
    Check if the input tensor is a grad tracking tensor.

    Note
    ----
    Defaults to ``False`` for versions of PyTorch before 2.0.0.

    Parameters
    ----------
    x : Tensor
        The tensor to check.

    Returns
    -------
    bool
        ``True`` if the tensor is a grad tracking tensor, ``False`` otherwise.
    """
    if __tversion__ >= (2, 0, 0):
        return torch._C._functorch.is_gradtrackingtensor(x)
    return False


def is_batched(x: Tensor) -> bool:
    """
    Check if the input tensor is a batched tensor.

    Only checks the first wrapper layer, i.e., grad-tracking tensors can
    obscure the batched nature of a tensor. Unwrap the tensor first to check
    the underlying tensor.

    Note
    ----
    Defaults to ``False`` for versions of PyTorch before 2.0.0.

    Parameters
    ----------
    x : Tensor
        The tensor to check.

    Returns
    -------
    bool
        ``True`` if the tensor is a batched tensor, ``False`` otherwise.
    """
    if __tversion__ >= (2, 0, 0):
        return torch._C._functorch.is_batchedtensor(x)
    return False


def is_functorch_tensor(x: Tensor) -> bool:
    """
    Check if the input tensor is a functorch tensor.

    Note
    ----
    Defaults to ``False`` for versions of PyTorch before 2.0.0.

    Parameters
    ----------
    x : Tensor
        The tensor to check.

    Returns
    -------
    bool
        ``True`` if the tensor is a functorch tensor, ``False`` otherwise.
    """
    if __tversion__ >= (2, 0, 0):
        return is_gradtracking(x) or is_batched(x)
    return False
