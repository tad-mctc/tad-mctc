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
Batch Utility: Sorting
======================

Sort a packed ``tensor`` while ignoring any padding values.
"""

from __future__ import annotations

import torch

from tad_mctc.typing import NamedTuple, Tensor

__all__ = ["pargsort", "psort"]


class _SortResult(NamedTuple):
    values: Tensor
    indices: Tensor


def pargsort(
    tensor: Tensor, mask: Tensor | None = None, dim: int = -1
) -> Tensor:
    """
    Returns indices that sort packed tensors while ignoring padding values.

    This function returns the indices that sort the elements of `tensor` along
    the specified dimension `dim` in ascending order by value, while ensuring
    padding values are shuffled to the end of the dimension.

    Parameters
    ----------
    tensor : Tensor
        The input tensor to be sorted.
    mask : Tensor, optional
        A boolean tensor where ``True`` indicates real values and ``False``
        indicates padding values. If not provided, the function will redirect to ``torch.argsort``. Default is None.
    dim : int, optional
        The dimension along which to sort the tensor. Default is -1.

    Returns
    -------
    Tensor
        A tensor of indices that sort the input tensor along the specified
        dimension ``dim``.

    Notes
    -----
    If no ``mask`` is provided, this function redirects to ``torch.argsort``.
    """
    if mask is None:
        return torch.argsort(tensor, dim=dim)
    else:
        # A secondary sorter is used to reorder the primary sorter so that
        # padding values are moved to the end.
        n = tensor.shape[dim]
        s1 = tensor.argsort(dim)
        s2 = (
            torch.arange(n, device=tensor.device) + (~mask.gather(dim, s1) * n)
        ).argsort(dim)
        return s1.gather(dim, s2)


def psort(
    tensor: Tensor, mask: Tensor | None = None, dim: int = -1
) -> _SortResult:
    """
    Sort a packed tensor while ignoring any padding values.

    This function sorts the elements of ``tensor`` along the specified
    dimension ``dim`` in ascending order by value, while ensuring padding
    values are shuffled to the end of the dimension.

    Parameters
    ----------
    tensor : Tensor
        The input tensor to be sorted.
    mask : Tensor, optional
        A boolean tensor where ``True`` indicates real values and ``False``
        indicates padding values. If not provided, the function will redirect
        to ``torch.sort``. Default is ``None``.
    dim : int, optional
        The dimension along which to sort the tensor. Default is -1.

    Returns
    -------
    _SortResult
        A namedtuple (values, indices) is returned, where ``values`` are the
        sorted  values and ``indices`` are the indices of the elements in the
        original input tensor.

    Notes
    -----
    If no ``mask`` is provided, this function redirects to ``torch.sort``.
    """

    if mask is None:
        values, indices = torch.sort(tensor, dim=dim)
        return _SortResult(values=values, indices=indices)

    indices = pargsort(tensor, mask, dim)
    sorted_tensor = tensor.gather(dim, indices)
    return _SortResult(values=sorted_tensor, indices=indices)
