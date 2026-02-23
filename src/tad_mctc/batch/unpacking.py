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
Batch Utility: Unpacking
========================

Functions for unpacking packed tensors and removal of padding.
"""

from __future__ import annotations

from functools import partial, reduce

import torch

from ..typing import Tensor

__all__ = ["deflate", "unpack"]


def deflate(
    tensor: Tensor, value: int | float | bool = 0, axis: int | None = None
) -> Tensor:
    """
    Remove extraneous, trailing padding values from a tensor.

    Shrinks the given tensor by removing superfluous trailing padding values.
    All axes are deflated by default, but a specific axis can be exempted from
    deflation using the `axis` parameter.

    Parameters
    ----------
    tensor : Tensor
        The tensor to be deflated.
    value : int | float | bool, optional
        The identity of the padding value, by default 0.
    axis : int or None, optional
        The axis which is exempt from deflation, by default None.

    Returns
    -------
    Tensor
        The deflated tensor.

    Notes
    -----
    Only trailing padding values are removed; columns are only removed from
    the end of a matrix, not the start or middle. Deflation is not performed
    on one-dimensional systems when `axis` is not None.

    Examples
    --------
    Remove unnecessary padding from a batch:

    >>> over_packed = torch.tensor([
    >>>     [0, 1, 2, 0, 0, 0],
    >>>     [3, 4, 5, 6, 0, 0],
    >>> ])
    >>> deflate(over_packed, value=0, axis=0)
    tensor([[0, 1, 2, 0],
            [3, 4, 5, 6]])

    Remove padding from a system once part of a batch:

    >>> packed = torch.tensor([
    >>>     [0, 1, 0, 0],
    >>>     [3, 4, 0, 0],
    >>>     [0, 0, 0, 0],
    >>>     [0, 0, 0, 0]])
    >>> deflate(packed, value=0)
    tensor([[0, 1],
            [3, 4]])

    Warnings
    --------
    Real elements may be misidentified as padding values if they match the
    padding value. Choose an appropriate padding value to mitigate this risk.

    Raises
    ------
    ValueError
        If `tensor` is 0-dimensional or 1-dimensional when `axis` is not None.
    """
    # Check shape is viable
    if axis is not None and tensor.ndim <= 1:
        raise ValueError(
            "Tensor must be at least 2D when specifying an ``axis``."
        )

    mask = tensor == value
    if axis is not None:
        mask = mask.all(axis)

    # Check for empty mask
    if mask.numel() == 0:
        return tensor

    slices = []

    # When multidimensional `all` is required
    if (ndim := mask.ndim) > 1:
        for dim in reversed(torch.combinations(torch.arange(ndim), ndim - 1)):
            # Count Nº of trailing padding values. Reduce/partial used here as
            # torch.all cannot operate on multiple dimensions like numpy.
            torchall = partial(torch.all, keepdims=True)
            v, c = (
                reduce(torchall, dim, mask)
                .squeeze()
                .unique_consecutive(return_counts=True)
            )

            # Slicer will be None if there are no trailing padding values.
            slices.append(slice(None, -c[-1] if v[-1] else None))

    # If mask is one dimensional, then no loop is needed
    else:
        v, c = mask.unique_consecutive(return_counts=True)
        slices.append(slice(None, -c[-1] if v[-1] else None))

    if axis is not None:
        slices.insert(axis, slice(None))  # <- dummy index for batch-axis

    return tensor[slices]


def unpack(
    tensor: Tensor, value: int | float | bool = 0, axis: int = 0
) -> tuple[Tensor, ...]:
    """
    Unpacks packed tensors into their constituents and removes padding.

    This function acts as the inverse of the `pack` operation. It splits a
    packed tensor along a specified axis and removes any padding that was added
    during packing.

    Parameters
    ----------
    tensor : Tensor
        The tensor to be unpacked.
    value : int | float | bool, optional
        The identity of the padding value, by default 0.
    axis : int, optional
        The axis along which the tensor was packed, by default 0.

    Returns
    -------
    tuple[Tensor, ...]
        A tuple of the constituent tensors after unpacking and deflating.

    Examples
    --------
    Suppose you have a tensor that has been packed along the first axis (axis=0):

    >>> packed_tensor = torch.tensor([
    >>>     [1, 2, 3, 0, 0],
    >>>     [4, 5, 0, 0, 0],
    >>>     [6, 7, 8, 9, 0]
    >>> ])

    Unpacking this tensor would yield:

    >>> unpacked_tensors = unpack(packed_tensor, value=0, axis=0)
    >>> for tensor in unpacked_tensors:
    >>>     print(tensor)
    tensor([1, 2, 3])
    tensor([4, 5])
    tensor([6, 7, 8, 9])
    """
    # Handle 1D tensor case
    if tensor.ndim == 1:
        # Split tensor into non-padded and padded parts
        non_padded, padded = torch.split(
            tensor,
            [
                int(torch.sum(tensor != value).item()),
                int(torch.sum(tensor == value).item()),
            ],
        )
        return (non_padded,) if non_padded.nelement() > 0 else (padded,)

    return tuple(deflate(i, value) for i in tensor.movedim(axis, 0))
