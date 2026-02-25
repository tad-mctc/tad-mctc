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
Batch Utility: Packing
======================

Pad a list of variable length tensors with zeros, or some other value, and
pack them into a single tensor.
"""

from __future__ import annotations

import torch

from ..typing import Literal, Size, Tensor, TensorOrTensors, overload

__all__ = ["pack"]


@overload
def pack(  # type: ignore[misc]
    tensors: TensorOrTensors,
    axis: int = 0,
    value: int | float = 0,
    size: Size | None = None,
    return_mask: Literal[False] = False,
) -> Tensor: ...


@overload
def pack(
    tensors: TensorOrTensors,
    axis: int = 0,
    value: int | float = 0,
    size: Size | None = None,
    return_mask: Literal[True] = True,
) -> tuple[Tensor, Tensor]: ...


def pack(
    tensors: TensorOrTensors,
    axis: int = 0,
    value: int | float = 0,
    size: Size | None = None,
    return_mask: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
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
    return_mask : bool, optional
        If `True`, a mask identifying the padding values is returned.
        Defaults to `False`.

    Returns
    -------
    Tensor | tuple[Tensor, Tensor]
        Input tensors padded and packed into a single tensor. Optionally, the
        mask is also returned.

    Examples
    --------
    Multiple tensors can be packed into a single tensor like so:

    >>> from tbmalt.common.batch import pack
    >>> import torch
    >>> a, b, c = torch.rand(2,2), torch.rand(3,3), torch.rand(4,4)
    >>> abc_packed_a = pack([a, b, c])
    >>> print(abc_packed_a.shape)
    torch.Size([3, 4, 4])
    >>> abc_packed_b = pack([a, b, c], axis=1)
    >>> print(abc_packed_b.shape)
    torch.Size([4, 3, 4])
    >>> abc_packed_c = pack([a, b, c], axis=-1)
    >>> print(abc_packed_c.shape)
    torch.Size([4, 4, 3])

    An optional mask identifying the padding values can also be returned:

    >>> packed, mask = pack(
    ...     [
    ...         torch.tensor([1.0]),
    ...         torch.tensor([2.0, 2.0]),
    ...         torch.tensor([3.0, 3.0, 3.0]),
    ...     ],
    ...     return_mask=True,
    ... )
    >>> print(packed)
    tensor([[1., 0., 0.],
            [2., 2., 0.],
            [3., 3., 3.]])
    >>> print(mask)
    tensor([[ True, False, False],
            [ True,  True, False],
            [ True,  True,  True]])
    """
    mask = None

    if isinstance(tensors, Tensor):
        return tensors

    _count = len(tensors)
    _device = tensors[0].device
    _dtype = tensors[0].dtype

    # Identify the maximum size, if one was not specified.
    if size is None:
        size = [
            int(x)
            for x in torch.tensor([i.shape for i in tensors]).max(0).values
        ]

    # Tensor to pack into, filled with padding value
    padded = torch.full((_count, *size), value, dtype=_dtype, device=_device)

    # Generate the mask if requested.
    if return_mask is True:
        mask = torch.full(
            (_count, *size), False, dtype=torch.bool, device=_device
        )

    # Loop over & pack "tensors" into "padded". A proxy index "n" must be used
    # for assignments rather than a slice to prevent in-place errors.
    for n, source in enumerate(tensors):
        # Slice operations not elegant but they are dimension agnostic & fast.
        padded[(n, *[slice(0, s) for s in source.shape])] = source

        # Update the mask if required.
        if return_mask is True and mask is not None:
            mask[(n, *[slice(0, s) for s in source.shape])] = True

    # If "axis" was anything other than 0, then "padded" must be permuted.
    if axis != 0:
        # Resolve relative any axes to their absolute equivalents to maintain
        # expected slicing behaviour when using the insert function.
        axis = padded.dim() + 1 + axis if axis < 0 else axis

        # Build a list of axes indices; but omit the axis on which the data
        # was concatenated (i.e. 0).
        order = list(range(1, padded.dim()))

        # Re-insert the concatenation axis as specified
        order.insert(axis, 0)

        # Perform the permeation
        padded = padded.permute(order)

        # Perform permeation on the mask is present.
        if return_mask is True and mask is not None:
            mask = mask.permute(order)

    # Return the packed tensor, and the mask if requested.
    return (
        (padded, mask) if return_mask is True and mask is not None else padded
    )
