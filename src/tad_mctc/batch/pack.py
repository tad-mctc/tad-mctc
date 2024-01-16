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
Batch Utility: Packing
======================

Pad a list of variable length tensors with zeros, or some other value, and
pack them into a single tensor.
"""
from __future__ import annotations

import torch

from ..typing import Any, Literal, Size, Tensor, TensorOrTensors, overload

__all__ = ["pack"]


@overload
def pack(  # type: ignore[misc]
    tensors: TensorOrTensors,
    axis: int = 0,
    value: int | float = 0,
    size: Size | None = None,
    return_mask: Literal[False] = False,
) -> Tensor:
    ...


@overload
def pack(
    tensors: TensorOrTensors,
    axis: int = 0,
    value: int | float = 0,
    size: Size | None = None,
    return_mask: Literal[True] = True,
) -> tuple[Tensor, Tensor]:
    ...


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
        size = [int(x) for x in torch.tensor([i.shape for i in tensors]).max(0).values]

    # Tensor to pack into, filled with padding value
    padded = torch.full((_count, *size), value, dtype=_dtype, device=_device)

    # Generate the mask if requested.
    if return_mask is True:
        mask = torch.full((_count, *size), False, dtype=torch.bool, device=_device)

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
    return (padded, mask) if return_mask is True and mask is not None else padded
