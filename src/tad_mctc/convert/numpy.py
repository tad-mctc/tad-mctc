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
Conversion: numpy
=================

This module contains safe functions for numpy and pytorch interconversion.
"""
from __future__ import annotations

import numpy as np
import torch
from numpy.typing import DTypeLike, NDArray

from ..typing import Any, Tensor, get_default_dtype

__all__ = ["numpy_to_tensor", "tensor_to_numpy"]


numpy_to_torch_dtype_dict = {
    np.dtype(np.float16).type: torch.float16,
    np.dtype(np.float32).type: torch.float32,
    np.dtype(np.float64).type: torch.float64,
    np.dtype(np.int8).type: torch.int8,
    np.dtype(np.int16).type: torch.int16,
    np.dtype(np.int32).type: torch.int32,
    np.dtype(np.int64).type: torch.int64,
    np.dtype(np.uint8).type: torch.uint8,
}
"""Dict of NumPy dtype -> torch dtype (when the correspondence exists)"""

torch_to_numpy_dtype_dict: dict[torch.dtype, DTypeLike] = {
    torch.float16: np.dtype(np.float16),
    torch.float32: np.dtype(np.float32),
    torch.float64: np.dtype(np.float64),
    torch.int8: np.dtype(np.int8),
    torch.int16: np.dtype(np.int16),
    torch.int32: np.dtype(np.int32),
    torch.int64: np.dtype(np.int64),
    torch.uint8: np.dtype(np.uint8),
}
"""Dict of torch dtype -> NumPy dtype conversion"""


def numpy_to_tensor(
    x: NDArray[Any],
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """
    Convert a numpy array to a PyTorch tensor.

    Parameters
    ----------
    x : NDArray[Any]
        Array to convert.
    device : torch.device | None, optional
        Device to store the tensor on. Defaults to `None`.
    dtype : torch.dtype | None, optional
        Data type of the tensor. Defaults to `None`.

    Returns
    -------
    Tensor
        Converted PyTorch tensor.
    """
    if dtype is None:
        dtype = numpy_to_torch_dtype_dict.get(x.dtype.type, get_default_dtype())
    assert dtype is not None

    return torch.from_numpy(x).type(dtype).to(device)


def tensor_to_numpy(x: Tensor, dtype: DTypeLike | None = None) -> NDArray[Any]:
    """
    Convert a PyTorch tensor to a numpy array.

    Parameters
    ----------
    x : Tensor
        Tensor to convert.
    dtype : np.dtype, optional
        Data type of the array. Defaults to `np.dtype(np.float64)`.

    Returns
    -------
    np.ndarray
        Converted numpy array.
    """
    if dtype is None:
        dtype = torch_to_numpy_dtype_dict.get(x.dtype, np.dtype(np.float64))

    xdtype = torch_to_numpy_dtype_dict.get(x.dtype)
    x = x.detach().cpu()

    # pylint: disable=protected-access
    # see: https://github.com/pytorch/pytorch/issues/91810
    if torch._C._functorch.is_gradtrackingtensor(x):
        while torch._C._functorch.is_functorch_wrapped_tensor(x) is True:
            x = torch._C._functorch.get_unwrapped(x)

        if torch.__version__ < (2, 0, 0):  # type: ignore[operator] # pragma: no cover
            interpreted = np.array(x.storage().tolist(), dtype=dtype)
        else:
            storage_bytes = bytes(x.untyped_storage())  # type: ignore
            interpreted = np.frombuffer(storage_bytes, dtype=xdtype).astype(dtype)

        return interpreted.reshape(x.shape)

    _x: NDArray[Any] = x.numpy()
    return _x.astype(dtype)
