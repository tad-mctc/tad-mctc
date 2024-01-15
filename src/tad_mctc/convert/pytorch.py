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
Conversion: PyTorch-specific Tools
==================================

This module contains PyTorch-specific conversion tools.
"""
from __future__ import annotations

import torch

from ..typing import Any, Tensor

__all__ = ["str_to_device", "any_to_tensor"]


def str_to_device(s: str) -> torch.device:
    """
    Convert device name to `torch.device`. Critically, this also sets the index
    for CUDA devices to `torch.cuda.current_device()`.

    Parameters
    ----------
    s : str
        Name of the device as string.

    Returns
    -------
    torch.device
        Device as torch class.

    Raises
    ------
    KeyError
        Unknown device name is given.
    """
    if "cpu" in s:
        return torch.device("cpu")

    if "cuda" in s:
        if not torch.cuda.is_available():
            raise KeyError(f"No CUDA devices available.")
        return torch.device("cuda", index=torch.cuda.current_device())

    raise KeyError(f"Unknown device '{s}' given.")


def any_to_tensor(
    x: Any,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """
    Convert various types of inputs to a PyTorch tensor.

    The device and dtype of the tensor can be specified. If the input is
    already a tensor, it's converted to the specified device and dtype.

    Parameters
    ----------
    x : Any
        The input to convert. Can be of type Tensor, float, int, bool, or str,
        or a list containing float, int, or bool.
    device : torch.device, optional
        The device on which to place the created tensor. If None, the default
        device is used.
    dtype : torch.dtype, optional
        The desired data type for the tensor. If None, the default data type is
        used or inferred from the input.

    Returns
    -------
    Tensor
        A PyTorch tensor representation of the input.

    Raises
    ------
    ValueError
        If `x` is a string that cannot be converted to a float or if the list
        contains elements other than float, int, or bool.
    TypeError
        If `x` is of a type that cannot be converted to a tensor.

    Examples
    --------
    >>> totensor(3.14)
    tensor(3.1400)

    >>> totensor(42, dtype=torch.float32)
    tensor(42.)

    >>> totensor(True)
    tensor(True)

    >>> totensor('2.718')
    tensor(2.7180)

    >>> totensor('not_a_number')
    ValueError: Cannot convert string 'not_a_number' to float

    >>> totensor(["1", "2"])
    TypeError: Tensor-incompatible type '<class 'list'>' of variable ["1", "2"].
    """
    if isinstance(x, Tensor):
        return x.to(device=device, dtype=dtype)

    if isinstance(x, list):
        if all(isinstance(item, (float, int, bool)) for item in x):
            return torch.tensor(x, device=device, dtype=dtype)
        else:
            raise ValueError("List must contain only float, int, or bool types.")

    if isinstance(x, (float, int, bool)):
        return torch.tensor(x, device=device, dtype=dtype)

    if isinstance(x, str):
        try:
            return torch.tensor(float(x), device=device, dtype=dtype)
        except ValueError:
            raise ValueError(f"Cannot convert string '{x}' to float")

    raise TypeError(f"Tensor-incompatible type '{type(x)}' of variable {x}.")
