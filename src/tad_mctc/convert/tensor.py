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
Conversion: Array/Tensor
========================

This module contains function for conversions of PyTorch tensors. This
includes, for example, reshaping.
Conversion into tensors from other data types (integer, float, etc.) is not
provided by this module.
"""

from __future__ import annotations

from functools import partial

import torch

from ..typing import Size, Tensor

__all__ = ["reshape_fortran", "symmetrize", "symmetrizef"]


def reshape_fortran(x: Tensor, shape: Size) -> Tensor:
    """
    Implements Fortran's `reshape` function (column-major).

    Parameters
    ----------
    x : Tensor
        Input tensor.
    shape : Size
        Output size to which `x` is reshaped.

    Returns
    -------
    Tensor
        Reshaped tensor of size `shape`.
    """
    if len(x.shape) > 0:
        x = x.permute(*reversed(range(len(x.shape))))
    return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))


def symmetrize(x: Tensor, force: bool = False) -> Tensor:
    """
    Symmetrize a tensor after checking if it is symmetric within a threshold.

    Parameters
    ----------
    x : Tensor
        Tensor to check and symmetrize.
    force : bool
        Whether symmetry should be forced. This allows switching between actual
        symmetrizing and only cleaning up numerical noise. Defaults to `False`.

    Returns
    -------
    Tensor
        Symmetrized tensor.

    Raises
    ------
    RuntimeError
        If the tensor is not symmetric within the threshold.
    """
    if x.ndim < 2:
        raise RuntimeError("Only matrices and batches thereof allowed.")

    if force is True:
        return (x + x.mT) / 2.0

    try:
        atol = torch.finfo(x.dtype).eps * 10
    except TypeError:  # pragma: no cover
        atol = 1e-5

    if not torch.allclose(x, x.mT, atol=atol):
        raise RuntimeError(
            f"Matrix appears to be not symmetric (atol={atol:.3e}, "
            f"dtype={x.dtype})."
        )

    return (x + x.mT) / 2.0


symmetrizef = partial(symmetrize, force=True)
"""Force symmetrization of a tensor."""
