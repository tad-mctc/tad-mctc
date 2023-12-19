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
Conversion: Array/Tensor
========================

This module contains function for array conversion or reshaping.
"""
from __future__ import annotations

import torch

from .._typing import Size, Tensor


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
    try:
        atol = torch.finfo(x.dtype).eps * 10
    except TypeError:
        atol = 1e-5

    if x.ndim < 2:
        raise RuntimeError("Only matrices and batches thereof allowed.")

    if force is False:
        if not torch.allclose(x, x.mT, atol=atol):
            raise RuntimeError(
                f"Matrix appears to be not symmetric (atol={atol:.3e}, "
                f"dtype={x.dtype})."
            )

    return (x + x.mT) / 2
