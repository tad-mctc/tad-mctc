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
Safe Operations: Utility and Helpers
====================================

Some helper functions.
"""
from __future__ import annotations

import torch

from .._typing import Tensor

__all__ = ["get_eps"]


def get_eps(x: Tensor) -> Tensor:
    """
    Get the smallest value corresponding to the input floating point precision
    of the input tensor. The small value will have the same dtype and lives on
    the same device as the input tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor that defines the dtype.

    Returns
    -------
    Tensor
        Smallest value of corresponding dtype.
    """
    return torch.tensor(torch.finfo(x.dtype).eps, device=x.device, dtype=x.dtype)
