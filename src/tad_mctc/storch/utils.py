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
SafeOps: Utility and Helpers
============================

Some helper functions.
"""

from __future__ import annotations

import torch

from ..typing import Tensor

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
    return torch.tensor(
        torch.finfo(x.dtype).eps, device=x.device, dtype=x.dtype
    )
