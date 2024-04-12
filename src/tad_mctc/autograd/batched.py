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
Autograd Utility: Batched
=========================

Batched versions of autograd functions.
"""

from __future__ import annotations

from ..typing import Callable, Tensor
from .internals import jacrev, vmap

__all__ = ["bjacrev"]


def bjacrev(
    func: Callable[..., Tensor], argnums: int = 0, **kwargs
) -> Callable[..., Tensor]:
    """
    Batched Jacobian of a function.

    Parameters
    ----------
    func : Callable[..., Tensor]
        The function whose result is differentiated.
    argnums : int, optional
        The variable w.r.t. which will be differentiated. Defaults to 0.

    Returns
    -------
    Callable[..., Tensor]
        Batched Jacobian function.
    """
    f: Callable = jacrev(func, argnums=argnums, **kwargs)  # type: ignore
    return vmap(f, in_dims=0, out_dims=0)
