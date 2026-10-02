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
Autograd Utility: Unwrapping
============================

Removal of the wrappers that ``torch.func`` transforms put around tensors.
"""

from __future__ import annotations

import torch

from ..tools.compile import is_compiling
from ..typing import Tensor

__all__ = ["unwrap_gradtracking"]


def unwrap_gradtracking(x: Tensor) -> Tensor:
    """
    Remove the grad-tracking wrappers a ``torch.func.jacrev``/``grad`` adds to
    a tensor created while it is active.

    Only the grad-tracking layers on the outside are removed; the unwrapping
    stops at a ``vmap`` layer. Use this for tensors that do not depend on a
    differentiated input, e.g. a constant built inside a transform that is
    kept beyond it (a cache): kept as is, it escapes the transform's level,
    and a later call under a different transform stack (e.g.
    ``jacrev(jacrev(...))``) fails with ``INTERNAL ASSERT FAILED ...
    escaped?``. For such a tensor, the plain tensor underneath carries the
    same values and no lost gradient.

    While ``torch.compile`` traces, the tensor is returned unchanged: the
    ``torch._C._functorch`` bindings cannot be traced.

    Parameters
    ----------
    x : Tensor
        The tensor to unwrap.

    Returns
    -------
    Tensor
        The tensor without its outer grad-tracking wrappers.
    """
    if is_compiling():
        return x

    ft = torch._C._functorch  # pyright: ignore[reportAttributeAccessIssue]
    while ft.is_gradtrackingtensor(x):
        x = ft.get_unwrapped(x)
    return x
