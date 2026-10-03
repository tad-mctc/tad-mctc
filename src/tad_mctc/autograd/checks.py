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
Autograd Utility: Checks
========================

Utility functions for checking properties of tensors in the context of
automatic differentiation, such as whether a tensor is a grad tracking tensor,
is ``vmap``-batched, or either of both (i.e., a "functorch" tensor).

All checks return ``True`` while ``torch.compile`` is tracing: the underlying
``torch._C._functorch`` bindings cannot be traced, and the values of a traced
tensor are not concrete either, so callers that skip data-dependent checks
(``if not is_functorch_tensor(x): ...``) or pick a ``vmap``-safe code path
(``if is_vmapped(x): ...``) get the safe answer without a graph break.
"""

from __future__ import annotations

import torch

from ..tools.compile import is_compiling
from ..typing import Tensor

__all__ = ["is_gradtracking", "is_vmapped", "is_functorch_tensor"]


def is_gradtracking(x: Tensor) -> bool:
    """
    Check if the input tensor is a grad tracking tensor.

    Note
    ----
    Always ``True`` while ``torch.compile`` is tracing (see module docstring).

    Parameters
    ----------
    x : Tensor
        The tensor to check.

    Returns
    -------
    bool
        ``True`` if the tensor is a grad tracking tensor, ``False`` otherwise.
    """
    if is_compiling():
        return True
    ft = torch._C._functorch  # pyright: ignore[reportAttributeAccessIssue]
    return ft.is_gradtrackingtensor(x)


def is_vmapped(x: Tensor) -> bool:
    """
    Check if the input tensor is wrapped by a ``torch.func.vmap`` at any layer
    of the functorch wrapper stack.

    This is about ``vmap`` only, not about a leading batch dimension (e.g.
    from :func:`tad_mctc.batch.pack`), which is an ordinary dimension of the
    tensor. Only a ``vmap`` layer makes data-dependent output shapes (e.g.
    ``torch.unique``) illegal; ``torch.func.jacrev``/``grad`` wrap *all*
    arguments in a grad-tracking layer, which :func:`is_functorch_tensor`
    also reports, but this check does not.

    The ``vmap`` layer is found even below such grad-tracking layers, e.g.
    under ``vmap(jacrev(jacrev(f)))`` for batched Hessians, where the
    outermost wrapper is a grad-tracking one.

    Note
    ----
    Always ``True`` while ``torch.compile`` is tracing (see module docstring).

    Parameters
    ----------
    x : Tensor
        The tensor to check.

    Returns
    -------
    bool
        ``True`` if a ``vmap`` is active on the tensor, ``False`` otherwise.
    """
    if is_compiling():
        return True

    ft = torch._C._functorch  # pyright: ignore[reportAttributeAccessIssue]
    while ft.is_functorch_wrapped_tensor(x):
        if ft.is_batchedtensor(x):
            return True
        x = ft.get_unwrapped(x)
    return False


def is_functorch_tensor(x: Tensor) -> bool:
    """
    Check if the input tensor is a functorch tensor.

    Note
    ----
    Always ``True`` while ``torch.compile`` is tracing (see module docstring).

    Parameters
    ----------
    x : Tensor
        The tensor to check.

    Returns
    -------
    bool
        ``True`` if the tensor is a functorch tensor, ``False`` otherwise.
    """
    if is_compiling():
        return True
    return is_gradtracking(x) or is_vmapped(x)
