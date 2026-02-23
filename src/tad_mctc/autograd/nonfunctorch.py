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
Autograd Utility: Loop-based Jacobian
=====================================

These derivative functions do not use `functorch`, but construct the Jacobian
row-by-row. This is slower than using `functorch`.
"""

from __future__ import annotations

import torch

from ..typing import Tensor

__all__ = ["jac"]


def jac(
    a: Tensor,
    b: Tensor,
    create_graph: bool | None = None,
    retain_graph: bool = True,
) -> Tensor:
    """
    Compute the Jacobian of ``a`` with respect to ``b`` row-by-row.

    Parameters
    ----------
    a : Tensor
        Variable that is differentiated.
    b : Tensor
        Variable with respect to which the derivative is taken.
    create_graph : bool | None, optional
        Whether to create a backprogatable graph. Required for additional
        (higher) derivatives. Defaults to ``True``.
    retain_graph : bool, optional
        Whether to use the multiple graph multiple times. Defaults to ``True``.
        Otherwise, the graph is deleted after the first call.

    Returns
    -------
    Tensor
        Jacobian of ``a`` with respect to ``b``.
    """
    # catch missing gradients (e.g., halogen bond correction evaluates to
    # zero if no donors/acceptors are present)
    if a.grad_fn is None:
        return torch.zeros(
            (*a.shape, b.numel()),
            dtype=b.dtype,
            device=b.device,
        )

    if create_graph is None:
        create_graph = torch.is_grad_enabled()
    assert create_graph is not None

    aflat = a.reshape(-1)
    anumel, bnumel = a.numel(), b.numel()
    res = torch.empty(
        (anumel, bnumel),
        dtype=a.dtype,
        device=a.device,
    )

    for i in range(aflat.numel()):
        (g,) = torch.autograd.grad(
            aflat[i],
            b,
            create_graph=create_graph,
            retain_graph=retain_graph,
        )
        res[i] = g.reshape(-1)

    return res.reshape((*a.shape, bnumel))
