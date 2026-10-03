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
Tree: ModuleNode
================

A ``torch.nn.Module`` as a :class:`~tad_mctc.tree.Node`: the architecture is
static context, the parameters and buffers are pytree leaves, so they can be
selected with :func:`~tad_mctc.tree.partition` and differentiated or batched
with ``torch.func``.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from .node import Node, child, context

__all__ = ["ModuleNode"]


class ModuleNode(Node):
    """
    A ``torch.nn.Module`` held in a node tree.

    ``module`` is used only for its architecture: it is always called
    through ``torch.func.functional_call`` with the tensors in ``params``,
    never with its own stored parameters or buffers.

    Parameters
    ----------
    module : torch.nn.Module
        The architecture. Static; compared and hashed by identity.
    params : dict[str, Tensor], optional
        Parameters and buffers by name, as in ``module.state_dict()`` keys.
    """

    module: torch.nn.Module = context()
    params: dict[str, Tensor] = child(default_factory=dict)

    @classmethod
    def from_module(cls, module: torch.nn.Module) -> ModuleNode:
        """
        Take the parameters and buffers of a module.

        Parameters
        ----------
        module : torch.nn.Module
            The module to wrap.

        Returns
        -------
        ModuleNode
            Node holding ``module`` and its named parameters and buffers.
        """
        params: dict[str, Tensor] = dict(module.named_parameters())
        params.update(dict(module.named_buffers()))
        return cls(module=module, params=params)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        Call the module with the tensors in ``params``.

        Parameters
        ----------
        *args : Any
            Positional arguments of the module's ``forward``.
        **kwargs : Any
            Keyword arguments of the module's ``forward``.

        Returns
        -------
        Any
            Output of the module's ``forward``.
        """
        return torch.func.functional_call(
            self.module, self.params, args, kwargs
        )
