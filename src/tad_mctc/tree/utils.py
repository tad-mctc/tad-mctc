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
Tree: utilities
===============

Helpers to inspect, split, merge and stack pytrees such as :class:`.Node`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
from torch import Tensor
from torch.utils import _pytree as pytree

__all__ = ["leaf_paths", "partition", "combine", "stack"]


def leaf_paths(tree: Any) -> list[str]:
    """
    Key strings of all leaves, e.g. ``".positions"`` or ``".extra['a']"``.

    Parameters
    ----------
    tree : Any
        Pytree to inspect.

    Returns
    -------
    list[str]
        One key string per leaf, in flattening order.

    Examples
    --------
    >>> import torch
    >>> from tad_mctc.tree import leaf_paths
    >>> leaf_paths({"a": torch.zeros(1), "b": [torch.zeros(2)]})
    ["['a']", "['b'][0]"]
    """
    leaves, _ = pytree.tree_flatten_with_path(tree)
    return [pytree.keystr(path) for path, _ in leaves]


def partition(
    tree: Any, select: Callable[[str, Tensor], bool]
) -> tuple[dict[str, Tensor], Any]:
    """
    Select tensor leaves by path.

    Parameters
    ----------
    tree : Any
        Pytree to select from.
    select : Callable[[str, Tensor], bool]
        Predicate on the key string and the tensor of a leaf.

    Returns
    -------
    tuple[dict[str, Tensor], Any]
        ``(params, tree)``; ``params`` maps key strings to the selected
        tensors.
    """
    leaves, _ = pytree.tree_flatten_with_path(tree)
    params: dict[str, Tensor] = {}
    for path, leaf in leaves:
        key = pytree.keystr(path)
        if isinstance(leaf, Tensor) and select(key, leaf):
            params[key] = leaf
    return params, tree


def combine(params: Mapping[str, Tensor], tree: Any) -> Any:
    """
    Return ``tree`` with the leaves named in ``params`` replaced.

    Parameters
    ----------
    params : Mapping[str, Tensor]
        Replacement tensors by key string (see :func:`leaf_paths`).
    tree : Any
        Pytree whose leaves are replaced.

    Returns
    -------
    Any
        The new tree; leaves not named in ``params`` are unchanged.

    Raises
    ------
    KeyError
        If a key in ``params`` matches no leaf.
    """
    leaves, spec = pytree.tree_flatten_with_path(tree)
    keys = [pytree.keystr(path) for path, _ in leaves]
    unknown = sorted(set(params) - set(keys))
    if unknown:
        raise KeyError(f"No leaves with paths {unknown} in the tree.")
    new = [params.get(k, leaf) for k, (_, leaf) in zip(keys, leaves)]
    return pytree.tree_unflatten(new, spec)


def stack(trees: Sequence[Any]) -> Any:
    """
    Stack trees of identical structure along a new leading dimension.

    Parameters
    ----------
    trees : Sequence[Any]
        Pytrees with identical structure.

    Returns
    -------
    Any
        Tree of the same structure with every leaf stacked.

    Raises
    ------
    ValueError
        If no tree is given or the structures differ.
    """
    if len(trees) == 0:
        raise ValueError("`stack` needs at least one tree.")
    flat = [pytree.tree_flatten(t) for t in trees]
    spec0 = flat[0][1]
    for i, (_, spec) in enumerate(flat[1:], start=1):
        if spec != spec0:
            raise ValueError(
                f"Tree {i} has a different structure than tree 0:\n"
                f"{spec}\nvs\n{spec0}"
            )
    stacked = [
        torch.stack(list(leaves)) for leaves in zip(*(f[0] for f in flat))
    ]
    return pytree.tree_unflatten(stacked, spec0)
