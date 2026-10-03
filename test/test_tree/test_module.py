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
Test `ModuleNode`: a small MLP held in a node tree.
"""

from __future__ import annotations

from typing import cast

import torch
from torch.func import jacrev, vmap
from torch.utils import _pytree as pytree

from tad_mctc.tree import ModuleNode, combine, partition


def _mlp(seed: int = 0) -> torch.nn.Module:
    torch.manual_seed(seed)
    return torch.nn.Sequential(
        torch.nn.Linear(3, 8),
        torch.nn.Tanh(),
        torch.nn.Linear(8, 1),
    ).to(torch.float64)


def test_call_matches_module() -> None:
    mlp = _mlp()
    x = torch.rand(5, 3, dtype=torch.float64)

    assert torch.equal(ModuleNode.from_module(mlp)(x), mlp(x))


def test_to_converts_params() -> None:
    mlp = _mlp()
    node = ModuleNode.from_module(mlp)
    new = node.to(dtype=torch.float32)

    assert new.module is mlp
    assert set(new.params) == set(node.params)
    assert all(p.dtype == torch.float32 for p in new.params.values())


def test_jacrev_through_partition_combine() -> None:
    mlp = _mlp()
    node = ModuleNode.from_module(mlp)
    x = torch.rand(5, 3, dtype=torch.float64)

    params, rest = partition(node, lambda p, t: p.startswith(".params"))
    assert len(params) == 4

    grads = cast(
        dict[str, torch.Tensor],
        jacrev(lambda p: combine(p, rest)(x).sum())(params),
    )

    reference = torch.autograd.grad(mlp(x).sum(), list(mlp.parameters()))
    assert len(grads) == len(reference)
    for g, r in zip(grads.values(), reference):
        assert torch.allclose(g, r)


def test_vmap_shared_node() -> None:
    node = ModuleNode.from_module(_mlp())
    batch = torch.rand(4, 5, 3, dtype=torch.float64)

    batched = vmap(lambda n, x: n(x), in_dims=(None, 0))(node, batch)
    looped = torch.stack([node(x) for x in batch])
    assert torch.allclose(batched, looped)


def test_tree_structure() -> None:
    mlp = _mlp()
    first = ModuleNode.from_module(mlp)
    second = ModuleNode.from_module(mlp)
    other = ModuleNode.from_module(_mlp(1))

    assert pytree.tree_structure(first) == pytree.tree_structure(second)
    assert pytree.tree_structure(first) != pytree.tree_structure(other)
