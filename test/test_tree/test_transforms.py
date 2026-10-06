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
Test `Node` under `torch.func` transforms and `torch.compile`.
"""

from __future__ import annotations

import importlib.util
from typing import cast

import pytest
import torch
from torch.func import jacfwd, jacrev, vmap
from torch.utils import _pytree as pytree

from tad_mctc.tools.testing import requires_compile
from tad_mctc.tree import combine, partition, stack

from ..conftest import DEVICE
from ..utils import compile_fullgraph
from .samples import Sub

DTYPE = torch.float64


def _sub(seed: int = 0, **kwargs) -> Sub:  # type: ignore[no-untyped-def]
    gen = torch.Generator().manual_seed(seed)
    return Sub(
        numbers=torch.tensor([1, 1, 8]),
        positions=torch.rand(3, 3, generator=gen, dtype=DTYPE, device="cpu").to(
            DEVICE
        ),
        **kwargs,
    )


def _energy(node: Sub) -> torch.Tensor:
    diff = node.positions[:, None, :] - node.positions[None, :, :]
    dist2 = (diff**2).sum(-1) + torch.eye(
        3, dtype=node.dtype, device=node.device
    )
    return (node.numbers.to(node.dtype) / dist2).sum() * node.cutoff


def _shifted(node: Sub, x: torch.Tensor) -> torch.Tensor:
    return (node.positions * x).sum() + node.numbers.sum()


def test_vmap_shared_node() -> None:
    node = _sub()
    batch = torch.rand(4, 3, 3, dtype=DTYPE)

    batched = vmap(_shifted, in_dims=(None, 0))(node, batch)
    looped = torch.stack([_shifted(node, x) for x in batch])
    assert torch.allclose(batched, looped)


def test_vmap_stacked_node() -> None:
    n1, n2 = _sub(1), _sub(2)

    batched = vmap(_energy, in_dims=(0,))(stack([n1, n2]))
    assert torch.allclose(batched[0], _energy(n1))
    assert torch.allclose(batched[1], _energy(n2))


def test_vmap_absent_optional_field() -> None:
    n1, n2 = _sub(1), _sub(2)
    assert n1.charge is None

    batched = vmap(_energy, in_dims=0)(stack([n1, n2]))
    assert torch.allclose(batched, torch.stack([_energy(n1), _energy(n2)]))


def test_construction_inside_vmap() -> None:
    numbers = torch.tensor([[1, 1, 8], [1, 1, 8]])
    positions = torch.rand(2, 3, 3, dtype=DTYPE)

    def build(n: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        return _energy(Sub(numbers=n, positions=pos))

    batched = vmap(build)(numbers, positions)
    looped = torch.stack([build(n, p) for n, p in zip(numbers, positions)])
    assert torch.allclose(batched, looped)


def _reference_gradient(node: Sub) -> torch.Tensor:
    pos = node.positions.clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(_energy(node.replace(positions=pos)), pos)
    return grad


def test_jacrev_through_partition_combine() -> None:
    node = _sub()
    params, rest = partition(node, lambda p, t: p == ".positions")

    grads = cast(
        dict[str, torch.Tensor],
        jacrev(lambda p: _energy(combine(p, rest)))(params),
    )
    assert list(grads) == [".positions"]
    assert torch.allclose(grads[".positions"], _reference_gradient(node))


def test_jacfwd_through_partition_combine() -> None:
    node = _sub()
    params, rest = partition(node, lambda p, t: p == ".positions")

    grads = cast(
        dict[str, torch.Tensor],
        jacfwd(lambda p: _energy(combine(p, rest)))(params),
    )
    assert torch.allclose(grads[".positions"], _reference_gradient(node))


@requires_compile
def test_compile_fullgraph_and_no_recompile() -> None:
    from torch._dynamo.testing import CompileCounter

    counter = CompileCounter()
    compiled = torch.compile(_energy, backend=counter, fullgraph=True)

    n1, n2 = _sub(1), _sub(2)
    assert torch.allclose(compiled(n1), _energy(n1))
    assert torch.allclose(compiled(n2), _energy(n2))
    assert counter.frame_count == 1


@requires_compile
def test_to_inside_compile() -> None:
    def fn(node: Sub) -> torch.Tensor:
        return _energy(node.to(dtype=torch.float32))

    node = _sub()
    compiled = compile_fullgraph(fn)
    assert torch.allclose(compiled(node), fn(node))


@requires_compile
def test_construction_inside_compile() -> None:
    def fn(n: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        return _energy(Sub(numbers=n, positions=pos))

    node = _sub()
    compiled = compile_fullgraph(fn)
    assert torch.allclose(
        compiled(node.numbers, node.positions), fn(node.numbers, node.positions)
    )


@pytest.mark.skipif(
    importlib.util.find_spec("optree") is None, reason="optree not installed"
)
def test_cxx_pytree() -> None:
    from torch.utils import _cxx_pytree

    node = _sub(charge=torch.zeros(3, dtype=DTYPE))
    cxx_leaves, _ = _cxx_pytree.tree_flatten(node)
    py_leaves, _ = pytree.tree_flatten(node)
    assert len(cxx_leaves) == len(py_leaves)
    assert all(a is b for a, b in zip(cxx_leaves, py_leaves))


@requires_compile
def test_replace_inside_compile() -> None:
    def fn(node: Sub) -> torch.Tensor:
        return _energy(node.replace(cutoff=10.0))

    node = _sub()
    compiled = compile_fullgraph(fn)
    assert torch.allclose(compiled(node), fn(node))
