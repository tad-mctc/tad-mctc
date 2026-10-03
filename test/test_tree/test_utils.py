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
Test the tree utilities: `leaf_paths`, `partition`, `combine` and `stack`.
"""

from __future__ import annotations

import pytest
import torch
from torch.utils import _pytree as pytree

from tad_mctc.tree import combine, leaf_paths, partition, stack

from .samples import Holder, Sub


def _sub(**kwargs) -> Sub:  # type: ignore[no-untyped-def]
    return Sub(
        numbers=torch.tensor([1, 1, 8]),
        positions=torch.rand(3, 3, dtype=torch.float64),
        **kwargs,
    )


def test_leaf_paths() -> None:
    holder = Holder(
        system=_sub(), extra={"a": torch.zeros(2, dtype=torch.float64)}
    )
    assert leaf_paths(holder) == [
        ".system.numbers",
        ".system.positions",
        ".extra['a']",
    ]


def test_partition_combine_round_trip() -> None:
    node = _sub()
    params, rest = partition(node, lambda p, t: p == ".positions")
    assert list(params) == [".positions"]
    assert params[".positions"] is node.positions

    new = combine(params, rest)
    for a, b in zip(pytree.tree_leaves(node), pytree.tree_leaves(new)):
        assert torch.equal(a, b)


def test_combine_replaces_one_leaf() -> None:
    node = _sub()
    positions = torch.zeros(3, 3, dtype=torch.float64)
    new = combine({".positions": positions}, node)
    assert new.positions is positions
    assert new.numbers is node.numbers


def test_combine_unknown_key() -> None:
    with pytest.raises(KeyError):
        combine({".nonexistent": torch.zeros(1)}, _sub())


def test_stack() -> None:
    n1 = _sub(rcov=abs)
    n2 = _sub(rcov=abs)
    new = stack([n1, n2])
    assert new.numbers.shape == (2, 3)
    assert new.positions.shape == (2, 3, 3)
    assert torch.equal(new.positions[1], n2.positions)
    assert new.rcov is abs
    assert new.label == "x" and new.cutoff == 25.0


def test_stack_errors() -> None:
    with pytest.raises(ValueError):
        stack([_sub(), _sub(charge=torch.zeros(3, dtype=torch.float64))])
    with pytest.raises(ValueError):
        stack([])
