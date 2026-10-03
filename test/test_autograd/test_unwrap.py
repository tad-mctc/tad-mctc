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
Test unwrapping of function-transformed tensors.
"""

from __future__ import annotations

import pytest
import torch
from torch.func import jacrev, vmap

import tad_mctc.autograd.unwrap as unwrap
from tad_mctc.autograd import unwrap_gradtracking

ft = torch._C._functorch  # pylint: disable=protected-access


def test_plain_tensor_unchanged() -> None:
    x = torch.arange(3.0)
    assert unwrap_gradtracking(x) is x


def test_constant_inside_jacrev() -> None:
    """A constant built inside `jacrev(jacrev(...))` is wrapped twice; both
    grad-tracking layers are removed."""
    seen: list[torch.Tensor] = []

    def f(x: torch.Tensor) -> torch.Tensor:
        const = torch.full_like(x, 2.0)
        assert ft.is_gradtrackingtensor(const)
        seen.append(unwrap_gradtracking(const))
        return (const * x**3).sum()

    jacrev(jacrev(f))(torch.arange(3.0))

    assert not ft.is_functorch_wrapped_tensor(seen[0])
    assert torch.equal(seen[0], torch.full((3,), 2.0))


def test_stops_at_vmap() -> None:
    """Under `jacrev(vmap(...))`, the argument's outer layer is the `vmap`
    one, where the unwrapping stops."""
    seen: list[torch.Tensor] = []

    def f(x: torch.Tensor) -> torch.Tensor:
        seen.append(unwrap_gradtracking(x))
        return x.sin()

    jacrev(lambda x: vmap(f)(x).sum())(torch.ones(2, 3))

    assert not ft.is_gradtrackingtensor(seen[0])
    assert ft.is_batchedtensor(seen[0])


def test_unchanged_while_compiling(monkeypatch: pytest.MonkeyPatch) -> None:
    """While tracing, the tensor is returned as is."""
    monkeypatch.setattr(unwrap, "is_compiling", lambda: True)

    x = torch.ones(3)
    assert unwrap_gradtracking(x) is x
