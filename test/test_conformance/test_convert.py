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
Transform conformance of `tad_mctc.convert.symmetrize` on a 3x3 tensor.

Exclusions: none. The default ``force=False`` validates symmetry by reading
the tensor values, so it is only checked eagerly, on already symmetric
input; the transforms use ``force=True``.
"""

from __future__ import annotations

import pytest
import torch
from torch.func import vmap

from tad_mctc.autograd import (
    dgradcheck,
    dgradgradcheck,
    dgradgradgradcheck,
    jacfwd_matches_jacrev,
    no_vmap_fallback,
)
from tad_mctc.convert import symmetrize

from ..conftest import DEVICE
from ..utils import DYNAMO_SUPPORTED, DYNAMO_UNSUPPORTED_REASON


def _matrix(seed: int = 0, batch: int | None = None) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    shape = (3, 3) if batch is None else (batch, 3, 3)
    return torch.rand(*shape, generator=gen, dtype=torch.float64).to(DEVICE)


def _forced(x: torch.Tensor) -> torch.Tensor:
    return symmetrize(x, force=True)


def _forced_sin(x: torch.Tensor) -> torch.Tensor:
    # `symmetrize` is linear, so its own gradient does not depend on `x` and
    # the higher-order checks need a nonlinear function around it
    return _forced(x).sin()


def test_vmap_matches_loop() -> None:
    batch = _matrix(batch=2)

    with no_vmap_fallback():
        batched = vmap(_forced)(batch)

    looped = torch.stack([_forced(x) for x in batch])
    assert torch.allclose(batched, looped)


def test_vmap_without_force_on_symmetric_input() -> None:
    sym = _forced(_matrix(batch=2))

    # the symmetry check reads values, so it is not vmappable: check eagerly
    assert torch.allclose(symmetrize(sym), sym)


@pytest.mark.grad
def test_gradcheck_orders() -> None:
    assert dgradcheck(_forced, _matrix().requires_grad_())
    assert dgradgradcheck(_forced_sin, _matrix().requires_grad_())
    assert dgradgradgradcheck(_forced_sin, _matrix().requires_grad_())


def test_forward_matches_reverse() -> None:
    assert jacfwd_matches_jacrev(_forced, _matrix())


@pytest.mark.skipif(not DYNAMO_SUPPORTED, reason=DYNAMO_UNSUPPORTED_REASON)
def test_compile_matches_eager() -> None:
    x = _matrix()

    compiled = torch.compile(_forced, fullgraph=True)
    assert torch.allclose(compiled(x), _forced(x))
