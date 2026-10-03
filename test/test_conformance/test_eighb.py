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
Transform conformance of `tad_mctc.storch.eighb` with
``broadening_method=None``, ``"cond"`` and ``"lorn"``.

The checked outputs are the eigenvalues and the projector onto the lowest
eigenvector (``v[:, 0] v[:, 0]^T``), which is invariant to the sign of the
eigenvector. Inputs are random symmetric 4x4 matrices, a batch of two for
`vmap`.

Exclusions: the `torch.compile` check of the two broadening methods
(custom autograd function with ``setup_context``) is skipped on torch < 2.5,
where Dynamo cannot inline it (``TypeError: too many positional
arguments``); the ``None`` path (``torch.linalg.eigh``) is compiled on every
version.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import pytest
import torch
from torch.func import vmap

from tad_mctc._version import __tversion__
from tad_mctc.autograd import (
    dgradcheck,
    dgradgradcheck,
    dgradgradgradcheck,
    jacfwd_matches_jacrev,
    no_vmap_fallback,
)
from tad_mctc.storch import eighb

from ..conftest import DEVICE
from ..utils import DYNAMO_SUPPORTED, DYNAMO_UNSUPPORTED_REASON

Method = Literal["cond", "lorn"] | None

METHODS = [
    pytest.param(None, id="none"),
    pytest.param("cond", id="cond"),
    pytest.param("lorn", id="lorn"),
]


def _symmetric(seed: int = 0, batch: int | None = None) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    shape = (4, 4) if batch is None else (batch, 4, 4)
    a = torch.rand(*shape, generator=gen, dtype=torch.float64)
    return ((a + a.mT) / 2).to(DEVICE)


def _eigenvalues(method: Method) -> Callable[[torch.Tensor], torch.Tensor]:
    def f(a: torch.Tensor) -> torch.Tensor:
        # `eighb` reads one triangle only: symmetrize, so that perturbing a
        # single entry (finite differences) is a valid symmetric input
        a = (a + a.mT) / 2
        w, _ = eighb(a, broadening_method=method)
        return w

    return f


def _projector(method: Method) -> Callable[[torch.Tensor], torch.Tensor]:
    def f(a: torch.Tensor) -> torch.Tensor:
        a = (a + a.mT) / 2
        _, v = eighb(a, broadening_method=method)
        return v[..., :, 0, None] * v[..., None, :, 0]

    return f


@pytest.mark.parametrize("method", METHODS)
def test_vmap_matches_loop(method: Method) -> None:
    batch = _symmetric(batch=2)

    for f in (_eigenvalues(method), _projector(method)):
        with no_vmap_fallback():
            batched = vmap(f)(batch)

        looped = torch.stack([f(a) for a in batch])
        assert torch.allclose(batched, looped, atol=1e-10)


@pytest.mark.grad
@pytest.mark.parametrize("method", METHODS)
def test_gradcheck_orders(method: Method) -> None:
    for f in (_eigenvalues(method), _projector(method)):
        assert dgradcheck(f, _symmetric().requires_grad_())
        assert dgradgradcheck(f, _symmetric().requires_grad_())
        assert dgradgradgradcheck(f, _symmetric().requires_grad_())


@pytest.mark.parametrize("method", METHODS)
def test_forward_matches_reverse(method: Method) -> None:
    for f in (_eigenvalues(method), _projector(method)):
        assert jacfwd_matches_jacrev(f, _symmetric(), atol=1e-8, rtol=1e-6)


OLD_TORCH = pytest.mark.skipif(
    __tversion__ < (2, 5, 0),
    reason="Dynamo cannot inline the custom autograd function before 2.5",
)

COMPILE_METHODS = [
    pytest.param(None, id="none"),
    pytest.param("cond", id="cond", marks=OLD_TORCH),
    pytest.param("lorn", id="lorn", marks=OLD_TORCH),
]


@pytest.mark.skipif(not DYNAMO_SUPPORTED, reason=DYNAMO_UNSUPPORTED_REASON)
@pytest.mark.parametrize("method", COMPILE_METHODS)
def test_compile_matches_eager(method: Method) -> None:
    a = _symmetric()

    for f in (_eigenvalues(method), _projector(method)):
        compiled = torch.compile(f, fullgraph=True)
        assert torch.allclose(compiled(a), f(a), atol=1e-10)
