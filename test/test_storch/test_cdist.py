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
Test cdist safeop version.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tad_mctc import storch
from tad_mctc.convert import numpy_to_tensor
from tad_mctc.typing import DD

from ..conftest import DEVICE
from ..utils import (
    DYNAMO_SUPPORTED,
    DYNAMO_UNSUPPORTED_REASON,
    compile_fullgraph,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_all(dtype: torch.dtype) -> None:
    """
    The three Euclidean implementations agree on the distance between two
    different vectors.

    `storch.cdist` is the direct expansion, so those two agree exactly. The
    quadratic expansion differs from them by at most 11 `eps` over 3000
    random inputs of this size, on CPU and CUDA, in either dtype. The
    distance of a vector to itself is left out here, see
    `test_distance_to_itself`.
    """
    dd: DD = {"device": DEVICE, "dtype": dtype}
    tol = 100 * torch.finfo(dtype).eps

    # Own generator, so that the input does not depend on the tests run
    # before this one.
    x = numpy_to_tensor(
        np.random.default_rng(0).standard_normal((2, 3, 4)), **dd
    )

    d1 = storch.cdist(x)
    d2 = storch.distance.cdist_direct_expansion(x, x, p=2)
    d3 = storch.distance.euclidean_dist_quadratic_expansion(x, x)

    different = ~torch.eye(3, dtype=torch.bool, device=d1.device)
    different = different.expand_as(d1)
    d1, d2, d3 = d1[different].cpu(), d2[different].cpu(), d3[different].cpu()

    assert torch.equal(d1, d2)
    assert pytest.approx(d2, abs=tol) == d3


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_distance_to_itself(dtype: torch.dtype) -> None:
    """
    The distance of a vector to itself is not zero but `sqrt(eps)`: the
    squared distance is clamped to `eps` before the root, to keep the
    gradient finite.
    """
    dd: DD = {"device": DEVICE, "dtype": dtype}
    sqrt_eps = torch.tensor(torch.finfo(dtype).eps, **dd).sqrt()

    x = numpy_to_tensor(
        np.random.default_rng(1).standard_normal((8, 16, 4)), **dd
    )

    for distances in (
        storch.cdist(x),
        storch.distance.cdist_direct_expansion(x, x, p=2),
    ):
        to_itself = torch.diagonal(distances, dim1=-2, dim2=-1)
        assert bool((to_itself == sqrt_eps).all())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_quadratic_expansion_distance_to_itself(dtype: torch.dtype) -> None:
    """
    In the quadratic expansion, the distance of a vector to itself is the
    square root of a rounding residual of `|x|^2`, clamped to at least
    `eps`. The residual depends on the summation order, which differs
    between devices (on CUDA, up to `1.95 * sqrt(eps * |x|^2)`), so only its
    scale is checked: positive and at most `4 * sqrt(eps * max(1, |x|^2))`.
    """
    dd: DD = {"device": DEVICE, "dtype": dtype}
    eps = torch.finfo(dtype).eps

    x = numpy_to_tensor(
        np.random.default_rng(1).standard_normal((8, 16, 4)), **dd
    )
    bound = 4.0 * torch.sqrt(eps * torch.clamp((x * x).sum(-1), min=1.0))

    distances = storch.distance.euclidean_dist_quadratic_expansion(x, x)
    to_itself = torch.diagonal(distances, dim1=-2, dim2=-1)
    assert bool((to_itself > 0).all())
    assert bool((to_itself <= bound).all())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("p", [2, 3, 4, 5])
def test_ps(dtype: torch.dtype, p: int) -> None:
    """
    `storch.cdist` agrees with `torch.cdist` for every power `p`.

    The tolerance is relative to the dtype's precision: over 3000 random
    inputs, on CPU and CUDA, the two differ by at most 8 `eps`.
    """
    dd: DD = {"device": DEVICE, "dtype": dtype}
    tol = 100 * torch.finfo(dtype).eps

    rng = np.random.default_rng(2)
    x = numpy_to_tensor(rng.standard_normal((2, 4, 5)), **dd)
    y = numpy_to_tensor(rng.standard_normal((2, 4, 5)), **dd)

    d1 = storch.cdist(x, y, p=p)
    d2 = torch.cdist(x, y, p=p)

    assert pytest.approx(d1.cpu(), abs=tol) == d2.cpu()


@pytest.mark.skipif(not DYNAMO_SUPPORTED, reason=DYNAMO_UNSUPPORTED_REASON)
def test_cdist_torch_compile_fullgraph() -> None:
    """
    ``storch.cdist`` must trace with ``torch.compile(fullgraph=True)``,
    since every dense path built on it (e.g. ``properties.enn``) does.
    """
    torch._dynamo.reset()

    dd: DD = {"device": DEVICE, "dtype": torch.float64}
    x = numpy_to_tensor(
        np.random.default_rng(3).standard_normal((2, 4, 3)), **dd
    )

    def f(x: torch.Tensor) -> torch.Tensor:
        return storch.cdist(x)

    compiled = compile_fullgraph(f)

    eager_value = f(x)
    compiled_value = compiled(x)

    # The differences of a vector to itself are exactly zero in either, so
    # only the other distances can round differently in a fused kernel.
    assert pytest.approx(eager_value.cpu(), abs=1e-12) == compiled_value.cpu()
    assert torch.equal(
        torch.diagonal(eager_value, dim1=-2, dim2=-1),
        torch.diagonal(compiled_value, dim1=-2, dim2=-1),
    )


@pytest.mark.skipif(not DYNAMO_SUPPORTED, reason=DYNAMO_UNSUPPORTED_REASON)
def test_quadratic_expansion_torch_compile_fullgraph() -> None:
    """
    The quadratic expansion calls ``storch.safe_sqrt``. A domain check there
    that branched on a tensor value would be rejected by
    ``torch.compile(fullgraph=True)`` (Dynamo) as data-dependent control
    flow.
    """
    torch._dynamo.reset()

    dd: DD = {"device": DEVICE, "dtype": torch.float64}
    x = numpy_to_tensor(
        np.random.default_rng(3).standard_normal((2, 4, 3)), **dd
    )

    def f(x: torch.Tensor) -> torch.Tensor:
        return storch.distance.euclidean_dist_quadratic_expansion(x, x)

    compiled = compile_fullgraph(f)

    eager_value = f(x)
    compiled_value = compiled(x)

    # The diagonal is the square root of a rounding residual, which a fused
    # (compiled) kernel rounds differently than the eager one. Only the
    # other distances have to agree tightly.
    diagonal = torch.eye(x.shape[-2], dtype=torch.bool, device=x.device)
    off = ~diagonal.expand_as(eager_value)
    assert pytest.approx(eager_value[off].cpu(), abs=1e-12) == (
        compiled_value[off].cpu()
    )
    assert (eager_value[~off] < 1e-6).all()
    assert (compiled_value[~off] < 1e-6).all()
