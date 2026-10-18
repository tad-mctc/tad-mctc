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
Test the k-point container `KPoints`, `pack_kpoints` and the conversion
between Cartesian and fractional coordinates.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch.func import jacrev, vmap

from tad_mctc.kpoints import (
    KPoints,
    pack_kpoints,
    reciprocal_lattice,
    to_cartesian,
    to_fractional,
)
from tad_mctc.typing import Tensor

from ..utils import compile_fullgraph
from .samples import LATTICE, dd


def _points(nk: int, seed: int) -> Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.rand(
        nk, 3, generator=gen, dtype=torch.float64, device="cpu"
    ).to(**dd)


def test_defaults() -> None:
    """Absent weights are uniform, an absent mask is all true."""
    kpts = KPoints(points=_points(4, seed=0))
    assert torch.equal(kpts.weights, torch.full((4,), 0.25, **dd))
    assert kpts.mask.dtype == torch.bool and bool(kpts.mask.all())
    assert kpts.nkpts == 4

    batch = KPoints(points=_points(6, seed=0).reshape(2, 3, 3))
    assert torch.allclose(batch.weights, torch.full((2, 3), 1 / 3, **dd))


def test_weights_cast() -> None:
    """Weights take the dtype of the points, also in `.to`."""
    kpts = KPoints(points=_points(2, seed=1), weights=torch.ones(2))
    assert kpts.weights.dtype == torch.float64
    single = kpts.type(torch.float32)
    assert single.points.dtype == torch.float32
    assert single.weights.dtype == torch.float32
    assert single.mask.dtype == torch.bool


def test_invalid() -> None:
    """Wrong shapes and dtypes are rejected."""
    with pytest.raises(RuntimeError, match="points"):
        KPoints(points=torch.zeros(4, 2, **dd))
    with pytest.raises(RuntimeError, match="floating-point"):
        KPoints(points=torch.zeros(4, 3, dtype=torch.long))
    with pytest.raises(RuntimeError, match="weights"):
        KPoints(points=_points(4, seed=2), weights=torch.ones(3, **dd))
    with pytest.raises(RuntimeError, match="mask"):
        KPoints(points=_points(4, seed=2), mask=torch.ones(4, **dd))


def test_pack() -> None:
    """Packing pads the points, weights and mask of the smaller sets."""
    small = KPoints(points=_points(2, seed=3))
    large = KPoints(points=_points(5, seed=4), weights=torch.arange(5.0, **dd))
    batch = pack_kpoints([small, large])

    assert batch.points.shape == (2, 5, 3)
    assert torch.equal(batch.points[0, :2], small.points)
    assert torch.equal(batch.points[0, 2:], torch.zeros(3, 3, **dd))
    assert torch.equal(
        batch.weights[0], torch.tensor([0.5, 0.5, 0, 0, 0], **dd)
    )
    assert torch.equal(batch.weights[1], torch.arange(5.0, **dd))
    expected = torch.tensor(
        [[1, 1, 0, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool
    )
    assert torch.equal(batch.mask.cpu(), expected.cpu())

    with pytest.raises(ValueError, match="empty"):
        pack_kpoints([])
    with pytest.raises(ValueError, match="unbatched"):
        pack_kpoints([batch])


def test_vmap() -> None:
    """A batched set passes through `vmap` like a structure."""
    batch = pack_kpoints(
        [KPoints(points=_points(2, seed=5)), KPoints(points=_points(3, seed=6))]
    )

    def masked_sum(kpts: KPoints) -> Tensor:
        return (kpts.weights * kpts.mask * kpts.points.sum(-1)).sum()

    result = vmap(masked_sum)(batch)
    expected = torch.stack(
        [
            masked_sum(KPoints(points=p, weights=w, mask=m))
            for p, w, m in zip(batch.points, batch.weights, batch.mask)
        ]
    )
    assert torch.allclose(result, expected)


def test_conversions() -> None:
    """The reciprocal lattice is dual to the lattice; conversions invert."""
    b = reciprocal_lattice(LATTICE)
    assert torch.allclose(LATTICE @ b.mT, 2 * math.pi * torch.eye(3, **dd))

    fractional = _points(5, seed=7)
    cartesian = to_cartesian(fractional, LATTICE)
    assert torch.allclose(to_fractional(cartesian, LATTICE), fractional)

    # the Bloch phase k.T of a lattice translation T = n A is 2 pi kappa.n
    n = torch.tensor([[1.0, -2.0, 3.0]], **dd)
    phase = (cartesian @ (n @ LATTICE).mT).squeeze(-1)
    assert torch.allclose(phase, 2 * math.pi * (fractional @ n.mT).squeeze(-1))

    batch = torch.stack([LATTICE, 1.2 * LATTICE])
    assert torch.allclose(
        to_cartesian(fractional.expand(2, -1, -1), batch)[1],
        to_cartesian(fractional, 1.2 * LATTICE),
    )


def test_conversions_jacrev() -> None:
    """The conversions are differentiable w.r.t. the lattice."""
    fractional = _points(3, seed=8)
    jac = jacrev(lambda lat: to_cartesian(fractional, lat))(LATTICE)
    eps = 1e-6
    lat = LATTICE.clone()
    lat[1, 2] += eps
    numerical = (
        to_cartesian(fractional, lat) - to_cartesian(fractional, LATTICE)
    ) / eps
    assert torch.allclose(jac[..., 1, 2], numerical, atol=1e-5)


def test_compile() -> None:
    """Consuming a set compiles as one graph and equals eager."""
    kpts = KPoints(points=_points(4, seed=9))
    n = torch.tensor([[1.0, 0.0, -1.0], [0.0, 2.0, 1.0]], **dd)

    def phases(points: Tensor, lattice: Tensor) -> Tensor:
        k = KPoints(points=points)
        return torch.cos(to_fractional(k.points, lattice) @ n.mT)

    compiled = compile_fullgraph(phases)
    assert torch.allclose(
        compiled(kpts.points, LATTICE), phases(kpts.points, LATTICE)
    )
