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
Test the pairs of atoms within per-atom radii: `pairs_within_radii`.
"""

from __future__ import annotations

import pytest
import torch

from tad_mctc.io.structure import Structure, pack_structures
from tad_mctc.neighbor import pairs_within_radii
from tad_mctc.typing import DD, Tensor

from ..conftest import DEVICE

dd: DD = {"device": DEVICE, "dtype": torch.float64}


def _molecule(nat: int, seed: int) -> tuple[Structure, Tensor]:
    """Random atoms in a box and random radii between 0.5 and 3"""
    gen = torch.Generator().manual_seed(seed)
    positions = 12.0 * torch.rand(
        nat, 3, generator=gen, dtype=torch.float64, device="cpu"
    )
    radii = 0.5 + 2.5 * torch.rand(
        nat, generator=gen, dtype=torch.float64, device="cpu"
    )
    numbers = torch.randint(1, 10, (nat,), generator=gen, device="cpu")
    structure = Structure(numbers=numbers, positions=positions).to(**dd)
    return structure, radii.to(**dd)


def _pair_set(i: Tensor, j: Tensor) -> set[tuple[int, int]]:
    return {(min(a, b), max(a, b)) for a, b in zip(i.tolist(), j.tolist())}


def _dense_reference(
    structure: Structure, radii: Tensor
) -> set[tuple[int, int]]:
    """All pairs `i < j` of real atoms within their radii, from all distances"""
    real = (structure.numbers != 0).reshape(-1)
    positions = structure.positions.reshape(-1, 3)
    r = radii.reshape(-1)
    nat = structure.numbers.shape[-1]

    distance2 = torch.cdist(positions, positions).pow(2)
    within = distance2 <= r[:, None] ** 2 + r[None, :] ** 2
    same_system = torch.arange(len(r), device=r.device) // nat
    within &= same_system[:, None] == same_system[None, :]
    within &= real[:, None] & real[None, :]
    i, j = torch.nonzero(torch.triu(within, diagonal=1), as_tuple=True)
    return _pair_set(i, j)


def test_single() -> None:
    """The pairs of a molecule equal the dense reference, each once."""
    structure, radii = _molecule(200, seed=0)
    i, j = pairs_within_radii(structure, radii)

    assert i.dtype == j.dtype == torch.long
    assert bool((i != j).all())
    assert len(_pair_set(i, j)) == len(i)
    assert _pair_set(i, j) == _dense_reference(structure, radii)


def test_distance_kernel() -> None:
    """A forced distance kernel gives the same pairs."""
    structure, radii = _molecule(100, seed=1)
    expected = _pair_set(*pairs_within_radii(structure, radii))
    pairs = pairs_within_radii(structure, radii, distance_kernel="baddbmm")
    assert _pair_set(*pairs) == expected


def test_batch() -> None:
    """A padded batch numbers its atoms flattened; padding has no pairs."""
    (s1, r1), (s2, r2) = _molecule(60, seed=2), _molecule(35, seed=3)
    batch = pack_structures([s1, s2])
    radii = torch.zeros(batch.numbers.shape, **dd)
    radii[0], radii[1, :35] = r1, r2
    radii[1, 35:] = 100.0  # ignored for padding

    i, j = pairs_within_radii(batch, radii)
    assert _pair_set(i, j) == _dense_reference(batch, radii)
    assert bool((i // 60 == j // 60).all())


def test_no_pairs() -> None:
    """Zero radii, a single atom or only padding give no pairs."""
    structure, radii = _molecule(10, seed=4)
    for s, r in [
        (structure, torch.zeros_like(radii)),
        (
            structure.replace(
                numbers=structure.numbers[:1], positions=structure.positions[:1]
            ),
            radii[:1],
        ),
        (structure.replace(numbers=torch.zeros_like(structure.numbers)), radii),
    ]:
        i, j = pairs_within_radii(s, r)
        assert i.shape == j.shape == (0,)


def test_invalid() -> None:
    """Wrong shapes, negative or infinite radii and lattices are rejected."""
    structure, radii = _molecule(10, seed=5)
    with pytest.raises(ValueError, match="one radius per atom"):
        pairs_within_radii(structure, radii[:-1])
    with pytest.raises(ValueError, match="finite and non-negative"):
        pairs_within_radii(structure, -radii)
    with pytest.raises(ValueError, match="finite and non-negative"):
        pairs_within_radii(structure, radii / 0.0)

    periodic = structure.replace(lattice=20.0 * torch.eye(3, **dd))
    with pytest.raises(ValueError, match="Periodic"):
        pairs_within_radii(periodic, radii)


def test_transforms() -> None:
    """Inside `jacrev`, the pairs are the same; `vmap` is refused."""
    structure, radii = _molecule(50, seed=6)
    expected = _pair_set(*pairs_within_radii(structure, radii))
    pairs = []

    def func(positions: Tensor) -> Tensor:
        moved = structure.replace(positions=positions)
        tracked_radii = radii + 0.0 * positions.sum()  # wrapped by jacrev
        i, j = pairs_within_radii(moved, tracked_radii)
        pairs.append((i, j))
        return (positions[i] - positions[j]).pow(2).sum()

    jacobian = torch.func.jacrev(func)(structure.positions)
    assert jacobian.shape == structure.positions.shape
    # read after `jacrev`: in older PyTorch (2.7), the values of tensors built
    # outside the transforms cannot be read inside them
    assert _pair_set(*pairs[0]) == expected

    batched = structure.positions.expand(2, -1, -1)
    with pytest.raises(RuntimeError, match="vmap"):
        torch.func.vmap(func)(batched)
