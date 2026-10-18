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
Test the pairs of atoms and periodic images within per-atom radii:
`periodic_pairs_within_radii`, against a brute-force sum over all shifts.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch.func import jacrev, vmap

from tad_mctc.io.structure import Structure, pack_structures
from tad_mctc.neighbor import (
    build_periodic_shifts,
    periodic_pairs_within_radii,
)
from tad_mctc.typing import DD, Tensor

from ..conftest import DEVICE

dd: DD = {"device": DEVICE, "dtype": torch.float64}

Key = tuple[int, int, tuple[int, int, int]]


def _key(i: int, j: int, n: list[int]) -> Key:
    """One key for both directions of a pair, (i, j, n) and (j, i, -n)"""
    forward = (i, j, tuple(n))
    backward = (j, i, tuple(-x for x in n))
    return min(forward, backward)  # type: ignore[return-value]


def _pair_set(i: Tensor, j: Tensor, shift: Tensor) -> set[Key]:
    return {
        _key(a, b, n) for a, b, n in zip(i.tolist(), j.tolist(), shift.tolist())
    }


def _brute_force(structure: Structure, radii: Tensor) -> set[Key]:
    """All pairs within the radii, from all shifts of a generous table"""
    lattice, periodic = structure.lattice, structure.periodic
    assert lattice is not None and periodic is not None
    nat = structure.numbers.shape[-1]
    numbers = structure.numbers.reshape(-1, nat)
    positions = structure.positions.reshape(-1, nat, 3)
    lattices = lattice.reshape(-1, 3, 3)
    masks = periodic.reshape(-1, 3).expand(len(lattices), 3)
    r = radii.reshape(-1, nat)

    pairs: set[Key] = set()
    for b in range(len(numbers)):
        real = (numbers[b] != 0).nonzero().flatten().tolist()
        if not real:
            continue
        pos = positions[b, real]
        span = torch.cdist(pos, pos).max().item()
        cutoff = math.sqrt(2.0) * r[b, real].max().item() + span
        shifts = build_periodic_shifts(lattices[b], masks[b], cutoff).shifts
        shifts = shifts[(shifts[:, ~masks[b]] == 0).all(-1)]
        trans = shifts.to(lattices.dtype) @ lattices[b]
        for i in real:
            for j in real:
                vec = positions[b, j] + trans - positions[b, i]
                within = vec.pow(2).sum(-1) <= r[b, i] ** 2 + r[b, j] ** 2
                for n in shifts[within].tolist():
                    if i == j and not any(n):
                        continue
                    pairs.add(_key(b * nat + i, b * nat + j, n))
    return pairs


def _cell(
    nat: int,
    seed: int,
    lattice: Tensor,
    periodic: list[bool] | None = None,
    radius: float = 3.0,
) -> tuple[Structure, Tensor]:
    """Random atoms in (and somewhat outside) a cell, random radii"""
    gen = torch.Generator().manual_seed(seed)
    fractional = 1.4 * torch.rand(
        nat, 3, generator=gen, dtype=torch.float64, device="cpu"
    )
    radii = radius * (
        0.3 + torch.rand(nat, generator=gen, dtype=torch.float64, device="cpu")
    )
    structure = Structure(
        numbers=torch.randint(1, 10, (nat,), generator=gen, device="cpu"),
        positions=(fractional - 0.2) @ lattice.cpu(),
        lattice=lattice.cpu(),
        periodic=(
            None if periodic is None else torch.tensor(periodic, device="cpu")
        ),
    ).to(**dd)
    return structure, radii.to(**dd)


TRICLINIC = torch.tensor(
    [[5.5, 0.0, 0.0], [1.0, 6.0, 0.0], [0.5, -0.7, 6.5]], dtype=torch.float64
)


def _check(structure: Structure, radii: Tensor) -> None:
    i, j, shift = periodic_pairs_within_radii(structure, radii)
    assert i.dtype == j.dtype == shift.dtype == torch.long
    assert shift.shape == (len(i), 3)
    pairs = _pair_set(i, j, shift)
    assert len(pairs) == len(i), "a pair occurs in both directions"
    assert pairs == _brute_force(structure, radii)


def test_triclinic() -> None:
    """Atoms partly outside a triclinic cell equal the brute force."""
    _check(*_cell(12, seed=0, lattice=TRICLINIC))


def test_one_atom() -> None:
    """A single atom has only its own images, each pair n/-n once."""
    structure, _ = _cell(1, seed=1, lattice=TRICLINIC)
    radii = torch.tensor([9.0], **dd)
    i, j, shift = periodic_pairs_within_radii(structure, radii)
    assert len(i) > 10
    assert bool((i == j).all())
    assert bool(shift.abs().sum(-1).gt(0).all())
    _check(structure, radii)


@pytest.mark.parametrize(
    "periodic",
    [[True, True, False], [True, False, False], [False, True, True]],
)
def test_low_dimensional(periodic: list[bool]) -> None:
    """No images along a short non-periodic lattice vector."""
    lattice = TRICLINIC.clone()
    for axis, p in enumerate(periodic):
        if not p:
            lattice[axis] *= 0.5  # images well within the radii
    structure, radii = _cell(6, seed=2, lattice=lattice, periodic=periodic)
    i, j, shift = periodic_pairs_within_radii(structure, radii)
    assert bool((shift[:, ~torch.tensor(periodic)] == 0).all())
    _check(structure, radii)


def test_batch() -> None:
    """A padded batch of cells numbers its atoms flattened."""
    s1, r1 = _cell(5, seed=3, lattice=TRICLINIC)
    s2, r2 = _cell(8, seed=4, lattice=1.3 * TRICLINIC.flip(0))
    structure = pack_structures([s1, s2])
    radii = torch.stack([torch.cat([r1, torch.zeros(3, **dd)]), r2])
    i, j, _ = periodic_pairs_within_radii(structure, radii)
    assert bool(((i // 8) == (j // 8)).all())
    assert not bool(((i % 8) >= 5)[i < 8].any())  # no padding atoms
    _check(structure, radii)


def test_no_pairs() -> None:
    """Zero radii give empty pairs of the right shapes."""
    structure, radii = _cell(4, seed=5, lattice=TRICLINIC)
    i, j, shift = periodic_pairs_within_radii(
        structure, torch.zeros_like(radii)
    )
    assert i.shape == j.shape == (0,)
    assert shift.shape == (0, 3)


def test_inside_jacrev() -> None:
    """Under `jacrev`, the pairs are found on the values underneath."""
    structure, radii = _cell(6, seed=6, lattice=TRICLINIC)
    expected = _pair_set(*periodic_pairs_within_radii(structure, radii))

    found = []

    def f(positions: Tensor) -> Tensor:
        cell = structure.replace(positions=positions)
        tracked_radii = radii + 0.0 * positions.sum()  # wrapped by jacrev
        found.append(periodic_pairs_within_radii(cell, tracked_radii))
        return positions.sum()

    jacrev(f)(structure.positions)
    # read after `jacrev`: in older PyTorch (2.7), the values of tensors built
    # inside it cannot be read there
    assert [_pair_set(*pairs) for pairs in found] == [expected]


def test_errors() -> None:
    """Molecules, wrong radii and `vmap` are rejected."""
    structure, radii = _cell(4, seed=7, lattice=TRICLINIC)
    molecule = Structure(
        numbers=structure.numbers, positions=structure.positions
    )
    with pytest.raises(ValueError, match="not periodic"):
        periodic_pairs_within_radii(molecule, radii)
    with pytest.raises(ValueError, match="one radius per atom"):
        periodic_pairs_within_radii(structure, radii[:-1])
    with pytest.raises(ValueError, match="non-negative"):
        periodic_pairs_within_radii(structure, -radii)

    batch = pack_structures([structure, structure])
    with pytest.raises(RuntimeError, match="vmap"):
        vmap(lambda s: periodic_pairs_within_radii(s, radii)[0])(batch)
