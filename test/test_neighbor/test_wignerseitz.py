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
Test the Wigner-Seitz cell against a line-by-line port of tblite's
`get_pairs` (`wignerseitz.f90`), and the mctc-lib order of the lattice
shifts.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch.func import vmap

from tad_mctc.autograd import no_vmap_fallback
from tad_mctc.data.structures import get_structure
from tad_mctc.neighbor import (
    lattice_shifts_mctclib,
    wignerseitz_cell,
    wrap_to_central_cell,
)
from tad_mctc.typing import DD, Tensor

from ..conftest import DEVICE

dd: DD = {"device": DEVICE, "dtype": torch.float64}

CELLS = [("x23", "CO2"), ("x23", "formamide"), ("ice10", "vi")]

PERIODIC = torch.tensor([True, True, True], device=DEVICE)


def _fortran_order(rep: tuple[int, int, int], origin: bool) -> list[list[int]]:
    """`get_lattice_points(lat, rep, origin, trans)` of mctc-lib"""
    out = []
    for ix in range(rep[0] + 1):
        for iy in range(rep[1] + 1):
            for iz in range(rep[2] + 1):
                if not origin and ix == iy == iz == 0:
                    continue
                for jx in range(1, -2 if ix > 0 else 0, -2):
                    for jy in range(1, -2 if iy > 0 else 0, -2):
                        for jz in range(1, -2 if iz > 0 else 0, -2):
                            out.append([ix * jx, iy * jy, iz * jz])
    return out


def _fortran_weights(positions: Tensor, lattice: Tensor) -> Tensor:
    """`new_wignerseitz_cell` of tblite, with Python loops"""
    thr = math.sqrt(torch.finfo(torch.float64).eps)
    tol = 0.01
    trans = torch.tensor(_fortran_order((1, 1, 1), True), **dd) @ lattice
    nat = positions.shape[0]
    weights = torch.zeros(nat, nat, len(trans), **dd)
    for i in range(nat):
        for j in range(nat):
            rij = positions[i] - positions[j]
            dist, index = [], []
            for itr in range(len(trans)):
                r2 = ((rij - trans[itr]) ** 2).sum().item()
                if r2 < thr:
                    continue
                dist.append(r2)
                index.append(itr)
            nearest = min(dist)
            images = [k for d, k in zip(dist, index) if d - nearest <= tol]
            weights[i, j, images] = 1.0 / len(images)
    return weights


@pytest.mark.parametrize("rep", [(1, 1, 1), (2, 2, 2), (0, 1, 2)])
@pytest.mark.parametrize("origin", [True, False])
def test_shift_order(rep: tuple[int, int, int], origin: bool) -> None:
    shifts = lattice_shifts_mctclib(rep, origin=origin)
    assert shifts.dtype == torch.long
    assert shifts.tolist() == _fortran_order(rep, origin)

    # every shift of the box, once
    box = math.prod(2 * r + 1 for r in rep) - (0 if origin else 1)
    assert len({tuple(s) for s in shifts.tolist()}) == box == len(shifts)


@pytest.mark.parametrize("cell", CELLS, ids="-".join)
def test_matches_fortran(cell: tuple[str, str]) -> None:
    structure = get_structure(*cell, **dd)
    assert structure.lattice is not None
    wsc = wignerseitz_cell(structure.positions, structure.lattice, PERIODIC)

    assert wsc.shifts.tolist() == _fortran_order((1, 1, 1), True)
    ref = _fortran_weights(structure.positions, structure.lattice)
    assert torch.equal(wsc.weights, ref)
    assert torch.allclose(wsc.weights.sum(-1), torch.ones_like(ref[..., 0]))


def test_cubic() -> None:
    """Two atoms half a cell apart see each other twice; every atom sees
    six images of itself, none at the zero shift."""
    lattice = 4.0 * torch.eye(3, **dd)
    positions = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], **dd)
    wsc = wignerseitz_cell(positions, lattice, PERIODIC)

    pair = wsc.shifts[wsc.weights[0, 1] > 0].tolist()
    assert sorted(pair) == [[-1, 0, 0], [0, 0, 0]]
    assert torch.allclose(wsc.weights[0, 1].sum(), torch.tensor(1.0, **dd))

    for i in range(2):
        own = wsc.shifts[wsc.weights[i, i] > 0]
        assert len(own) == 6
        assert (own.abs().sum(-1) == 1).all()
        assert torch.allclose(
            wsc.weights[i, i][wsc.weights[i, i] > 0],
            torch.full((6,), 1 / 6, **dd),
        )


def test_open_axis() -> None:
    """A shift along an axis that is not periodic is never an image, even
    with a short placeholder lattice vector."""
    lattice = torch.diag(torch.tensor([4.0, 4.0, 1.0], **dd))
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.5]], **dd)
    periodic = torch.tensor([True, True, False], device=DEVICE)
    wsc = wignerseitz_cell(positions, lattice, periodic)

    used = wsc.weights > 0
    assert not used[..., wsc.shifts[:, 2] != 0].any()
    assert wsc.shifts[wsc.weights[0, 1] > 0].tolist() == [[0, 0, 0]]
    assert len(wsc.shifts[wsc.weights[0, 0] > 0]) == 4


def test_molecule() -> None:
    """Without periodic axes, an atom has no image of itself: zero
    weights, not `nan`."""
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], **dd)
    periodic = torch.tensor([False, False, False], device=DEVICE)
    wsc = wignerseitz_cell(positions, torch.eye(3, **dd), periodic)
    assert (wsc.weights[0, 0] == 0).all()
    assert wsc.weights[0, 1].sum() == 1.0


def test_wrap_invariant() -> None:
    """Moving an atom by a lattice vector changes its shifts, but not the
    image vectors they select."""
    structure = get_structure("x23", "CO2", **dd)
    lattice = structure.lattice
    assert lattice is not None
    moved = structure.positions.clone()
    moved[0] += lattice[1]
    wrapped, _ = wrap_to_central_cell(moved, lattice, PERIODIC)

    def image_vectors(positions: Tensor) -> Tensor:
        wsc = wignerseitz_cell(positions, lattice, PERIODIC)
        trans = wsc.shifts.to(positions.dtype) @ lattice
        vectors = positions[None, :, :] - positions[:, None, :]
        images = vectors[..., None, :] + trans
        return (wsc.weights[..., None] * images).sum(-2)

    assert torch.allclose(
        image_vectors(wrapped), image_vectors(structure.positions), atol=1e-12
    )


def test_batch() -> None:
    structures = [get_structure(*cell, **dd) for cell in CELLS[:2]]
    nat = max(s.positions.shape[0] for s in structures)
    positions = torch.zeros(len(structures), nat, 3, **dd)
    for b, s in enumerate(structures):
        positions[b, : s.positions.shape[0]] = s.positions
    lattices = torch.stack([s.lattice for s in structures])  # type: ignore[misc]

    wsc = wignerseitz_cell(positions, lattices, PERIODIC.expand(2, 3))
    for b, s in enumerate(structures):
        n = s.positions.shape[0]
        assert s.lattice is not None
        single = wignerseitz_cell(s.positions, s.lattice, PERIODIC)
        assert torch.equal(wsc.weights[b, :n, :n], single.weights)


def test_vmap() -> None:
    structure = get_structure("x23", "CO2", **dd)
    lattice = structure.lattice
    assert lattice is not None
    batch = torch.stack(
        [structure.positions, structure.positions + 0.3, structure.positions]
    )

    def weights(positions: Tensor) -> Tensor:
        return wignerseitz_cell(positions, lattice, PERIODIC).weights

    with no_vmap_fallback():
        batched = vmap(weights)(batch)
    assert torch.equal(batched, torch.stack([weights(p) for p in batch]))


def test_no_gradient() -> None:
    structure = get_structure("x23", "CO2", **dd)
    assert structure.lattice is not None
    positions = structure.positions.clone().requires_grad_(True)
    wsc = wignerseitz_cell(positions, structure.lattice, PERIODIC)
    assert not wsc.weights.requires_grad
