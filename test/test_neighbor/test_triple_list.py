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
Test :class:`tad_mctc.neighbor.triples.TripleList`: the triangles of a
list's pairs, with their sides as slots of the list, against
:func:`~tad_mctc.neighbor.triples.triples_from_neighborlist`, whose sides
are looked up here independently, in a dictionary.
"""

from __future__ import annotations

from collections import Counter

import pytest
import torch

from tad_mctc.io.structure import Structure, pack_structures
from tad_mctc.neighbor import pair_distance_squared, split_lattice
from tad_mctc.neighbor.list import NeighborList, build_neighborlist
from tad_mctc.neighbor.triples import (
    TripleList,
    _PairLookup,
    triples_from_neighborlist,
)
from tad_mctc.typing import DD

from ..conftest import DEVICE
from ..utils import hydrogens
from .test_triples import _fcc_cell, _random_positions, _triclinic_cell

DD64: DD = {"device": DEVICE, "dtype": torch.double}

Shift = tuple[int, int, int]
Side = tuple[int, int, Shift]
Triple = tuple[int, int, int, int, int, int]
"""``(i, j, k, side_ij, side_jk, side_ik)``, `i` and `k` in either order."""


def _forward(a: int, b: int, shift: Shift) -> Side:
    """The pair `(a, b, shift)`, turned to point forward."""
    back = (-shift[0], -shift[1], -shift[2])
    if a > b or (a == b and shift < (0, 0, 0)):
        return b, a, back
    return a, b, shift


def _sides(nbl: NeighborList) -> dict[Side, int]:
    """The slot of every pair of `nbl`, by its forward form."""
    out: dict[Side, int] = {}
    for slot in nbl.mask.nonzero().squeeze(-1).tolist():
        shift = tuple(int(x) for x in nbl.shift[slot].tolist())
        side = _forward(int(nbl.idx_i[slot]), int(nbl.idx_j[slot]), shift)
        assert side not in out, "a pair is stored twice"
        out[side] = slot
    return out


def _unordered(i: int, j: int, k: int, ij: int, jk: int, ik: int) -> Triple:
    (i, ij), (k, jk) = sorted([(i, ij), (k, jk)])
    return i, j, k, ij, jk, ik


def _expected(nbl: NeighborList, structure: Structure, cutoff: float) -> list:
    """The triples of `triples_from_neighborlist`, with their sides."""
    sides = _sides(nbl)
    out = []
    for chunk in triples_from_neighborlist(nbl, structure, cutoff):
        for i, j, k, si, sk in zip(
            chunk.idx_i.tolist(),
            chunk.idx_j.tolist(),
            chunk.idx_k.tolist(),
            chunk.shift_i.tolist(),
            chunk.shift_k.tolist(),
        ):
            ik = (sk[0] - si[0], sk[1] - si[1], sk[2] - si[2])
            out.append(
                _unordered(
                    i,
                    j,
                    k,
                    sides[_forward(j, i, tuple(si))],
                    sides[_forward(j, k, tuple(sk))],
                    sides[_forward(i, k, ik)],
                )
            )
    return out


def _within(
    triples: TripleList, structure: Structure, cutoff: float
) -> list[Triple]:
    """The triples with all three sides within `cutoff` at the positions
    of `structure`, from the distances of the slots, as a consumer does."""
    nbl = triples.pairs
    nat = structure.positions.shape[-2]
    positions = torch.cat(
        [
            structure.positions.reshape(-1, 3),
            structure.positions.new_zeros(1, 3),
        ]
    )
    shared, per_system = split_lattice(structure.lattice)
    if per_system is not None:
        per_system = torch.cat([per_system, per_system.new_zeros(1, 3, 3)])
    r2 = pair_distance_squared(
        nbl.idx_i,
        nbl.idx_j,
        nbl.shift,
        positions,
        shared_lattice=shared,
        system_lattices=per_system,
        atoms_per_system=nat,
    )
    inside = r2 <= cutoff * cutoff
    keep = (
        inside[triples.side_ij]
        & inside[triples.side_jk]
        & inside[triples.side_ik]
    )
    return [
        _unordered(*row)
        for row in zip(
            triples.idx_i[keep].tolist(),
            triples.idx_j[keep].tolist(),
            triples.idx_k[keep].tolist(),
            triples.side_ij[keep].tolist(),
            triples.side_jk[keep].tolist(),
            triples.side_ik[keep].tolist(),
        )
    ]


def _check(structure: Structure, cutoff: float, **build: float) -> TripleList:
    """Build the list and its triples, and check them against
    `triples_from_neighborlist` at `cutoff`, for the structure the list was
    built from."""
    nbl = build_neighborlist(structure, cutoff=cutoff, tile=16, **build)
    triples = TripleList.from_neighborlist(nbl)
    _check_sides(triples)

    expected = _expected(nbl, structure, cutoff)
    assert len(expected) > 0, "nothing to compare"
    got = _within(triples, structure, cutoff)
    assert len(got) == len(set(got)), "a triple is listed twice"
    assert len(got) == len(expected)
    assert set(got) == set(expected)
    return triples


def _check_sides(triples: TripleList) -> None:
    """Each side is a real slot of the list between two atoms of its
    triple, the ones it is named after."""
    nbl = triples.pairs
    for side, (a, b) in (
        (triples.side_ij, (triples.idx_i, triples.idx_j)),
        (triples.side_jk, (triples.idx_j, triples.idx_k)),
        (triples.side_ik, (triples.idx_i, triples.idx_k)),
    ):
        assert bool(nbl.mask[side].all()), "a side is a padding slot"
        first, second = nbl.idx_i[side], nbl.idx_j[side]
        same = (first == a) & (second == b)
        swapped = (first == b) & (second == a)
        assert bool((same | swapped).all()), "a side joins other atoms"


def _molecule(nat: int, seed: int) -> Structure:
    return hydrogens(_random_positions(nat, 0.03, seed, DD64))


###############################################################################


def test_molecule() -> None:
    triples = _check(_molecule(40, 1), 6.0)
    assert triples.idx_i.shape[0] > 0


def test_molecule_every_triangle_is_within_the_cutoff() -> None:
    """Without a skin, the triangles of the list are exactly the triples
    within its cutoff."""
    structure = _molecule(40, 2)
    nbl = build_neighborlist(structure, cutoff=6.0, tile=16)
    triples = TripleList.from_neighborlist(nbl)
    assert len(_within(triples, structure, 6.0)) == triples.idx_i.shape[0]


@pytest.mark.parametrize("chunk_size", [1, 7, 2_000_000])
def test_chunks_give_the_same_triples(chunk_size: int) -> None:
    structure = _molecule(30, 3)
    nbl = build_neighborlist(structure, cutoff=6.0, tile=16)
    whole = TripleList.from_neighborlist(nbl)
    chunked = TripleList.from_neighborlist(nbl, chunk_size=chunk_size)
    for name in ("idx_i", "idx_j", "idx_k", "side_ij", "side_jk", "side_ik"):
        assert torch.equal(getattr(whole, name), getattr(chunked, name))


def test_skin_reaches_cutoff_plus_skin() -> None:
    """With a skin, the triangles are the triples within `cutoff + skin`
    (of a list built there without one), and masked at `cutoff`, those of
    the list at its cutoff."""
    structure = _molecule(40, 4)
    cutoff, skin = 5.0, 1.5
    triples = _check(structure, cutoff, skin=skin)
    assert triples.reach == cutoff + skin

    wide = build_neighborlist(structure, cutoff=cutoff + skin, tile=16)
    assert triples.idx_i.shape[0] == len(
        _expected(wide, structure, cutoff + skin)
    )


def test_skin_still_holds_every_triple_after_a_move() -> None:
    """While no atom has moved by more than half the skin, the triples
    within the cutoff at the new positions are all there."""
    structure = _molecule(40, 5)
    cutoff, skin = 5.0, 1.0
    nbl = build_neighborlist(structure, cutoff=cutoff, tile=16, skin=skin)
    triples = TripleList.from_neighborlist(nbl)

    gen = torch.Generator().manual_seed(6)
    step = torch.rand(structure.positions.shape, generator=gen, device="cpu")
    step = step - 0.5
    step = 0.45 * skin * step / step.norm(dim=-1, keepdim=True)
    moved = structure.replace(positions=structure.positions + step.to(**DD64))
    assert not bool(triples.stale(moved))

    def atoms(rows: list[Triple]) -> Counter[tuple[int, int, int]]:
        return Counter(tuple(sorted(row[:3])) for row in rows)

    fresh = build_neighborlist(moved, cutoff=cutoff, tile=16)
    expected = atoms(_expected(fresh, moved, cutoff))
    assert sum(expected.values()) > 0
    assert atoms(_within(triples, moved, cutoff)) == expected


def test_batch_of_molecules() -> None:
    batch = pack_structures([_molecule(30, 7), _molecule(20, 8)])
    triples = _check(batch, 6.0)
    assert bool((triples.idx_j >= 30).any()), "no triple of the second"


def test_cell_with_self_images() -> None:
    positions, lattice = _fcc_cell(DD64)
    triples = _check(hydrogens(positions, lattice=lattice), 8.3)
    nbl = triples.pairs
    assert bool(
        (nbl.idx_i[triples.side_ij] == nbl.idx_j[triples.side_ij]).any()
    )


def test_triclinic_cell_with_an_atom_outside() -> None:
    positions, lattice = _triclinic_cell(DD64)
    _check(hydrogens(positions, lattice=lattice), 8.1, skin=1.0)


def test_slab() -> None:
    positions = torch.tensor(
        [[0.1, 0.2, 0.0], [2.0, 1.5, 1.2], [3.5, 3.0, -1.4]], **DD64
    )
    lattice = torch.tensor(
        [[5.0, 0.0, 0.0], [1.0, 5.5, 0.0], [0.0, 0.0, 20.0]], **DD64
    )
    periodic = torch.tensor([True, True, False], device=DEVICE)
    _check(hydrogens(positions, lattice=lattice, periodic=periodic), 7.1)


def test_batch_of_two_cells() -> None:
    first = hydrogens(
        torch.tensor([[0.3, 0.2, 0.1], [2.1, 1.7, 2.9]], **DD64),
        lattice=torch.tensor(
            [[5.0, 0.0, 0.0], [0.5, 5.5, 0.0], [0.3, 0.4, 6.0]], **DD64
        ),
    )
    second = hydrogens(
        torch.tensor(
            [[0.0, 0.1, 0.2], [2.5, 2.4, 2.6], [4.0, 1.0, 1.0]], **DD64
        ),
        lattice=torch.tensor(
            [[6.5, 0.0, 0.0], [0.0, 6.5, 0.0], [1.0, 1.0, 5.0]], **DD64
        ),
    )
    _check(pack_structures([first, second]), 7.1)


def test_molecule_in_a_batch_of_cells() -> None:
    molecule = hydrogens(
        torch.tensor(
            [[0.0, 0.0, 0.0], [1.5, 0.2, 0.1], [0.3, 2.0, 1.1]], **DD64
        ),
        lattice=torch.eye(3, **DD64),
        periodic=torch.tensor([False, False, False], device=DEVICE),
    )
    positions, lattice = _triclinic_cell(DD64)
    _check(
        pack_structures([molecule, hydrogens(positions, lattice=lattice)]), 7.1
    )


def test_unwrapped_positions_far_from_the_cell() -> None:
    """Atoms many cells away from the central one: large shifts."""
    positions, lattice = _triclinic_cell(DD64)
    far = torch.tensor([[300, 0, 0], [0, -250, 0], [0, 0, 0], [7, 7, -400]])
    unwrapped = positions + far.to(**DD64) @ lattice
    triples = _check(hydrogens(unwrapped, lattice=lattice), 8.1)
    assert int(triples.pairs.shift.abs().max()) >= 250


def test_no_triples() -> None:
    """Two atoms are one pair and no triple; atoms out of reach, none."""
    for positions in (
        torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]], **DD64),
        torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 20.0], [20.0, 0, 0]], **DD64),
    ):
        nbl = build_neighborlist(hydrogens(positions), cutoff=6.0)
        triples = TripleList.from_neighborlist(nbl)
        assert triples.idx_i.shape == (0,) and triples.side_ik.shape == (0,)


def test_check_compatible_and_stale_are_those_of_the_list() -> None:
    structure = _molecule(20, 9)
    nbl = build_neighborlist(structure, cutoff=5.0, skin=1.0)
    triples = TripleList.from_neighborlist(nbl)

    triples.check_compatible(structure, 5.0)
    with pytest.raises(ValueError, match="cutoff"):
        triples.check_compatible(structure, 5.5)
    assert not bool(triples.stale(structure))


def test_lookup_refuses_keys_beyond_64_bits() -> None:
    none = torch.zeros(0, dtype=torch.long)
    with pytest.raises(ValueError, match="64-bit"):
        _PairLookup(
            2**32, none, none, none, torch.zeros(0, 3, dtype=torch.long)
        )


def test_lookup_does_not_alias_a_shift_beyond_the_list() -> None:
    """A shift beyond those of the list is not found, even where clamping
    it would hit a pair of the list."""
    slot = torch.tensor([0, 1])
    atoms = torch.tensor([0, 0])
    shifts = torch.tensor([[1, 0, 0], [2, 0, 0]])
    lookup = _PairLookup(1, slot, atoms, atoms, shifts)

    found_slot, found = lookup.find(
        atoms, atoms, torch.tensor([[2, 0, 0], [5, 0, 0]])
    )
    assert found.tolist() == [True, False]
    assert int(found_slot[0]) == 1
    _, found = lookup.find(atoms[:1], atoms[:1], torch.tensor([[-1, 0, 0]]))
    assert found.tolist() == [True], "the reverse of a pair is the pair"
