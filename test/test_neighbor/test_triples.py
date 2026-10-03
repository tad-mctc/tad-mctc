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
Test :mod:`tad_mctc.neighbor.triples`: enumeration only, no dispersion
physics -- just positions, a built :class:`.NeighborList`, and the
resulting candidate-triple index tensors.
"""

from __future__ import annotations

from typing import Iterator

import pytest
import torch

from tad_mctc.io.structure import Structure, pack_structures
from tad_mctc.neighbor.list import build_neighborlist
from tad_mctc.neighbor.triples import (
    TripleChunk,
    TripleIndex,
    _iter_triple_chunks_loop,
    _iter_triple_chunks_vectorized,
    _oriented_csr,
    _triangle_rows,
    _triangular_indices,
    triples_from_neighborlist,
)
from tad_mctc.typing import DD, Tensor

from ..conftest import DEVICE
from ..utils import hydrogens, load_structure


def _as_triple_set(
    trip_i: Tensor, trip_j: Tensor, trip_k: Tensor
) -> set[tuple[int, int, int]]:
    return set(zip(trip_i.tolist(), trip_j.tolist(), trip_k.tolist()))


def _as_unordered_triple_set(
    trip_i: Tensor, trip_j: Tensor, trip_k: Tensor
) -> set[tuple[int, int, int]]:
    """Triples with the two outer atoms sorted, for comparing triples of
    two different lists: which neighbour comes out as ``i`` and which as
    ``k`` depends on how the list stores each pair."""
    return {
        (min(i, k), j, max(i, k))
        for i, j, k in zip(trip_i.tolist(), trip_j.tolist(), trip_k.tolist())
    }


def _collect(
    chunks: Iterator[TripleChunk],
) -> tuple[Tensor, Tensor, Tensor]:
    """Concatenate the ``idx_i``, ``idx_j`` and ``idx_k`` of every yielded
    :class:`TripleChunk` into the full, unchunked triple list."""
    left_parts: list[Tensor] = []
    centre_parts: list[Tensor] = []
    right_parts: list[Tensor] = []
    for chunk in chunks:
        left_parts.append(chunk.idx_i)
        centre_parts.append(chunk.idx_j)
        right_parts.append(chunk.idx_k)

    if not left_parts:
        empty = torch.zeros(0, dtype=torch.long, device=DEVICE)
        return empty, empty, empty

    return (
        torch.cat(left_parts),
        torch.cat(centre_parts),
        torch.cat(right_parts),
    )


def _random_positions(nat: int, density: float, seed: int, dd: DD) -> Tensor:
    box_edge = (nat / density) ** (1.0 / 3.0)
    gnrtr = torch.Generator().manual_seed(seed)
    return torch.rand(nat, 3, generator=gnrtr, device="cpu").to(**dd) * box_edge


def test_molecular_triples_are_listed_once_with_the_smallest_atom_as_centre() -> (
    None
):
    """A molecular triple of atoms `a < b < c` is yielded exactly once, as
    `TripleChunk` with `idx_j=a`, the smallest atom, as the centre and `b`,
    `c` as its two neighbours in either order. The reference is a
    brute-force loop over all atom triples, independent of any neighbour
    list."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    nat = 30
    positions = _random_positions(nat, density=0.05, seed=1, dd=dd)
    cutoff = 8.0

    expected: set[tuple[int, int, int]] = set()
    for a in range(nat):
        for b in range(a + 1, nat):
            for c in range(b + 1, nat):
                sides = [
                    (positions[a] - positions[b]).norm(),
                    (positions[a] - positions[c]).norm(),
                    (positions[b] - positions[c]).norm(),
                ]
                if all(float(side) <= cutoff for side in sides):
                    expected.add((b, a, c))
    assert len(expected) > 0, "box too sparse to exercise any triples"

    structure = hydrogens(positions)
    nbl = build_neighborlist(structure, cutoff=cutoff, tile=16)

    chunks = list(triples_from_neighborlist(nbl, structure, cutoff))
    assert all(isinstance(chunk, TripleChunk) for chunk in chunks)

    got = [
        (min(i, k), j, max(i, k))
        for chunk in chunks
        for i, j, k in zip(
            chunk.idx_i.tolist(), chunk.idx_j.tolist(), chunk.idx_k.tolist()
        )
    ]
    assert len(got) == len(set(got)), "a triple was listed twice"
    assert set(got) == expected

    for chunk in chunks:
        assert chunk.shift_i.shape == (chunk.idx_i.shape[0], 3)
        assert chunk.shift_k.shape == (chunk.idx_k.shape[0], 3)
        assert not bool(chunk.shift_i.any())
        assert not bool(chunk.shift_k.any())


Point = tuple[int, tuple[int, int, int]]
"""An atom and an image shift, compared lexicographically."""

PeriodicTriple = tuple[
    int, tuple[int, int, int], int, int, tuple[int, int, int]
]
"""``(i, shift_i, j, k, shift_k)``, the pair ``(i, k)`` sorted by point."""


def _periodic_triple(
    centre: int,
    first: Point,
    second: Point,
) -> PeriodicTriple:
    (i, shift_i), (k, shift_k) = sorted([first, second])
    return (i, shift_i, centre, k, shift_k)


def _as_periodic_triple_list(
    chunks: Iterator[TripleChunk], offset: int = 0
) -> list[PeriodicTriple]:
    """Every row of every chunk as a hashable `PeriodicTriple`, atoms
    shifted by ``offset`` (the first atom of a system in a batch)."""
    out: list[PeriodicTriple] = []
    for chunk in chunks:
        for i, j, k, shift_i, shift_k in zip(
            chunk.idx_i.tolist(),
            chunk.idx_j.tolist(),
            chunk.idx_k.tolist(),
            chunk.shift_i.tolist(),
            chunk.shift_k.tolist(),
        ):
            out.append(
                _periodic_triple(
                    j - offset,
                    (i - offset, tuple(shift_i)),
                    (k - offset, tuple(shift_k)),
                )
            )
    return out


def _brute_force_periodic_triples(
    positions: Tensor,
    lattice: Tensor,
    periodic: list[bool],
    cutoff: float,
) -> set[PeriodicTriple]:
    """
    Every triple of a periodic system, from explicit lattice translations
    alone -- no neighbour list.

    For each atom `c` in the central cell, the neighbours are all points
    `(atom, shift)` with `x_atom + shift @ lattice` inside `cutoff` of
    `x_c`. A triple is kept when both neighbours are *upward* -- a larger
    `(atom, shift)` than `(c, 0)` -- and their mutual distance is inside
    `cutoff` too. Asserts that no side lies within 1e-6 of the cutoff, so
    that the comparison cannot hinge on rounding.
    """
    nat = positions.shape[0]
    spread = float((positions[:, None] - positions[None]).norm(dim=-1).max())
    inverse = torch.linalg.inv(lattice)
    reach = [
        (
            int(torch.ceil((cutoff + spread) * inverse[:, axis].norm())) + 1
            if periodic[axis]
            else 0
        )
        for axis in range(3)
    ]

    shifts = torch.tensor(
        [
            (a, b, c)
            for a in range(-reach[0], reach[0] + 1)
            for b in range(-reach[1], reach[1] + 1)
            for c in range(-reach[2], reach[2] + 1)
        ],
        dtype=positions.dtype,
        device=positions.device,
    )
    points = positions[:, None, :] + (shifts @ lattice)[None]  # (nat, n, 3)

    result: set[PeriodicTriple] = set()
    for c in range(nat):
        distance = (points - positions[c]).norm(dim=-1)  # (nat, n)
        assert not bool(((distance - cutoff).abs() < 1e-6).any())

        upward: list[Point] = []
        for atom in range(nat):
            for index in range(shifts.shape[0]):
                shift = tuple(int(x) for x in shifts[index].tolist())
                if float(distance[atom, index]) > cutoff:
                    continue
                if (atom, shift) > (c, (0, 0, 0)):
                    upward.append((atom, shift))

        for p, (a, shift_a) in enumerate(upward):
            x_a = points[a, _shift_index(shifts, shift_a)]
            for b, shift_b in upward[p + 1 :]:
                x_b = points[b, _shift_index(shifts, shift_b)]
                third = float((x_a - x_b).norm())
                assert abs(third - cutoff) >= 1e-6
                if third <= cutoff:
                    result.add(_periodic_triple(c, (a, shift_a), (b, shift_b)))
    return result


def _shift_index(shifts: Tensor, shift: tuple[int, int, int]) -> int:
    target = torch.tensor(shift, dtype=shifts.dtype, device=shifts.device)
    return int((shifts == target).all(dim=-1).nonzero()[0, 0])


def test_periodic_triples_one_atom_cubic_cell_match_brute_force() -> None:
    """All triples of a single atom in a small cubic cell are images of that
    one atom: every neighbour is a self-image, and every triple has to be
    listed exactly once, not once per choice of centre."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    positions = torch.tensor([[0.3, 0.2, 0.1]], **dd)
    lattice = torch.eye(3, **dd) * 5.0
    cutoff = 7.3

    expected = _brute_force_periodic_triples(
        positions, lattice, [True, True, True], cutoff
    )
    assert len(expected) > 0

    structure = hydrogens(positions, lattice=lattice)
    nbl = build_neighborlist(structure, cutoff=cutoff, tile=16)
    got = _as_periodic_triple_list(
        triples_from_neighborlist(nbl, structure, cutoff)
    )

    assert len(got) == len(set(got)), "a triple was listed twice"
    assert set(got) == expected


def _fcc_cell(dd: DD) -> tuple[Tensor, Tensor]:
    """Two atoms in an fcc primitive cell: the pairs between the atoms
    and between images of each atom."""
    a = 2.6
    positions = torch.tensor([[0.0, 0.0, 0.0], [a, a, a]], **dd)
    lattice = torch.tensor([[0.0, a, a], [a, 0.0, a], [a, a, 0.0]], **dd) * 2.0
    return positions, lattice


def _triclinic_cell(dd: DD) -> tuple[Tensor, Tensor]:
    """Four atoms in a triclinic cell, the last one outside of it."""
    positions = torch.tensor(
        [
            [0.1, 0.2, 0.3],
            [2.0, 0.5, 1.0],
            [-3.0, 4.0, 1.5],
            [9.5, 2.5, 7.0],
        ],
        **dd,
    )
    lattice = torch.tensor(
        [[7.0, 0.0, 0.0], [1.5, 6.5, 0.0], [0.8, 1.2, 7.5]], **dd
    )
    return positions, lattice


def test_periodic_triples_fcc_cell_with_self_images_match_brute_force() -> None:
    """Two atoms in an fcc cell: the list holds atom-atom pairs and
    self-image pairs (`i == j`, shift != 0), which stand for both images
    `+S` and `-S`."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    positions, lattice = _fcc_cell(dd)
    cutoff = 8.3

    expected = _brute_force_periodic_triples(
        positions, lattice, [True, True, True], cutoff
    )
    assert len(expected) > 0

    structure = hydrogens(positions, lattice=lattice)
    nbl = build_neighborlist(structure, cutoff=cutoff, tile=16)
    real = nbl.mask
    assert bool((nbl.idx_i[real] == nbl.idx_j[real]).any()), "no self-image"

    got = _as_periodic_triple_list(
        triples_from_neighborlist(nbl, structure, cutoff)
    )

    assert len(got) == len(set(got)), "a triple was listed twice"
    assert set(got) == expected


def test_periodic_triples_triclinic_cell_match_brute_force() -> None:
    """A triclinic cell with an atom outside the central cell: shifts are
    relative to the caller's, unwrapped positions."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    positions, lattice = _triclinic_cell(dd)
    cutoff = 8.1

    expected = _brute_force_periodic_triples(
        positions, lattice, [True, True, True], cutoff
    )
    assert len(expected) > 0

    structure = hydrogens(positions, lattice=lattice)
    nbl = build_neighborlist(structure, cutoff=cutoff, tile=16)
    got = _as_periodic_triple_list(
        triples_from_neighborlist(nbl, structure, cutoff)
    )

    assert len(got) == len(set(got)), "a triple was listed twice"
    assert set(got) == expected


def test_periodic_triples_slab_match_brute_force() -> None:
    """A slab, periodic along x and y only: no triple uses a z image."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    positions = torch.tensor(
        [[0.1, 0.2, 0.0], [2.0, 1.5, 1.2], [3.5, 3.0, -1.4]], **dd
    )
    lattice = torch.tensor(
        [[5.0, 0.0, 0.0], [1.0, 5.5, 0.0], [0.0, 0.0, 20.0]], **dd
    )
    periodic = torch.tensor([True, True, False], device=DEVICE)
    cutoff = 7.1

    expected = _brute_force_periodic_triples(
        positions, lattice, [True, True, False], cutoff
    )
    assert len(expected) > 0

    structure = hydrogens(positions, lattice=lattice, periodic=periodic)
    nbl = build_neighborlist(structure, cutoff=cutoff, tile=16)
    got = _as_periodic_triple_list(
        triples_from_neighborlist(nbl, structure, cutoff)
    )

    assert len(got) == len(set(got)), "a triple was listed twice"
    assert set(got) == expected
    assert all(triple[1][2] == 0 and triple[4][2] == 0 for triple in got)


def test_periodic_triples_are_one_per_translation_class() -> None:
    """A triple is an unordered set of three points up to a common lattice
    translation. Translating each yielded triple so that its smallest
    point sits at the origin must give a different result for every
    triple: no class appears twice, whichever atom is the centre."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    cutoff = 8.1

    for positions, lattice in (_fcc_cell(dd), _triclinic_cell(dd)):
        structure = hydrogens(positions, lattice=lattice)
        nbl = build_neighborlist(structure, cutoff=cutoff, tile=16)

        classes: list[tuple[Point, Point, Point]] = []
        for i, shift_i, j, k, shift_k in _as_periodic_triple_list(
            triples_from_neighborlist(nbl, structure, cutoff)
        ):
            points = sorted([(i, shift_i), (j, (0, 0, 0)), (k, shift_k)])
            origin = points[0][1]
            classes.append(
                tuple(  # type: ignore[arg-type]
                    (atom, tuple(s - o for s, o in zip(shift, origin)))
                    for atom, shift in points
                )
            )

        assert len(classes) > 0
        assert len(classes) == len(set(classes))


def test_oriented_csr_lists_each_neighbour_image_once_per_centre() -> None:
    """Within one centre's block, `(atom, shift)` must be unique: the
    one-per-class count rests on it, and a duplicate would double count
    every triple that uses it."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    cutoff = 8.1

    for positions, lattice in (_fcc_cell(dd), _triclinic_cell(dd)):
        structure = hydrogens(positions, lattice=lattice)
        nbl = build_neighborlist(structure, cutoff=cutoff, tile=16)
        nat = positions.shape[0]

        offset, neighbours, neighbour_shift, degree = _oriented_csr(
            nat, nbl.idx_i[nbl.mask], nbl.idx_j[nbl.mask], nbl.shift[nbl.mask]
        )
        # Every stored pair is upward from exactly one of its two ends.
        assert int(degree.sum()) == int(nbl.mask.sum())

        for centre in range(nat):
            start, stop = int(offset[centre]), int(
                offset[centre] + degree[centre]
            )
            block = [
                (atom, tuple(shift))
                for atom, shift in zip(
                    neighbours[start:stop].tolist(),
                    neighbour_shift[start:stop].tolist(),
                )
            ]
            assert len(block) == len(set(block))
            assert all(point > (centre, (0, 0, 0)) for point in block)


def test_periodic_triples_loop_matches_vectorized() -> None:
    """The CPU loop and the vectorized enumeration must list the same
    periodic triples, shifts included, for a chunked and an unchunked
    run. Both generators are called directly, bypassing the device
    dispatch."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    positions, lattice = _triclinic_cell(dd)
    nat = positions.shape[0]
    cutoff = 8.1

    structure = hydrogens(positions, lattice=lattice)
    nbl = build_neighborlist(structure, cutoff=cutoff, tile=16)
    real_i, real_j = nbl.idx_i[nbl.mask], nbl.idx_j[nbl.mask]
    real_shift = nbl.shift[nbl.mask]

    reference = _as_periodic_triple_list(
        _iter_triple_chunks_loop(
            nat,
            real_i,
            real_j,
            real_shift,
            positions,
            lattice,
            nat,
            cutoff,
            chunk_size=None,
        )
    )
    assert len(reference) > 0

    for chunk_size in (None, 1, 7):
        loop = _as_periodic_triple_list(
            _iter_triple_chunks_loop(
                nat,
                real_i,
                real_j,
                real_shift,
                positions,
                lattice,
                nat,
                cutoff,
                chunk_size=chunk_size,
            )
        )
        vectorized = _as_periodic_triple_list(
            _iter_triple_chunks_vectorized(
                nat,
                real_i,
                real_j,
                real_shift,
                positions,
                lattice,
                nat,
                cutoff,
                chunk_size=chunk_size,
            )
        )
        assert len(loop) == len(set(loop))
        assert len(vectorized) == len(set(vectorized))
        assert set(loop) == set(reference), f"loop, chunk_size={chunk_size}"
        assert set(vectorized) == set(
            reference
        ), f"vec, chunk_size={chunk_size}"


def test_periodic_triples_chunks_respect_chunk_size() -> None:
    """Through the public API, a periodic run is cut into chunks of at most
    `chunk_size` rows and yields the same triples as an unchunked run."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    positions, lattice = _triclinic_cell(dd)
    cutoff = 8.1

    structure = hydrogens(positions, lattice=lattice)
    nbl = build_neighborlist(structure, cutoff=cutoff, tile=16)

    unchunked = _as_periodic_triple_list(
        triples_from_neighborlist(nbl, structure, cutoff, chunk_size=None)
    )
    for chunk_size in (1, 7):
        chunks = list(
            triples_from_neighborlist(
                nbl, structure, cutoff, chunk_size=chunk_size
            )
        )
        assert len(chunks) > 1
        assert all(c.idx_i.shape[0] <= chunk_size for c in chunks)
        assert sorted(_as_periodic_triple_list(iter(chunks))) == sorted(
            unchunked
        )


def test_periodic_triples_from_a_skinned_list_match_the_exact_list() -> None:
    """A list built with a skin holds pairs beyond the cutoff; the triples
    must be the same as those of a list built at the exact cutoff."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    positions, lattice = _triclinic_cell(dd)
    cutoff = 8.1

    structure = hydrogens(positions, lattice=lattice)
    exact = build_neighborlist(structure, cutoff=cutoff, tile=16)
    skinned = build_neighborlist(structure, cutoff=cutoff, tile=16, skin=3.0)
    assert int(skinned.mask.sum()) > int(exact.mask.sum())

    expected = _as_periodic_triple_list(
        triples_from_neighborlist(exact, structure, cutoff)
    )
    got = _as_periodic_triple_list(
        triples_from_neighborlist(skinned, structure, cutoff)
    )
    assert len(got) == len(set(got))
    assert set(got) == set(expected)


def test_periodic_triples_batch_of_two_cells_match_each_system() -> None:
    """A batch with a different lattice per system: each triple takes the
    cell of its own system, and the atoms are numbered `b * nat + i`."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    cutoff = 7.1

    positions_a = torch.tensor([[0.3, 0.2, 0.1], [2.1, 1.7, 2.9]], **dd)
    lattice_a = torch.tensor(
        [[5.0, 0.0, 0.0], [0.5, 5.5, 0.0], [0.3, 0.4, 6.0]], **dd
    )
    positions_b = torch.tensor([[0.0, 0.1, 0.2], [2.5, 2.4, 2.6]], **dd)
    lattice_b = torch.tensor(
        [[6.5, 0.0, 0.0], [0.0, 6.5, 0.0], [1.0, 1.0, 5.0]], **dd
    )
    assert not torch.equal(lattice_a, lattice_b)

    system_a = hydrogens(positions_a, lattice=lattice_a)
    system_b = hydrogens(positions_b, lattice=lattice_b)
    nat = 2

    expected: list[PeriodicTriple] = []
    for system, structure in enumerate([system_a, system_b]):
        nbl = build_neighborlist(structure, cutoff=cutoff, tile=16)
        for i, shift_i, j, k, shift_k in _as_periodic_triple_list(
            triples_from_neighborlist(nbl, structure, cutoff)
        ):
            expected.append(
                _periodic_triple(
                    j + system * nat,
                    (i + system * nat, shift_i),
                    (k + system * nat, shift_k),
                )
            )
    assert len(expected) > 0

    batch = pack_structures([system_a, system_b])
    batch_nbl = build_neighborlist(batch, cutoff=cutoff, tile=16)
    got = _as_periodic_triple_list(
        triples_from_neighborlist(batch_nbl, batch, cutoff)
    )

    assert len(got) == len(set(got))
    assert set(got) == set(expected)


def _two_cells(dd: DD) -> tuple[Tensor, Tensor]:
    """Two systems of two atoms, each with its own lattice."""
    positions = torch.tensor(
        [
            [[0.3, 0.2, 0.1], [2.1, 1.7, 2.9]],
            [[0.0, 0.1, 0.2], [2.5, 2.4, 2.6]],
        ],
        **dd,
    )
    lattice = torch.tensor(
        [
            [[5.0, 0.0, 0.0], [0.5, 5.5, 0.0], [0.3, 0.4, 6.0]],
            [[6.5, 0.0, 0.0], [0.0, 6.5, 0.0], [1.0, 1.0, 5.0]],
        ],
        **dd,
    )
    return positions, lattice


def test_periodic_triples_batch_vectorized_matches_loop() -> None:
    """The vectorized path gathers one cell per triple from its centre's
    system; the loop path picks one cell per centre. Called directly on the
    same batch of two different cells, they must list the same triples,
    which the public dispatch on CPU never checks for the vectorized
    path."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    positions, lattice = _two_cells(dd)
    cutoff = 7.1

    batch = pack_structures(
        [
            hydrogens(positions[0], lattice=lattice[0]),
            hydrogens(positions[1], lattice=lattice[1]),
        ]
    )
    nbl = build_neighborlist(batch, cutoff=cutoff, tile=16)
    flat = batch.positions.reshape(-1, 3)
    args = (
        flat.shape[0],
        nbl.idx_i[nbl.mask],
        nbl.idx_j[nbl.mask],
        nbl.shift[nbl.mask],
        flat,
        batch.lattice,
        2,
    )

    for chunk_size in (None, 3):
        loop = _as_periodic_triple_list(
            _iter_triple_chunks_loop(*args, cutoff, chunk_size=chunk_size)
        )
        vectorized = _as_periodic_triple_list(
            _iter_triple_chunks_vectorized(*args, cutoff, chunk_size=chunk_size)
        )
        assert len(loop) > 0
        assert len(vectorized) == len(set(vectorized))
        assert set(vectorized) == set(loop), f"chunk_size={chunk_size}"


def test_periodic_triples_batch_sharing_one_cell_match_each_system() -> None:
    """A batch whose systems share one cell, given as a single `(3, 3)`
    lattice, applies that cell to every system."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    positions, lattice = _two_cells(dd)
    cell = lattice[0]
    cutoff = 7.1

    expected: list[PeriodicTriple] = []
    for system in range(2):
        structure = hydrogens(positions[system], lattice=cell)
        nbl = build_neighborlist(structure, cutoff=cutoff, tile=16)
        for i, shift_i, j, k, shift_k in _as_periodic_triple_list(
            triples_from_neighborlist(nbl, structure, cutoff)
        ):
            expected.append(
                _periodic_triple(
                    j + 2 * system,
                    (i + 2 * system, shift_i),
                    (k + 2 * system, shift_k),
                )
            )
    assert len(expected) > 0

    batch = hydrogens(positions, lattice=cell)
    nbl = build_neighborlist(batch, cutoff=cutoff, tile=16)
    got = _as_periodic_triple_list(
        triples_from_neighborlist(nbl, batch, cutoff)
    )
    assert len(got) == len(set(got))
    assert set(got) == set(expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_periodic_triples_batch_cuda_match_cpu() -> None:
    """A batch of two different cells on CUDA (vectorized enumeration)
    gives the triples of the CPU loop."""
    dd: DD = {"device": torch.device("cpu"), "dtype": torch.double}
    positions, lattice = _two_cells(dd)
    cutoff = 7.1

    cpu = hydrogens(positions, lattice=lattice)
    cuda = hydrogens(positions.cuda(), lattice=lattice.cuda())

    on_cpu = _as_periodic_triple_list(
        triples_from_neighborlist(
            build_neighborlist(cpu, cutoff=cutoff, tile=16), cpu, cutoff
        )
    )
    on_cuda = _as_periodic_triple_list(
        triples_from_neighborlist(
            build_neighborlist(cuda, cutoff=cutoff, tile=16), cuda, cutoff
        )
    )
    assert len(on_cpu) > 0
    assert set(on_cuda) == set(on_cpu)


def _small_and_larger_cell(dd: DD) -> tuple[Structure, Structure]:
    """A 2-atom and a 3-atom cell with different lattices: packed, the
    first carries one padding atom."""
    small = hydrogens(
        torch.tensor([[0.3, 0.2, 0.1], [2.1, 1.7, 2.9]], **dd),
        lattice=torch.tensor(
            [[5.0, 0.0, 0.0], [0.5, 5.5, 0.0], [0.3, 0.4, 6.0]], **dd
        ),
    )
    larger = hydrogens(
        torch.tensor(
            [[0.0, 0.1, 0.2], [2.5, 2.4, 2.6], [-1.0, 3.5, 4.0]], **dd
        ),
        lattice=torch.tensor(
            [[6.5, 0.0, 0.0], [0.0, 6.5, 0.0], [1.0, 1.0, 5.0]], **dd
        ),
    )
    return small, larger


def _triples_of_each_system(
    systems: list[Structure], nat: int, cutoff: float
) -> list[PeriodicTriple]:
    """The triples of every system on its own, in the flat numbering
    `b * nat + i` of the batch."""
    out: list[PeriodicTriple] = []
    for b, structure in enumerate(systems):
        nbl = build_neighborlist(structure, cutoff=cutoff, tile=16)
        for i, shift_i, j, k, shift_k in _as_periodic_triple_list(
            triples_from_neighborlist(nbl, structure, cutoff)
        ):
            out.append(
                _periodic_triple(
                    j + b * nat,
                    (i + b * nat, shift_i),
                    (k + b * nat, shift_k),
                )
            )
    return out


def test_periodic_triples_batch_of_cells_of_different_sizes() -> None:
    """A 2-atom and a 3-atom cell: the first system carries a padding
    atom (`numbers == 0`), `nat` is the padded count 3, and the batch's
    triples are each system's own, offset by `b * nat`. No triple may
    touch the padding atom, flat index 2."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    small, larger = _small_and_larger_cell(dd)
    cutoff = 7.1

    expected = _triples_of_each_system([small, larger], 3, cutoff)
    assert len(expected) > 0

    batch = pack_structures([small, larger])
    assert batch.numbers.tolist() == [[1, 1, 0], [1, 1, 1]]
    nbl = build_neighborlist(batch, cutoff=cutoff, tile=16)
    chunks = list(triples_from_neighborlist(nbl, batch, cutoff))
    got = _as_periodic_triple_list(iter(chunks))

    assert len(got) == len(set(got))
    assert set(got) == set(expected)
    for chunk in chunks:
        for atoms in (chunk.idx_i, chunk.idx_j, chunk.idx_k):
            assert not bool((atoms == 2).any())


def test_periodic_triples_molecule_in_a_batch_of_cells() -> None:
    """A molecule packed next to a cell (identity lattice, no periodic
    axis) has exactly its molecular triples, all with zero shifts."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    cutoff = 7.1

    molecule_positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.5, 0.2, 0.1], [0.3, 2.0, 1.1]], **dd
    )
    molecule = hydrogens(molecule_positions)
    as_cell = hydrogens(
        molecule_positions,
        lattice=torch.eye(3, **dd),
        periodic=torch.tensor([False, False, False], device=DEVICE),
    )
    _, cell = _small_and_larger_cell(dd)
    nat = 3

    alone = build_neighborlist(molecule, cutoff=cutoff, tile=16)
    expected = [
        _periodic_triple(j, (i, si), (k, sk))
        for i, si, j, k, sk in _as_periodic_triple_list(
            triples_from_neighborlist(alone, molecule, cutoff)
        )
    ]
    assert len(expected) > 0
    expected_cell = [
        (i + nat, si, j + nat, k + nat, sk)
        for i, si, j, k, sk in _triples_of_each_system([cell], nat, cutoff)
    ]
    assert len(expected_cell) > 0

    batch = pack_structures([as_cell, cell])
    nbl = build_neighborlist(batch, cutoff=cutoff, tile=16)
    got = _as_periodic_triple_list(
        triples_from_neighborlist(nbl, batch, cutoff)
    )

    assert len(got) == len(set(got))
    assert set(got) == set(expected) | set(expected_cell)
    molecule_rows = [t for t in got if t[2] < nat]
    assert set(molecule_rows) == set(expected)
    assert all(t[1] == (0, 0, 0) and t[4] == (0, 0, 0) for t in molecule_rows)


def test_periodic_triples_unequal_batch_vectorized_matches_loop() -> None:
    """Loop and vectorized enumeration agree on a batch of unequal sizes
    and lattices, unchunked and chunked: the per-centre cell of the loop
    (`centre // atoms_per_system`) against the per-triple gather of the
    vectorized path."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    small, larger = _small_and_larger_cell(dd)
    cutoff = 7.1

    batch = pack_structures([small, larger])
    nbl = build_neighborlist(batch, cutoff=cutoff, tile=16)
    flat = batch.positions.reshape(-1, 3)
    args = (
        flat.shape[0],
        nbl.idx_i[nbl.mask],
        nbl.idx_j[nbl.mask],
        nbl.shift[nbl.mask],
        flat,
        batch.lattice,
        3,
    )

    expected = set(_triples_of_each_system([small, larger], 3, cutoff))
    for chunk_size in (None, 1, 7):
        loop = _as_periodic_triple_list(
            _iter_triple_chunks_loop(*args, cutoff, chunk_size=chunk_size)
        )
        vectorized = _as_periodic_triple_list(
            _iter_triple_chunks_vectorized(*args, cutoff, chunk_size=chunk_size)
        )
        assert len(loop) == len(set(loop))
        assert len(vectorized) == len(set(vectorized))
        assert set(loop) == expected, f"loop, chunk_size={chunk_size}"
        assert set(vectorized) == expected, f"vec, chunk_size={chunk_size}"


def test_periodic_triples_chunks_straddle_the_boundary_between_systems() -> (
    None
):
    """With small chunks, a chunk holds rows of both systems and another
    chunk starts inside a system; the union is still every system's own
    triples."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    small, larger = _small_and_larger_cell(dd)
    cutoff = 7.1

    batch = pack_structures([small, larger])
    nbl = build_neighborlist(batch, cutoff=cutoff, tile=16)
    expected = set(_triples_of_each_system([small, larger], 3, cutoff))

    for chunk_size in (1, 7):
        chunks = list(
            triples_from_neighborlist(nbl, batch, cutoff, chunk_size=chunk_size)
        )
        assert all(c.idx_i.shape[0] <= chunk_size for c in chunks)
        got = _as_periodic_triple_list(iter(chunks))
        assert len(got) == len(set(got))
        assert set(got) == expected, f"chunk_size={chunk_size}"

    # The loop path buffers kept rows across centres, so a chunk of 7
    # reaches from one system into the next. (The vectorized path cuts
    # the flat candidate range instead, where a boundary chunk may keep
    # rows of one system only.) Called directly, on the batch's CPU copy.
    cpu_batch = Structure(
        numbers=batch.numbers.cpu(),
        positions=batch.positions.cpu(),
        lattice=batch.lattice.cpu(),
        periodic=batch.periodic.cpu(),
    )
    cpu_nbl = build_neighborlist(cpu_batch, cutoff=cutoff, tile=16)
    systems_per_chunk = [
        set((chunk.idx_j // 3).tolist())
        for chunk in _iter_triple_chunks_loop(
            6,
            cpu_nbl.idx_i[cpu_nbl.mask],
            cpu_nbl.idx_j[cpu_nbl.mask],
            cpu_nbl.shift[cpu_nbl.mask],
            cpu_batch.positions.reshape(-1, 3),
            cpu_batch.lattice,
            3,
            cutoff,
            chunk_size=7,
        )
    ]
    assert any(
        len(systems) == 2 for systems in systems_per_chunk
    ), "no chunk straddles the boundary between the systems"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_periodic_triples_unequal_batch_cuda_match_cpu() -> None:
    """The unequal batch on CUDA (vectorized enumeration) equals each
    system's own triples from the CPU loop."""
    dd: DD = {"device": torch.device("cpu"), "dtype": torch.double}
    small, larger = _small_and_larger_cell(dd)
    cutoff = 7.1

    expected = set(_triples_of_each_system([small, larger], 3, cutoff))
    assert len(expected) > 0

    packed = pack_structures([small, larger])
    batch = Structure(
        numbers=packed.numbers.cuda(),
        positions=packed.positions.cuda(),
        lattice=packed.lattice.cuda(),
        periodic=packed.periodic.cuda(),
    )
    nbl = build_neighborlist(batch, cutoff=cutoff, tile=16)
    got = _as_periodic_triple_list(
        triples_from_neighborlist(nbl, batch, cutoff)
    )
    assert set(got) == expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_periodic_triples_cuda_match_cpu() -> None:
    """The periodic triples on CUDA (vectorized enumeration) equal the CPU
    ones (loop enumeration)."""
    dd: DD = {"device": torch.device("cpu"), "dtype": torch.double}
    positions, lattice = _triclinic_cell(dd)
    cutoff = 8.1

    cpu = hydrogens(positions, lattice=lattice)
    cuda = hydrogens(positions.cuda(), lattice=lattice.cuda())

    on_cpu = _as_periodic_triple_list(
        triples_from_neighborlist(
            build_neighborlist(cpu, cutoff=cutoff, tile=16), cpu, cutoff
        )
    )
    on_cuda = _as_periodic_triple_list(
        triples_from_neighborlist(
            build_neighborlist(cuda, cutoff=cutoff, tile=16), cuda, cutoff
        )
    )
    assert len(on_cpu) > 0
    assert set(on_cuda) == set(on_cpu)


def test_triple_list_loop_matches_vectorized() -> None:
    """`_iter_triple_chunks_loop`'s and `_iter_triple_chunks_vectorized`'s
    unchunked (``chunk_size=None``) output must be the exact same triple
    set: `triples_from_neighborlist`'s device dispatch
    (``positions.device.type == "cpu"`` -> loop, else -> vectorized) is
    only safe if both sides genuinely agree. Both internal generators are
    therefore called directly here, on the same CPU input, bypassing the
    dispatch entirely -- mirrors the "build both, convert to a set of
    tuples, compare" idiom of ``dense_pair_set``/``neighborlist_pair_set``
    in ``test/test_neighbor/test_list.py``.

    Three cases, each asserting the same exact-match:

    1. A dense condensed box with ``nbl`` built at exactly ``cutoff``
       (skin=0). Several centre atoms end up with degree >= 3, so the
       vectorized path's triangular-index inversion is exercised across
       multiple rows of a centre's block, not just a single pair.
    2. The same box, but with ``nbl`` built with nonzero skin, so some of
       its real pairs extend past ``cutoff``. Both the loop and vectorized
       generators must prune every one of the three sides (not just one)
       against ``cutoff`` -- this case proves that pruning is actually
       exercised, not just present in the source.
    3. A sparse box where several centre atoms end up with degree < 2 (no
       candidate triples at all), and one with a cutoff small enough that
       there are no candidate triples anywhere. This is the one regime
       where ``torch.searchsorted``'s zero-length-block handling in
       ``_iter_triple_chunks_vectorized`` could misattribute a triple to
       the wrong centre (or the empty-list early return) if it were
       wrong -- a box where every centre has degree >= 2 would never
       reach that code at all.
    """

    def check(
        nat: int,
        density: float,
        cutoff_value: float,
        seed_positions: int,
        skin: float = 0.0,
    ) -> tuple[int, Tensor]:
        dd: DD = {"device": DEVICE, "dtype": torch.double}
        positions = _random_positions(nat, density, seed_positions, dd)

        cutoff = cutoff_value
        nbl = build_neighborlist(
            hydrogens(positions), cutoff=cutoff_value, tile=16, skin=skin
        )
        real_i = nbl.idx_i[nbl.mask]
        real_j = nbl.idx_j[nbl.mask]

        loop_i, loop_j, loop_k = _collect(
            _iter_triple_chunks_loop(
                nat,
                real_i,
                real_j,
                nbl.shift[nbl.mask],
                positions,
                None,
                nat,
                cutoff,
                chunk_size=None,
            )
        )
        vec_i, vec_j, vec_k = _collect(
            _iter_triple_chunks_vectorized(
                nat,
                real_i,
                real_j,
                nbl.shift[nbl.mask],
                positions,
                None,
                nat,
                cutoff,
                chunk_size=None,
            )
        )

        loop_set = _as_triple_set(loop_i, loop_j, loop_k)
        vec_set = _as_triple_set(vec_i, vec_j, vec_k)

        assert (
            len(loop_set) == loop_i.shape[0]
        ), "loop path listed a triple twice"
        assert (
            len(vec_set) == vec_i.shape[0]
        ), "vectorized path listed a triple twice"
        assert loop_set == vec_set

        _, _, _, degree = _oriented_csr(
            nat, real_i, real_j, nbl.shift[nbl.mask]
        )
        return len(loop_set), degree

    # Case 1: dense, condensed box, `nbl` built at exactly `cutoff`.
    n_triples, degree = check(
        nat=50, density=0.05, cutoff_value=8.0, seed_positions=4
    )
    assert n_triples > 0, "dense case too sparse to exercise any triples"
    assert bool((degree >= 3).any()), "dense case never reaches degree >= 3"

    # Case 2: same box, `nbl` built with skin -- some real pairs now extend
    # past `cutoff`, so pruning must actually discard candidate triples it
    # would otherwise keep at skin=0.
    n_triples_skin, _ = check(
        nat=50,
        density=0.05,
        cutoff_value=8.0,
        seed_positions=4,
        skin=4.0,
    )
    assert n_triples_skin > 0

    # Case 3: sparse box -- a genuine mix of centres with degree < 2 (no
    # candidate triples at all, i.e. a zero-length block in the flat
    # enumeration) alongside centres with degree >= 2 that do contribute
    # triples, so `total > 0` overall but `searchsorted` must still route
    # around the zero-length blocks correctly.
    n_triples_sparse, degree_sparse = check(
        nat=50, density=0.005, cutoff_value=4.0, seed_positions=6
    )
    assert n_triples_sparse > 0, "sparse case has no triples to compare"
    assert bool((degree_sparse < 2).any()), (
        "sparse case has no low-degree centres to exercise the "
        "zero-length-block case"
    )
    assert bool(
        (degree_sparse >= 2).any()
    ), "sparse case has no centres that actually contribute triples"

    # Case 4: cutoff small enough that there are no candidate triples at
    # all -- exercises `_iter_triple_chunks_vectorized`'s `total == 0`
    # early return against the loop path's equivalent empty return.
    n_triples_empty, _ = check(
        nat=20, density=0.05, cutoff_value=0.01, seed_positions=8
    )
    assert n_triples_empty == 0


@pytest.mark.parametrize("degree", [2, 3, 4, 5, 17, 32, 33, 100])
def test_triangular_indices_matches_triu_indices(degree: int) -> None:
    """Every flat index of a centre's triangle must map to the same
    `(a, b)` slot pair that `torch.triu_indices` lists at that position."""
    expected_a, expected_b = torch.triu_indices(
        degree, degree, offset=1, device=DEVICE
    )
    flat = torch.arange(expected_a.shape[0], device=DEVICE)
    _, row_offset = _triangle_rows(torch.tensor([degree], device=DEVICE))

    a, b = _triangular_indices(flat, row_offset, 0)

    assert torch.equal(a, expected_a)
    assert torch.equal(b, expected_b)


def test_triangular_indices_several_centres() -> None:
    """The flat numbering of several centres, including some with fewer
    than two neighbours and so no rows at all, as the vectorized path
    inverts it."""
    degrees = [2, 0, 7, 1, 3, 12]
    degree = torch.tensor(degrees, device=DEVICE)

    expected_a_parts, expected_b_parts, centre_parts = [], [], []
    for centre, d in enumerate(degrees):
        expected_a, expected_b = torch.triu_indices(
            d, d, offset=1, device=DEVICE
        )
        expected_a_parts.append(expected_a)
        expected_b_parts.append(expected_b)
        centre_parts.append(torch.full_like(expected_a, centre))
    centre = torch.cat(centre_parts)
    flat = torch.arange(centre.shape[0], device=DEVICE)

    first_row, row_offset = _triangle_rows(degree)
    a, b = _triangular_indices(flat, row_offset, first_row[centre])

    assert torch.equal(a, torch.cat(expected_a_parts))
    assert torch.equal(b, torch.cat(expected_b_parts))


def test_triangular_indices_large_degree_at_row_boundaries() -> None:
    """At a degree far beyond any real neighbour list, the first and last
    index of rows spread over the whole triangle must still map exactly,
    checked against the triangular numbers in Python integers."""
    degree = 2**20 + 3
    rows = [0, 1, 2, 1000, 2**19, 2**20 - 7, degree - 3, degree - 2]

    def row_start(row: int) -> int:
        return row * (2 * degree - 1 - row) // 2

    flat_list, expected_a, expected_b = [], [], []
    for row in rows:
        first = row_start(row)
        last = row_start(row + 1) - 1
        flat_list += [first, last]
        expected_a += [row, row]
        expected_b += [row + 1, degree - 1]

    flat = torch.tensor(flat_list, device=DEVICE)
    _, row_offset = _triangle_rows(torch.tensor([degree], device=DEVICE))

    a, b = _triangular_indices(flat, row_offset, 0)

    assert a.tolist() == expected_a
    assert b.tolist() == expected_b


def test_triple_chunks_vectorized_yields_no_empty_chunk() -> None:
    """A chunk whose candidates all fail the cutoff must be skipped, as
    the loop path skips it, rather than yielded empty. The list is built
    at a much larger cutoff than the triples, so most chunks of the flat
    candidate range hold no triple at all."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    nat = 40
    positions = _random_positions(nat, density=0.05, seed=4, dd=dd)
    list_cutoff = 8.0
    triple_cutoff = 3.0

    nbl = build_neighborlist(hydrogens(positions), cutoff=list_cutoff, tile=16)
    real_i = nbl.idx_i[nbl.mask]
    real_j = nbl.idx_j[nbl.mask]

    vectorized = list(
        _iter_triple_chunks_vectorized(
            nat,
            real_i,
            real_j,
            nbl.shift[nbl.mask],
            positions,
            None,
            nat,
            triple_cutoff,
            chunk_size=5,
        )
    )
    loop = list(
        _iter_triple_chunks_loop(
            nat,
            real_i,
            real_j,
            nbl.shift[nbl.mask],
            positions,
            None,
            nat,
            triple_cutoff,
            chunk_size=5,
        )
    )

    assert len(vectorized) > 0, "no triple within the cutoff at all"
    assert all(chunk.idx_i.shape[0] > 0 for chunk in vectorized)
    assert all(chunk.idx_i.shape[0] > 0 for chunk in loop)
    assert _as_triple_set(*_collect(iter(vectorized))) == _as_triple_set(
        *_collect(iter(loop))
    )


def test_triple_chunks_loop_matches_unchunked() -> None:
    """`_iter_triple_chunks_loop`'s chunking must enumerate exactly the
    same triples regardless of ``chunk_size``, for both ways it bounds the
    candidate-triple working set: across centres (buffering and flushing
    before every centre is processed) and *within* a single centre
    (falling back from `torch.triu_indices` to `_triangular_indices` once
    one centre's own `C(degree, 2)` candidate count alone would exceed
    `chunk_size`).

    `chunk_size` in {1, 2, 3, 7} are deliberately adversarial and smaller
    than any single centre's own candidate count on this box (several
    centres reach degree >= 3, i.e. `C(degree, 2) >= 3`), so every one of
    them forces the within-centre fallback branch on at least one centre,
    not just the across-centre buffering.
    """
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    nat = 12
    positions = _random_positions(nat, density=0.05, seed=4, dd=dd)
    cutoff = 8.0

    nbl = build_neighborlist(hydrogens(positions), cutoff=cutoff, tile=16)
    real_i = nbl.idx_i[nbl.mask]
    real_j = nbl.idx_j[nbl.mask]

    _, _, _, degree = _oriented_csr(nat, real_i, real_j, nbl.shift[nbl.mask])
    assert bool((degree >= 3).any()), (
        "box has no degree >= 3 centre to exercise the within-centre "
        "fallback at all"
    )

    unchunked_i, unchunked_j, unchunked_k = _collect(
        _iter_triple_chunks_loop(
            nat,
            real_i,
            real_j,
            nbl.shift[nbl.mask],
            positions,
            None,
            nat,
            cutoff,
            chunk_size=None,
        )
    )
    unchunked_set = _as_triple_set(unchunked_i, unchunked_j, unchunked_k)
    assert len(unchunked_set) == unchunked_i.shape[0]
    assert unchunked_i.shape[0] > 0, "box too sparse to exercise chunking"

    for chunk_size in (1, 2, 3, 7):
        chunk_i_parts: list[Tensor] = []
        chunk_j_parts: list[Tensor] = []
        chunk_k_parts: list[Tensor] = []
        n_chunks = 0
        for chunk in _iter_triple_chunks_loop(
            nat,
            real_i,
            real_j,
            nbl.shift[nbl.mask],
            positions,
            None,
            nat,
            cutoff,
            chunk_size=chunk_size,
        ):
            n_chunks += 1
            assert chunk.idx_i.shape[0] <= chunk_size
            chunk_i_parts.append(chunk.idx_i)
            chunk_j_parts.append(chunk.idx_j)
            chunk_k_parts.append(chunk.idx_k)

        assert n_chunks > 1, (
            f"chunk_size={chunk_size} did not force multiple chunks -- "
            "not exercising the chunk boundary at all"
        )

        chunked_i = torch.cat(chunk_i_parts)
        chunked_j = torch.cat(chunk_j_parts)
        chunked_k = torch.cat(chunk_k_parts)
        chunked_set = _as_triple_set(chunked_i, chunked_j, chunked_k)

        assert (
            len(chunked_set) == chunked_i.shape[0]
        ), f"chunk_size={chunk_size} listed a triple twice"
        assert chunked_set == unchunked_set, (
            f"chunk_size={chunk_size} disagrees with the unchunked "
            "loop triple list"
        )


def test_triple_chunks_vectorized_matches_unchunked() -> None:
    """`_iter_triple_chunks_vectorized`'s chunking must enumerate exactly
    the same triples regardless of where `chunk_size` happens to cut the
    flat candidate-triple index: `torch.searchsorted` (centre recovery)
    and `_triangular_indices` (within-centre inversion) both only depend
    on where each flat index falls relative to `tri_offset`, not on where
    the chunk's own range starts, but that is exactly the thing a
    chunk-boundary bug would get wrong first. Called directly on CPU,
    bypassing `triples_from_neighborlist`'s device dispatch -- same idiom
    as `test_triple_list_loop_matches_vectorized` above -- since the
    vectorized generator is otherwise only exercised on CUDA in the
    device-dispatched path, which this CPU-only test suite never reaches.

    `chunk_size` in {1, 2, 3, 7} are deliberately adversarial: 1 forces
    *every* flat index into its own chunk (one `searchsorted` call per
    single candidate), the maximally hostile case for the centre-recovery
    machinery; 2, 3, 7 are small enough to routinely split a single
    centre's own candidate block across a chunk boundary on this box's
    degree distribution (10-11, from a quick standalone check), which
    `chunk_size=1` alone would not distinguish from an off-by-one at
    exactly the first index of each block.
    """
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    nat = 12
    positions = _random_positions(nat, density=0.05, seed=4, dd=dd)
    cutoff = 8.0

    nbl = build_neighborlist(hydrogens(positions), cutoff=cutoff, tile=16)
    real_i = nbl.idx_i[nbl.mask]
    real_j = nbl.idx_j[nbl.mask]

    unchunked_i, unchunked_j, unchunked_k = _collect(
        _iter_triple_chunks_vectorized(
            nat,
            real_i,
            real_j,
            nbl.shift[nbl.mask],
            positions,
            None,
            nat,
            cutoff,
            chunk_size=None,
        )
    )
    unchunked_set = _as_triple_set(unchunked_i, unchunked_j, unchunked_k)
    assert len(unchunked_set) == unchunked_i.shape[0]
    assert unchunked_i.shape[0] > 0, "box too sparse to exercise chunking"

    for chunk_size in (1, 2, 3, 7):
        chunk_i_parts: list[Tensor] = []
        chunk_j_parts: list[Tensor] = []
        chunk_k_parts: list[Tensor] = []
        n_chunks = 0
        for chunk in _iter_triple_chunks_vectorized(
            nat,
            real_i,
            real_j,
            nbl.shift[nbl.mask],
            positions,
            None,
            nat,
            cutoff,
            chunk_size=chunk_size,
        ):
            n_chunks += 1
            assert chunk.idx_i.shape[0] <= chunk_size
            chunk_i_parts.append(chunk.idx_i)
            chunk_j_parts.append(chunk.idx_j)
            chunk_k_parts.append(chunk.idx_k)

        assert n_chunks > 1, (
            f"chunk_size={chunk_size} did not force multiple chunks -- "
            "not exercising the chunk boundary at all"
        )

        chunked_i = torch.cat(chunk_i_parts)
        chunked_j = torch.cat(chunk_j_parts)
        chunked_k = torch.cat(chunk_k_parts)
        chunked_set = _as_triple_set(chunked_i, chunked_j, chunked_k)

        assert (
            len(chunked_set) == chunked_i.shape[0]
        ), f"chunk_size={chunk_size} listed a triple twice"
        assert chunked_set == unchunked_set, (
            f"chunk_size={chunk_size} disagrees with the unchunked "
            "vectorized triple list"
        )


def test_triples_from_neighborlist_matches_direct_loop() -> None:
    """`triples_from_neighborlist` is the public entry point: on a CPU
    tensor it must dispatch to the loop generator and reproduce exactly
    the same triple set, chunked or not -- this is the one test that goes
    through the public API end to end, rather than reaching into the
    private per-device generators directly."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    nat = 30
    positions = _random_positions(nat, density=0.05, seed=1, dd=dd)
    cutoff = 8.0

    structure = hydrogens(positions)
    nbl = build_neighborlist(structure, cutoff=cutoff, tile=16)
    real_i = nbl.idx_i[nbl.mask]
    real_j = nbl.idx_j[nbl.mask]

    expected_i, expected_j, expected_k = _collect(
        _iter_triple_chunks_loop(
            nat,
            real_i,
            real_j,
            nbl.shift[nbl.mask],
            positions,
            None,
            nat,
            cutoff,
            chunk_size=None,
        )
    )
    expected_set = _as_triple_set(expected_i, expected_j, expected_k)
    assert len(expected_set) > 0, "box too sparse to exercise any triples"

    # Unchunked via the public API.
    public_i, public_j, public_k = _collect(
        triples_from_neighborlist(nbl, structure, cutoff, chunk_size=None)
    )
    assert _as_triple_set(public_i, public_j, public_k) == expected_set

    # Chunked via the public API -- must reproduce the same set, and every
    # chunk must actually respect chunk_size.
    n_chunks = 0
    chunk_parts: list[tuple[Tensor, Tensor, Tensor]] = []
    for chunk in triples_from_neighborlist(
        nbl, structure, cutoff, chunk_size=5
    ):
        n_chunks += 1
        assert chunk.idx_i.shape[0] <= 5
        chunk_parts.append((chunk.idx_i, chunk.idx_j, chunk.idx_k))
    assert n_chunks > 1, "chunk_size=5 did not force multiple chunks"

    chunked_i = torch.cat([p[0] for p in chunk_parts])
    chunked_j = torch.cat([p[1] for p in chunk_parts])
    chunked_k = torch.cat([p[2] for p in chunk_parts])
    assert _as_triple_set(chunked_i, chunked_j, chunked_k) == expected_set

    # Default chunk_size (2_000_000) is a no-op on a box this small: a
    # single chunk holding every candidate triple.
    default_n_chunks = sum(
        1 for _ in triples_from_neighborlist(nbl, structure, cutoff)
    )
    assert default_n_chunks == 1


def test_triples_from_neighborlist_periodic_list_needs_the_lattice() -> None:
    """A periodic `nbl` paired with a structure without a lattice has no
    image translations to apply, so it must raise (via `check_compatible`)
    rather than return molecular-looking triples. It raises on the call,
    before any chunk is asked for."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], **dd
    )
    lattice = torch.eye(3, **dd) * 10.0
    cutoff = 5.0

    nbl = build_neighborlist(
        hydrogens(positions, lattice=lattice), cutoff=cutoff
    )
    assert nbl.periodic

    with pytest.raises(ValueError, match="lattice"):
        triples_from_neighborlist(nbl, hydrogens(positions), cutoff)


def test_triples_from_neighborlist_empty_nbl_yields_nothing() -> None:
    """A neighbour list with no real pairs at all (every atom isolated
    past `cutoff`) must yield zero chunks, not raise or fabricate a
    triple -- exercises `_iter_triple_chunks_vectorized`'s `total == 0`
    early return and the loop path's equivalent empty degree-2 skip via
    the public dispatcher."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [0.0, 100.0, 0.0]], **dd
    )
    cutoff = 1.0

    structure = hydrogens(positions)
    nbl = build_neighborlist(structure, cutoff=cutoff, tile=4)
    assert int(nbl.mask.sum()) == 0

    chunks = list(triples_from_neighborlist(nbl, structure, cutoff))
    assert chunks == []


def test_triples_from_neighborlist_batched_matches_each_system() -> None:
    """A batched list numbers atom `i` of system `b` as `b * nat + i`, so
    the triples of a batch are each system's own triples in that
    numbering, and none spans two systems."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    first = load_structure("mb16_43", "01", dd)
    second = load_structure("mb16_43", "SiH4", dd)
    batch = pack_structures([first, second])
    nat = batch.numbers.shape[-1]
    cutoff = 6.0

    expected_set: set[tuple[int, int, int]] = set()
    for system, structure in enumerate([first, second]):
        nbl = build_neighborlist(structure, cutoff=cutoff)
        trip_i, trip_j, trip_k = _collect(
            triples_from_neighborlist(nbl, structure, cutoff)
        )
        offset = system * nat
        expected_set |= _as_unordered_triple_set(
            trip_i + offset, trip_j + offset, trip_k + offset
        )
    assert len(expected_set) > 0, "no triples to compare"

    batch_nbl = build_neighborlist(batch, cutoff=cutoff)
    got_i, got_j, got_k = _collect(
        triples_from_neighborlist(batch_nbl, batch, cutoff)
    )
    assert _as_unordered_triple_set(got_i, got_j, got_k) == expected_set


def test_triples_from_neighborlist_rejects_a_list_of_another_shape() -> None:
    """A list built for one system indexes that system's atoms only.
    Paired with a batch of the same system, its indices would address the
    wrong positions, so it must raise instead of indexing out of range
    (on CUDA, a device-side assert that breaks the whole process)."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    single = load_structure("mb16_43", "01", dd)
    batch = pack_structures([single, single])
    cutoff = 6.0

    nbl = build_neighborlist(single, cutoff=cutoff)

    with pytest.raises(ValueError, match="shape"):
        triples_from_neighborlist(nbl, batch, cutoff)


def test_triples_from_neighborlist_rejects_a_list_below_the_cutoff() -> None:
    """A list built at a smaller cutoff misses pairs, so it misses every
    triple with an edge between the two cutoffs."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    structure = load_structure("mb16_43", "01", dd)

    nbl = build_neighborlist(structure, cutoff=4.0)

    with pytest.raises(ValueError, match="cutoff"):
        triples_from_neighborlist(nbl, structure, 6.0)


def test_triple_index_slices_reproduce_the_enumeration_on_demand() -> None:
    """`TripleIndex.chunk(start, stop)` is a pure function of the slice and
    the geometry: slices of any size, asked for in any order and more than
    once, give the triples of the unchunked enumeration, periodic and
    batched included. This is what lets a consumer regenerate the indices
    in the backward pass."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    small, larger = _small_and_larger_cell(dd)
    cutoff = 7.1
    batch = pack_structures([small, larger])
    nbl = build_neighborlist(batch, cutoff=cutoff, tile=16)
    flat = batch.positions.reshape(-1, 3)

    index = TripleIndex(
        flat.shape[0],
        nbl.idx_i[nbl.mask],
        nbl.idx_j[nbl.mask],
        nbl.shift[nbl.mask],
        3,
        cutoff,
        periodic=True,
    )
    expected = set(_triples_of_each_system([small, larger], 3, cutoff))
    assert index.total > len(expected)

    for size in (1, 5, index.total):
        starts = list(range(0, index.total, size))[::-1]  # reversed order
        got: list[PeriodicTriple] = []
        for start in starts:
            stop = min(start + size, index.total)
            chunk = index.chunk(start, stop, flat, batch.lattice)
            again = index.chunk(start, stop, flat, batch.lattice)
            assert torch.equal(chunk.idx_j, again.idx_j)
            got += _as_periodic_triple_list(iter([chunk]))
        assert len(got) == len(set(got))
        assert set(got) == expected, f"slice size {size}"
