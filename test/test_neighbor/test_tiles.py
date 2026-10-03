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
Test the tile-based neighbour screen: `Tiles` and `tile_pairs`.

No physics is involved here, only geometry, so every check asserts exact
agreement (equality as a set of atom-pair indices) against an explicit
`(nat, nat)` distance matrix, rather than an approximate tolerance.
"""

from __future__ import annotations

import math

import pytest
import torch

from tad_mctc.data.structures import get_structure
from tad_mctc.neighbor._tiles import Tiles, tile_pairs
from tad_mctc.typing import DD, Tensor

from ..conftest import DEVICE


def dense_pair_set(positions: Tensor, cutoff: float) -> set[tuple[int, int]]:
    """All atom pairs ``i <= j`` within ``cutoff``, from a dense distance
    matrix."""
    nat = positions.shape[0]
    distances = torch.cdist(positions, positions)

    row_index = torch.arange(nat).unsqueeze(-1).expand(nat, nat)
    col_index = torch.arange(nat).unsqueeze(0).expand(nat, nat)

    upper_triangle = row_index <= col_index
    within_cutoff = distances <= cutoff
    keep = upper_triangle & within_cutoff

    rows, cols = keep.nonzero(as_tuple=True)
    return set(zip(rows.tolist(), cols.tolist()))


def tile_pair_atom_set(
    positions: Tensor, cutoff: float, tile: int
) -> tuple[set[tuple[int, int]], Tiles]:
    """Expand `tile_pairs`'s candidate tile pairs into an exact atom-pair
    set, by applying a plain distance filter within each surviving tile
    pair. This is what B2.3 calls "the pair set from Tiles + tile_pairs +
    a distance filter"."""
    tiles = Tiles(positions, tile=tile)
    tile_a, tile_b = tile_pairs(tiles, cutoff)

    pairs: set[tuple[int, int]] = set()
    for a, b in zip(tile_a.tolist(), tile_b.tolist()):
        atoms_a = tiles.index[a][tiles.valid[a]].tolist()
        atoms_b = tiles.index[b][tiles.valid[b]].tolist()

        for i in atoms_a:
            for j in atoms_b:
                lo, hi = (i, j) if i <= j else (j, i)
                distance = (positions[lo] - positions[hi]).norm()
                if distance <= cutoff:
                    pairs.add((lo, hi))

    return pairs, tiles


_GEOMETRIES: list[tuple[str, str]] = [
    ("mb16_43", "H2"),
    ("heavy28", "h2o"),
    ("mb16_43", "CH4"),
    ("mb16_43", "SiH4"),
    ("other", "C6H5I-CH3SH"),
    ("mb16_43", "01"),
]


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("collection,record", _GEOMETRIES)
def test_matches_dense_distance_matrix(
    dtype: torch.dtype, collection: str, record: str
) -> None:
    """The tiled pair set must equal, as a set, the dense brute-force one."""
    positions = get_structure(
        collection, record, device=DEVICE, dtype=dtype
    ).positions

    cutoff = 6.0
    got, _ = tile_pair_atom_set(positions, cutoff, tile=4)
    want = dense_pair_set(positions, cutoff)

    assert got == want


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_tile_radius_within_one_bin_diagonal(dtype: torch.dtype) -> None:
    """No tile spans more than one bin diagonal (what makes the stencil
    width in `tile_pairs` independent of system size)."""
    dd: DD = {"device": DEVICE, "dtype": dtype}

    torch.manual_seed(0)
    positions = torch.randn(300, 3, **dd) * 15.0

    tiles = Tiles(positions, tile=32)
    extent = tiles.hi - tiles.lo
    radius = 0.5 * extent.norm(dim=-1)

    bin_diagonal = tiles.width.norm()
    assert bool((radius <= bin_diagonal + 1e-9).all())


@pytest.mark.parametrize(
    ("positions", "tile"),
    [
        # The smallest case: 4 atoms, two close pairs far
        # apart. `max_cells = 4 * ceil(4 / 32) = 4`, but the pre-coarsening
        # grid along the long axis is enormous, so the halving loop fires.
        (
            torch.tensor(
                [[0.0, 0, 0], [1.0, 0, 0], [50.0, 0, 0], [51.0, 0, 0]],
                dtype=torch.double,
            ),
            32,
        ),
        # Three close pairs spread along one axis: same mechanism, more
        # tiles, to check the invariant holds beyond a single tile pair.
        (
            torch.tensor(
                [
                    [0.0, 0, 0],
                    [1.0, 0, 0],
                    [50.0, 0, 0],
                    [51.0, 0, 0],
                    [100.0, 0, 0],
                    [101.0, 0, 0],
                ],
                dtype=torch.double,
            ),
            32,
        ),
        # Many atoms scattered along a long, thin axis with a small tile:
        # forces the halving loop via `max_cells`, not just `nat == tile`.
        (
            torch.cat(
                [
                    torch.linspace(0.0, 300.0, 50, dtype=torch.double).reshape(
                        -1, 1
                    ),
                    torch.zeros(50, 2, dtype=torch.double),
                ],
                dim=1,
            ),
            16,
        ),
    ],
    ids=["two-pairs", "three-pairs", "elongated-many"],
)
def test_tile_bbox_within_coarsened_bin_diagonal(
    positions: Tensor, tile: int
) -> None:
    """No tile's bounding-box radius may exceed one *coarsened* bin
    diagonal, computed independently of `Tiles`' own `width` attribute
    (straight from `positions` and the final `n_axis`), on inputs chosen
    to force the coarsening (halving) loop in `Tiles.__init__` to fire.

    A bin width kept from before the halving would let the last bin along
    an axis silently absorb atoms that are actually many bins away."""
    tiles = Tiles(positions, tile=tile)

    extent = positions.max(dim=0).values - positions.min(dim=0).values
    coarsened_width = extent.clamp_min(1e-6) / tiles.n_axis
    bin_diagonal = coarsened_width.norm()

    tile_extent = tiles.hi - tiles.lo
    radius = 0.5 * tile_extent.norm(dim=-1)

    assert bool((radius <= bin_diagonal + 1e-9).all())


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_tiles_partition_the_atoms(dtype: torch.dtype) -> None:
    """Every atom belongs to exactly one tile slot."""
    dd: DD = {"device": DEVICE, "dtype": dtype}

    torch.manual_seed(1)
    positions = torch.randn(97, 3, **dd) * 10.0

    tiles = Tiles(positions, tile=16)
    covered = torch.zeros(positions.shape[0], dtype=torch.bool)

    for atom_row, valid_row in zip(tiles.index, tiles.valid):
        for atom_index, is_valid in zip(atom_row.tolist(), valid_row.tolist()):
            if not is_valid:
                continue
            assert not covered[atom_index], "atom assigned to two tiles"
            covered[atom_index] = True

    assert bool(covered.all())


@pytest.mark.parametrize(
    "nat", [0, 1, 2], ids=["empty", "one-atom", "two-atom"]
)
def test_small_systems_do_not_crash(nat: int) -> None:
    """Degenerate small systems must build and query without raising."""
    torch.manual_seed(2)
    positions = torch.randn(nat, 3)

    tiles = Tiles(positions, tile=4)
    # Two atoms can land in different bins, so `ntile` is only bounded by
    # `nat`, not necessarily `ceil(nat / tile)`.
    assert 0 <= tiles.ntile <= max(nat, 1)

    a, b = tile_pairs(tiles, cutoff=5.0)
    assert a.shape == b.shape

    want = dense_pair_set(positions, cutoff=5.0)
    got, _ = tile_pair_atom_set(positions, cutoff=5.0, tile=4)
    assert got == want


def test_planar_system_matches_dense() -> None:
    """A zero-extent axis (planar molecule) must not blow up the bin
    volume or crash the bin count."""
    torch.manual_seed(3)
    positions = torch.randn(40, 3)
    positions[:, 2] = 0.0

    cutoff = 3.0
    got, _ = tile_pair_atom_set(positions, cutoff, tile=5)
    want = dense_pair_set(positions, cutoff)
    assert got == want


def test_linear_system_matches_dense() -> None:
    """Two zero-extent axes (linear molecule) must not blow up the bin
    volume or crash the bin count."""
    positions = torch.zeros(20, 3)
    positions[:, 0] = torch.linspace(0, 30, 20)

    cutoff = 4.0
    got, _ = tile_pair_atom_set(positions, cutoff, tile=4)
    want = dense_pair_set(positions, cutoff)
    assert got == want


def test_thin_slab_stays_under_max_cells() -> None:
    """A thin, wide slab must not produce an unbounded bin grid."""
    torch.manual_seed(4)
    nat = 200
    tile = 16

    positions = torch.rand(nat, 3)
    positions[:, :2] *= 50.0
    positions[:, 2] *= 0.01

    tiles = Tiles(positions, tile=tile)
    max_cells = 4 * math.ceil(nat / tile)
    assert tiles.ncells <= max_cells

    cutoff = 5.0
    got, _ = tile_pair_atom_set(positions, cutoff, tile=tile)
    want = dense_pair_set(positions, cutoff)
    assert got == want


def test_cutoff_larger_than_grid_still_exact() -> None:
    """A cutoff several times the box edge would make an unclamped
    stencil span exceed the grid; the clamp in `tile_pairs` must still
    reproduce the exact brute-force pair set (not merely avoid crashing)."""
    torch.manual_seed(5)
    positions = torch.rand(30, 3) * 2.0  # box edge ~2

    cutoff = 50.0
    got, _ = tile_pair_atom_set(positions, cutoff, tile=4)
    want = dense_pair_set(positions, cutoff)
    assert got == want


def test_no_pair_appears_twice() -> None:
    """`a <= b` alone would pass trivially; check it together with the
    exact-match tests above, not as a substitute for them."""
    torch.manual_seed(6)
    positions = torch.randn(50, 3) * 8.0

    tiles = Tiles(positions, tile=8)
    a, b = tile_pairs(tiles, cutoff=5.0)

    pairs = list(zip(a.tolist(), b.tolist()))
    assert len(pairs) == len(set(pairs))
    assert bool((a <= b).all())


@pytest.mark.parametrize("anchor_fraction", [0.0, 0.1, 1.0])
def test_tile_pairs_with_anchors_are_the_pairs_with_an_anchor_tile(
    anchor_fraction: float,
) -> None:
    """With `anchors`, exactly the tile pairs that contain an anchor tile
    come back (as `a <= b` pairs, each once), the same ones a full
    enumeration finds. The pairs are enumerated from the anchor tiles
    only."""
    torch.manual_seed(5)
    positions = torch.rand(300, 3, device=DEVICE) * 40.0
    tiles = Tiles(positions, tile=8)
    anchors = torch.rand(tiles.ntile, device=DEVICE) < anchor_fraction

    all_a, all_b = tile_pairs(tiles, 7.0)
    got_a, got_b = tile_pairs(tiles, 7.0, anchors=anchors)

    want = {
        (a, b)
        for a, b in zip(all_a.tolist(), all_b.tolist())
        if bool(anchors[a]) or bool(anchors[b])
    }
    got = list(zip(got_a.tolist(), got_b.tolist()))
    assert len(got) == len(set(got)), "a tile pair was returned twice"
    assert set(got) == want
    assert bool((got_a <= got_b).all())


@pytest.mark.parametrize(
    "box",
    [
        (40.0, 40.0, 40.0),
        (60.0, 60.0, 0.0),
        (5.0, 5.0, 120.0),
        (90.0, 3.0, 30.0),
    ],
    ids=["cube", "flat-slab", "rod", "anisotropic"],
)
def test_tile_pairs_forward_stencil_finds_every_tile_pair(
    box: tuple[float, float, float],
) -> None:
    """Without `anchors`, only the forward half of the bin stencil is
    searched. It finds the same tile pairs as the full stencil, which
    `anchors` marking every tile searches."""
    torch.manual_seed(6)
    extent = torch.tensor(box, device=DEVICE)
    positions = torch.rand(400, 3, device=DEVICE) * extent
    tiles = Tiles(positions, tile=8)
    every_tile = torch.ones(tiles.ntile, dtype=torch.bool, device=DEVICE)

    half_a, half_b = tile_pairs(tiles, 7.0)
    full_a, full_b = tile_pairs(tiles, 7.0, anchors=every_tile)

    half = list(zip(half_a.tolist(), half_b.tolist()))
    full = set(zip(full_a.tolist(), full_b.tolist()))
    assert len(half) == len(set(half)), "a tile pair was returned twice"
    assert set(half) == full


@pytest.mark.cuda
def test_tiles_and_tile_pairs_stay_on_input_device() -> None:
    """`Tiles` and `tile_pairs` must place every tensor they build on
    `positions`' own device, not on whatever the ambient default device
    happens to be. Several internal tensors (`torch.arange`, `torch.zeros`
    for `index`/`valid`, the stencil offsets, the bin-id and candidate
    index arrays) are built inside `Tiles`, and a missing `device=` on any
    of them only shows up when the ambient default device differs from
    `positions`' device -- exactly the case here, since this test never
    touches the global default."""
    torch.manual_seed(7)
    positions = torch.randn(40, 3, device="cuda") * 6.0

    tiles = Tiles(positions, tile=8)
    assert tiles.index.device.type == "cuda"
    assert tiles.valid.device.type == "cuda"
    assert tiles.lo.device.type == "cuda"
    assert tiles.hi.device.type == "cuda"
    assert tiles.tile_cc.device.type == "cuda"

    a, b = tile_pairs(tiles, cutoff=5.0)
    assert a.device.type == "cuda"
    assert b.device.type == "cuda"


@pytest.mark.cuda
def test_tiles_empty_system_stays_on_input_device() -> None:
    """The degenerate, atom-free branch (`Tiles._build_empty`) must also
    place its tensors on `positions`' device."""
    positions = torch.zeros(0, 3, device="cuda")

    tiles = Tiles(positions, tile=8)
    assert tiles.n_axis.device.type == "cuda"
    assert tiles.strides.device.type == "cuda"

    empty_a, empty_b = tile_pairs(tiles, cutoff=5.0)
    assert empty_a.device.type == "cuda"
    assert empty_b.device.type == "cuda"
