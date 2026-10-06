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
Test the padded neighbour list: `NeighborList`, `build_neighborlist` and
`build_neighborlists`.
"""

from __future__ import annotations

import pytest
import torch

from tad_mctc.data.structures import get_structure
from tad_mctc.io.structure import Structure, pack_structures
from tad_mctc.ncoord import cn_d3
from tad_mctc.neighbor._tiles import Tiles, _is_forward
from tad_mctc.neighbor.images import (
    build_periodic_shifts,
    count_image_rings_mctclib,
    wrap_to_central_cell,
)
from tad_mctc.neighbor.list import (
    NeighborList,
    _ghost_pool,
    _ghost_shift_in_original_coordinates,
    _neighbor_list,
    _Pairs,
    _split_cells_by_pair_budget,
    _tiles_holding_primary,
    build_neighborlist,
    build_neighborlists,
)
from tad_mctc.typing import DD, Tensor

from ..conftest import DEVICE
from ..utils import hydrogens, load_structure


def dense_pair_set(positions: Tensor, cutoff: float) -> set[tuple[int, int]]:
    """All atom pairs ``i < j`` within ``cutoff``, from a dense distance
    matrix."""
    nat = positions.shape[0]
    distances = torch.cdist(positions, positions)

    index = torch.arange(nat, device=positions.device)
    row_index = index.unsqueeze(-1).expand(nat, nat)
    col_index = index.unsqueeze(0).expand(nat, nat)

    strict_upper_triangle = row_index < col_index
    within_cutoff = distances <= cutoff
    keep = strict_upper_triangle & within_cutoff

    rows, cols = keep.nonzero(as_tuple=True)
    return set(zip(rows.tolist(), cols.tolist()))


def neighborlist_pair_set(nbl) -> set[tuple[int, int]]:  # type: ignore[no-untyped-def]
    """The real pairs held by a `NeighborList`, as a set of ``i < j``
    tuples. Also checks that no pair appears twice."""
    idx_i = nbl.idx_i[nbl.mask].tolist()
    idx_j = nbl.idx_j[nbl.mask].tolist()

    pairs = set()
    for i, j in zip(idx_i, idx_j):
        lo, hi = (i, j) if i < j else (j, i)
        pairs.add((lo, hi))

    assert len(pairs) == len(idx_i), "a pair was listed more than once"
    return pairs


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
    """The padded list's real pairs must equal, as a set, the dense
    brute-force ones."""
    positions = get_structure(
        collection, record, device=DEVICE, dtype=dtype
    ).positions

    cutoff = 6.0
    nbl = build_neighborlist(hydrogens(positions), cutoff, tile=4)

    assert neighborlist_pair_set(nbl) == dense_pair_set(positions, cutoff)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_padding_targets_the_phantom_row(dtype: torch.dtype) -> None:
    """Padded slots must carry `idx_i == idx_j == nat` and `mask ==
    False`."""
    dd: DD = {"device": DEVICE, "dtype": dtype}

    torch.manual_seed(0)
    positions = torch.randn(30, 3, **dd) * 6.0
    nat = positions.shape[0]

    nbl = build_neighborlist(hydrogens(positions), cutoff=3.0, tile=8)
    padded = ~nbl.mask

    assert bool((nbl.idx_i[padded] == nat).all())
    assert bool((nbl.idx_j[padded] == nat).all())
    assert bool((nbl.idx_i[nbl.mask] != nat).all())
    assert bool((nbl.idx_j[nbl.mask] != nat).all())


def test_padding_atoms_of_a_single_structure_have_no_pairs() -> None:
    """Padding atoms (`numbers == 0`) of a single structure must get no
    pairs, as in a batch, even when they sit on top of real atoms: the
    sparse CN must then match the dense one."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    numbers = torch.tensor([8, 1, 1, 0, 0], device=DEVICE)
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.8, 0.0, 0.0],
            [-0.5, 1.7, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        **dd,
    )
    structure = Structure(numbers=numbers, positions=positions)

    nbl = build_neighborlist(structure, cn_d3.cutoff)

    assert neighborlist_pair_set(nbl) == {(0, 1), (0, 2), (1, 2)}
    assert torch.allclose(cn_d3(structure, nbl), cn_d3(structure))


def test_to_keeps_integer_and_boolean_slots() -> None:
    """`NeighborList.to()` must move every slot to the requested device
    and dtype, but never cast `idx_i`/`idx_j` away from `int32`, `shift`
    away from `int16`, or `mask` away from `bool` -- those are index data,
    not floating-point physics. `TensorLike.to()`, which `NeighborList`
    overrides, casts every slot to the requested floating dtype."""
    positions = torch.randn(10, 3, dtype=torch.float64)
    nbl = build_neighborlist(hydrogens(positions), cutoff=3.0, tile=4)

    moved = nbl.to(dtype=torch.float32)

    assert moved.idx_i.dtype == torch.int32
    assert moved.idx_j.dtype == torch.int32
    assert moved.shift.dtype == torch.int16
    assert moved.mask.dtype == torch.bool


def test_undersized_capacity_sets_overflow_without_raising() -> None:
    """An explicit `capacity` too small for the real pair count must set
    `overflow`, not raise."""
    torch.manual_seed(1)
    positions = torch.randn(150, 3) * 12.0

    full = build_neighborlist(hydrogens(positions), cutoff=6.0, tile=16)
    npair = int(full.mask.sum().item())
    assert npair > 8, "test needs a system with more than 8 real pairs"

    small_capacity = npair // 2
    truncated = build_neighborlist(
        hydrogens(positions), cutoff=6.0, tile=16, capacity=small_capacity
    )

    assert truncated.overflow is True
    assert truncated.idx_i.shape[0] == small_capacity
    assert int(truncated.mask.sum().item()) == small_capacity


def test_sufficient_capacity_does_not_overflow() -> None:
    """A `capacity` large enough for the real pair count must not set
    `overflow`."""
    torch.manual_seed(1)
    positions = torch.randn(150, 3) * 12.0

    full = build_neighborlist(hydrogens(positions), cutoff=6.0, tile=16)
    npair = int(full.mask.sum().item())

    generous = build_neighborlist(
        hydrogens(positions), cutoff=6.0, tile=16, capacity=npair + 64
    )
    assert generous.overflow is False


def test_stale_with_zero_skin_is_always_true() -> None:
    """With `skin == 0.0`, `stale` always returns `True`."""
    torch.manual_seed(2)
    positions = torch.randn(20, 3) * 5.0

    nbl = build_neighborlist(hydrogens(positions), cutoff=3.0, tile=8, skin=0.0)
    assert bool(nbl.stale(hydrogens(positions)))


def test_stale_reacts_to_drift_past_half_the_skin() -> None:
    """Drift past `skin / 2` must rebuild; drift under it must not."""
    torch.manual_seed(3)
    positions = torch.randn(20, 3) * 5.0

    nbl = build_neighborlist(hydrogens(positions), cutoff=3.0, tile=8, skin=2.0)

    tiny_drift = positions.clone()
    tiny_drift[0, 0] += 1e-6
    assert not bool(nbl.stale(hydrogens(tiny_drift)))

    large_drift = positions.clone()
    large_drift[0, 0] += 100.0
    assert bool(nbl.stale(hydrogens(large_drift)))


def test_stale_sees_positions_changed_in_place() -> None:
    """An MD loop typically updates positions in place. The list must
    compare against its own copy of the build-time positions, not against
    the caller's tensor, or it could never go stale."""
    torch.manual_seed(3)
    positions = torch.randn(20, 3) * 5.0

    structure = hydrogens(positions)
    nbl = build_neighborlist(structure, cutoff=3.0, tile=8, skin=2.0)

    structure.positions[0, 0] += 100.0
    assert bool(nbl.stale(structure))


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_batch_produces_no_cross_system_pairs(dtype: torch.dtype) -> None:
    """Two systems that overlap in space, packed into one batched
    `Structure`, must never produce a pair between them, and each
    system's own pairs must match its standalone dense reference."""
    dd: DD = {"device": DEVICE, "dtype": dtype}

    torch.manual_seed(4)
    n1, n2 = 40, 55
    positions_1 = torch.randn(n1, 3, **dd) * 6.0
    positions_2 = torch.randn(n2, 3, **dd) * 6.0  # deliberately overlapping

    batch = pack_structures([hydrogens(positions_1), hydrogens(positions_2)])
    nat = batch.numbers.shape[-1]  # atom `i` of system `b` is `b * nat + i`

    cutoff = 4.0
    nbl = build_neighborlist(batch, cutoff, tile=8)
    got = neighborlist_pair_set(nbl)

    for i, j in got:
        same_system = (i < nat) == (j < nat)
        assert same_system, "cross-system pair leaked into the list"

    want_1 = dense_pair_set(positions_1, cutoff)
    want_2 = dense_pair_set(positions_2, cutoff)

    got_1 = {(i, j) for i, j in got if j < nat}
    got_2 = {(i - nat, j - nat) for i, j in got if i >= nat}

    assert got_1 == want_1
    assert got_2 == want_2


def test_batch_coarsened_grid_produces_no_cross_system_pairs() -> None:
    """`Tiles.__init__` must recompute the bin width after its halving
    loop coarsens `n_axis`. With the pre-coarsening width, the last bin
    along an axis would silently absorb atoms that are many bins away.

    `_separate_systems` relies on `Tiles`' bounding-box screen to reject
    cross-system tile pairs by shifting systems apart in space. With a
    stale width, atoms from different, widely-separated systems would land
    in the same oversized terminal tile, and the exact distance filter
    would then approve the pair using each atom's real (unshifted,
    genuinely close) coordinates. `tile=32` here forces the halving loop
    (`max_cells = 4 * ceil(4 / 32) = 4`, but the pre-coarsening grid along
    the long axis is enormous)."""
    positions = torch.tensor(
        [[[0.0, 0, 0], [1.0, 0, 0]], [[0.2, 0, 0], [1.2, 0, 0]]],
        dtype=torch.double,
    )

    nbl = build_neighborlist(hydrogens(positions), 4.0, tile=32)
    got = neighborlist_pair_set(nbl)

    for i, j in got:
        same_system = (i < 2) == (j < 2)
        assert same_system, "cross-system pair leaked into the list"

    assert got == {(0, 1), (2, 3)}


def test_batch_many_single_atom_systems_produces_no_cross_system_pairs() -> (
    None
):
    """`Tiles.__init__` sizes its `max_cells` budget purely from total atom
    count and the caller's `tile`, oblivious to the artificial spatial
    separation `_separate_systems` introduces between batched systems.
    With `nat=16` and the default `tile=32`, `max_cells = 4 * ceil(16 /
    32) = 4`, forcing the coarsening loop to produce a bin far wider than
    the inter-system gap -- atoms from different systems then land in the
    same bin/tile, and the exact distance filter approves the pair using
    real (correctly close, but cross-system) coordinates.

    16 single-atom systems on a line, `cutoff=2.0`: no two atoms share a
    system, so the correct answer is exactly 0 pairs; a bin wider than the
    gap gives 26."""
    n_systems = 16
    cutoff = 2.0
    positions = torch.tensor(
        [[[float(s), 0.0, 0.0]] for s in range(n_systems)], dtype=torch.double
    )

    nbl = build_neighborlist(hydrogens(positions), cutoff)
    got = neighborlist_pair_set(nbl)

    assert got == set(), "no two atoms share a system: expected 0 pairs"


@pytest.mark.parametrize("n_systems", [10, 50, 200])
def test_batch_scale_sweep_produces_no_cross_system_pairs(
    n_systems: int,
) -> None:
    """Scale sweep for the same property as
    `test_batch_many_single_atom_systems_produces_no_cross_system_pairs`:
    a leak of cross-system pairs through a stale bin width would grow with
    batch size rather than show up only at small `nat`, so this sweeps
    `n_systems` with small, random per-system clusters at the default
    `tile=32` and checks every size, not just the smallest one.

    Cross-checked against a dense/brute-force reference restricted to
    each system's own atoms (masking by `batch[i] == batch[j]`, not just
    real distance): two atoms in *different* systems that happen to sit
    close together in raw coordinates must never appear together, no
    matter how close.

    At `n_systems=10` with this seed, this also pins the coarsening
    bound. `Tiles`' coarsening loop is constrained by `2 * cutoff` (see
    `_separate_systems`'s docstring for the derivation), not by the looser
    `separation` (`box_span + 2 * cutoff`). With `separation`, atom 2
    (system 0) and atom 4 (system 1) land in the same coarsened tile and
    are real-space neighbours (distance a little over 2, under
    `cutoff=3.0`), so this test fails; running this configuration with
    each bound confirms it. Do not "fix" a future failure here by widening
    the bound towards `separation`."""
    torch.manual_seed(0)
    atoms_per_system = 3
    cutoff = 3.0

    per_system_positions = [
        torch.randn(atoms_per_system, 3, dtype=torch.double) * 2.0
        for _ in range(n_systems)
    ]
    # Systems of equal size, so atom `i` of system `b` is `b * 3 + i`,
    # both in the batched list and in `batch` below.
    structure = hydrogens(torch.stack(per_system_positions))
    batch = torch.repeat_interleave(torch.arange(n_systems), atoms_per_system)

    nbl = build_neighborlist(structure, cutoff)
    got = neighborlist_pair_set(nbl)

    want = set()
    for s, system_positions in enumerate(per_system_positions):
        offset = s * atoms_per_system
        for i, j in dense_pair_set(system_positions, cutoff):
            want.add((i + offset, j + offset))

    for i, j in got:
        assert batch[i] == batch[j], (
            f"cross-system pair ({i}, {j}) leaked into the list "
            f"(batch {int(batch[i])} vs {int(batch[j])})"
        )

    assert got == want


def test_build_neighborlists_matches_separate_calls() -> None:
    """`build_neighborlists` sharing one traversal must be set-equal to
    two separate `build_neighborlist` calls, and the shorter-cutoff list
    must have a strictly smaller capacity."""
    torch.manual_seed(5)
    positions = torch.rand(3000, 3) * 40.0

    long_shared, short_shared = build_neighborlists(
        hydrogens(positions), (10.0, 4.0), tile=32
    )
    long_alone = build_neighborlist(hydrogens(positions), 10.0, tile=32)
    short_alone = build_neighborlist(hydrogens(positions), 4.0, tile=32)

    assert neighborlist_pair_set(long_shared) == neighborlist_pair_set(
        long_alone
    )
    assert neighborlist_pair_set(short_shared) == neighborlist_pair_set(
        short_alone
    )

    # The shared traversal fixes no pair ordering, so only set equality
    # is asserted above, never tensor equality.
    assert short_shared.idx_i.shape[0] < long_shared.idx_i.shape[0]


##############################################################################
# Periodic boundary conditions
##############################################################################


def test_one_atom_cell_folds_every_neighbour_to_three_entries() -> None:
    """A single atom in a periodic cell has, for a cutoff between the
    lattice constant and its face diagonal, exactly six image neighbours
    (one in each of ``+x``, ``-x``, ``+y``, ``-y``, ``+z``, ``-z``). All
    six are self-image pairs, so the list's folding of ``+shift``/``-shift``
    together (see `NeighborList`'s docstring) halves that count to three
    stored entries, not six."""
    sample = get_structure("other", "periodic_one_atom")
    assert sample.lattice is not None and sample.periodic is not None
    positions = sample.positions.to(dtype=torch.double)
    lattice = sample.lattice.to(dtype=torch.double)
    periodic = sample.periodic

    cutoff = (
        6.0  # between the 5 Bohr lattice constant and its face diagonal (~7.07)
    )
    nbl = build_neighborlist(
        hydrogens(positions, lattice=lattice, periodic=periodic), cutoff=cutoff
    )

    assert int(nbl.mask.sum().item()) == 3


@pytest.mark.parametrize("kernel", ["baddbmm", "broadcast"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.double])
def test_images_exactly_at_the_cutoff_are_kept_by_every_kernel(
    kernel: str, dtype: torch.dtype
) -> None:
    """The images of an atom at integer multiples of the lattice constant lie
    exactly at a cutoff that is such a multiple. Whether the squared distance
    rounds above or below the squared cutoff depends on the arithmetic of the
    kernel (the `baddbmm` expansion vs. the broadcast difference), so the
    list keeps them with a few ulps of slack, for all kernels alike."""
    edge = 4.0
    cutoff = 5 * edge
    positions = torch.tensor(
        [
            [3.8802120072262123, 2.831279457599152, 1.8375317725098035],
            [3.682990736487841, 2.580096480491059, 3.1645915687212147],
        ],
        dtype=dtype,
    )
    lattice = torch.eye(3, dtype=dtype) * edge
    periodic = torch.ones(3, dtype=torch.bool)

    nbl = build_neighborlist(
        hydrogens(positions, lattice=lattice, periodic=periodic),
        cutoff=cutoff,
        distance_kernel=kernel,
    )

    # Every atom with its own images: all lattice vectors n with |n| <= 5,
    # stored once per pair of +n and -n.
    r = range(-5, 6)
    n_images = sum(
        1 for a in r for b in r for c in r if 0 < a * a + b * b + c * c <= 25
    )
    own_images = nbl.mask & (nbl.idx_i == nbl.idx_j)
    assert int(own_images.sum().item()) == 2 * (n_images // 2)


def periodic_neighborlist_pair_set(
    nbl: NeighborList,
) -> set[tuple[int, int, int, int, int]]:
    """The real pairs held by a periodic `NeighborList`, as a set of
    `(idx_i, idx_j, shift_x, shift_y, shift_z)` tuples. Also checks that
    no pair appears twice.

    An entry may store a pair in either direction, `(i, j, +n)` or
    `(j, i, -n)`. Each tuple is written in the direction with `i < j`, or
    for a self-image pair with the first non-zero shift component
    positive, so that two lists compare equal whenever they describe the
    same pairs."""
    idx_i = nbl.idx_i[nbl.mask].tolist()
    idx_j = nbl.idx_j[nbl.mask].tolist()
    shift = nbl.shift[nbl.mask].tolist()

    pairs = set()
    for i, j, s in zip(idx_i, idx_j, shift):
        if i > j or (i == j and tuple(s) < (0, 0, 0)):
            i, j, s = j, i, [-component for component in s]
        pairs.add((i, j, s[0], s[1], s[2]))

    assert len(pairs) == len(idx_i), "a pair was listed more than once"
    return pairs


def periodic_pair_geometry(
    nbl: NeighborList, positions: Tensor, lattice: Tensor
) -> set[tuple[int, int, float]]:
    """The *physics* a periodic `NeighborList` describes: every real pair
    as `(idx_i, idx_j, distance)`, with the distance reconstructed the
    way a consumer does it (``positions[j] - positions[i] + shift @
    lattice``, see ``ncoord/common.py``).

    Unlike `periodic_neighborlist_pair_set`, this is invariant under
    which periodic image the list chose to anchor a pair at, so it can
    compare two lists built from coordinates that differ by whole
    lattice vectors. The two atoms are sorted, since an entry may store
    a pair in either direction.
    """
    idx_i = nbl.idx_i[nbl.mask]
    idx_j = nbl.idx_j[nbl.mask]
    shift = nbl.shift[nbl.mask]

    difference = positions[idx_j] - positions[idx_i]
    difference = difference + shift.to(positions.dtype) @ lattice
    distance = difference.norm(dim=-1)

    return {
        (min(i, j), max(i, j), round(d, 8))
        for i, j, d in zip(idx_i.tolist(), idx_j.tolist(), distance.tolist())
    }


def _cn_via_explicit_supercell(
    numbers: Tensor,
    positions: Tensor,
    lattice: Tensor,
    cutoff: float,
    margin: int,
) -> Tensor:
    """Coordination number of every atom in `positions`, from an explicit,
    non-periodic supercell enumeration rather than the periodic
    `NeighborList` under test: replicate `margin` rings of images by
    hand, run the already-tested *molecular* sparse path (`nbl` with no
    `lattice`) on the whole supercell at the same cutoff, and read off the
    center cell's atoms. `margin` must be generous enough that `cutoff`
    never reaches the supercell's own edge starting from the center cell,
    so this is independent of anything `count_image_rings` decided."""
    nat = numbers.shape[0]
    axis_range = torch.arange(-margin, margin + 1, device=positions.device)
    shifts = torch.cartesian_prod(axis_range, axis_range, axis_range)
    translations = shifts.to(positions.dtype) @ lattice

    super_positions = (
        positions.unsqueeze(0) + translations.unsqueeze(1)
    ).reshape(-1, 3)
    super_numbers = numbers.repeat(translations.shape[0])

    # The test cutoff is smaller than `cn_d3`'s own default. The model
    # masks pairs at its own cutoff and rejects a list built at a smaller
    # one, so it is set to the cutoff this comparison runs at.
    model = cn_d3.replace(cutoff=cutoff)
    nbl_super = build_neighborlist(
        hydrogens(super_positions), cutoff, capacity=None
    )
    supercell = Structure(numbers=super_numbers, positions=super_positions)
    cn_super = model(supercell, pairs=nbl_super)

    center = int((shifts == 0).all(-1).nonzero(as_tuple=True)[0])
    return cn_super[center * nat : (center + 1) * nat]


_CUBIC_CELL = 8.0 * torch.eye(3, dtype=torch.double)
_TRICLINIC_CELL = torch.tensor(
    [[7.0, 0.0, 0.0], [1.2, 6.5, 0.0], [0.6, 0.9, 6.0]], dtype=torch.double
)


@pytest.mark.parametrize(
    "lattice, cutoff, margin",
    [
        (_CUBIC_CELL, 9.0, 2),  # cutoff larger than the box edge
        (_TRICLINIC_CELL, 8.0, 2),
    ],
)
def test_periodic_matches_explicit_supercell_enumeration(
    lattice: Tensor, cutoff: float, margin: int
) -> None:
    """A periodic `NeighborList`, consumed by the already-tested sparse
    `cn_d3`, must match an independent, non-periodic supercell
    enumeration to floating-point noise."""
    torch.manual_seed(6)
    nat = 6
    lattice = lattice.to(DEVICE)
    periodic = torch.tensor([True, True, True], device=DEVICE)

    fractional = torch.rand(nat, 3, dtype=torch.double, device=DEVICE)
    positions = fractional @ lattice  # every atom stays inside the cell
    numbers = torch.randint(1, 30, (nat,), device=DEVICE)

    model = cn_d3.replace(cutoff=cutoff)
    nbl = build_neighborlist(
        hydrogens(positions, lattice=lattice, periodic=periodic),
        cutoff,
        capacity=None,
    )
    structure = Structure(numbers=numbers, positions=positions, lattice=lattice)
    got = model(structure, pairs=nbl)
    want = _cn_via_explicit_supercell(
        numbers, positions, lattice, cutoff, margin
    )

    assert torch.allclose(got, want, atol=1e-11, rtol=1e-10)


def test_periodic_pairs_survive_coordinates_outside_the_cell() -> None:
    """The same physical system written with an atom several cells over
    must give the same pairs at the same distances.

    The ghost pool is sized from `lattice` and `cutoff` alone, so without
    wrapping it only reaches `rings` cells out and a pair needing a
    larger image simply disappears -- silently, which is the ordinary
    case for an unwrapped MD trajectory.
    """
    edge = 10.0
    lattice = edge * torch.eye(3, dtype=torch.double, device=DEVICE)
    periodic = torch.tensor([True, True, True], device=DEVICE)

    inside = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.double, device=DEVICE
    )
    outside = inside.clone()
    outside[1, 0] += 2.0 * edge  # same bond, written two cells over

    def build(positions: Tensor) -> NeighborList:
        return build_neighborlist(
            hydrogens(positions, lattice=lattice, periodic=periodic), cutoff=3.0
        )

    got = periodic_pair_geometry(build(outside), outside, lattice)
    want = periodic_pair_geometry(build(inside), inside, lattice)

    assert want == {(0, 1, 1.0)}, "the in-cell reference itself is wrong"
    assert got == want


def test_periodic_pairs_survive_a_translation_of_thousands_of_cells() -> None:
    """A structure written 20000 cells away folds by more than the
    ``int16`` shifts of `_ghost_shift_in_original_coordinates` can hold,
    so the search keeps them in ``int64``; the pairs must not change."""
    edge = 10.0
    lattice = edge * torch.eye(3, dtype=torch.double, device=DEVICE)
    periodic = torch.tensor([True, True, True], device=DEVICE)

    inside = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
        dtype=torch.double,
        device=DEVICE,
    )
    far_away = inside.clone()
    far_away[:, 0] += 20000 * edge

    def build(positions: Tensor) -> NeighborList:
        return build_neighborlist(
            hydrogens(positions, lattice=lattice, periodic=periodic), cutoff=3.0
        )

    got = periodic_pair_geometry(build(far_away), far_away, lattice)
    want = periodic_pair_geometry(build(inside), inside, lattice)

    assert len(want) > 0, "the in-cell reference found no pairs at all"
    assert got == want


def test_periodic_pairs_survive_atoms_written_hundreds_of_cells_apart() -> None:
    """In an unwrapped MD trajectory the two atoms of a pair can drift
    hundreds of cells apart; their shift then exceeds ``int8`` (127
    cells) and must still be stored exactly."""
    edge = 10.0
    lattice = edge * torch.eye(3, dtype=torch.double, device=DEVICE)
    periodic = torch.tensor([True, True, True], device=DEVICE)

    inside = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
        dtype=torch.double,
        device=DEVICE,
    )
    apart = inside.clone()
    apart[1, 0] += 200 * edge  # one atom written 200 cells over
    apart[2, 1] -= 300 * edge  # another 300 cells the other way

    def build(positions: Tensor) -> NeighborList:
        return build_neighborlist(
            hydrogens(positions, lattice=lattice, periodic=periodic), cutoff=3.0
        )

    nbl = build(apart)
    got = periodic_pair_geometry(nbl, apart, lattice)
    want = periodic_pair_geometry(build(inside), inside, lattice)

    assert len(want) > 0, "the in-cell reference found no pairs at all"
    assert got == want
    assert int(nbl.shift[nbl.mask].abs().max()) == 300


def test_periodic_shift_beyond_its_dtype_names_unwrapped_positions() -> None:
    """A pair whose atoms are written farther apart than the stored shift
    can count must raise, and the message must point at the unwrapped
    positions, not only at the cutoff."""
    edge = 10.0
    lattice = edge * torch.eye(3, dtype=torch.double, device=DEVICE)
    periodic = torch.tensor([True, True, True], device=DEVICE)

    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.double, device=DEVICE
    )
    positions[1, 0] += 40000 * edge

    with pytest.raises(ValueError, match="wrap `positions`"):
        build_neighborlist(
            hydrogens(positions, lattice=lattice, periodic=periodic), cutoff=3.0
        )


def test_shift_range_is_checked_only_for_the_pairs_kept() -> None:
    """A pair dropped by a too-small `capacity` is not stored, so a shift
    out of `NeighborList.shift`'s range on it must not raise and hide
    `overflow`."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    kept_pair = torch.tensor([0], dtype=torch.int32, device=DEVICE)
    pairs = _Pairs(idx_i=kept_pair, idx_j=kept_pair + 1, n_found=2)
    shift_raw = torch.tensor([[0, 0, 0], [40000, 0, 0]], device=DEVICE)

    nbl = _neighbor_list(
        pairs,
        cutoff=3.0,
        skin=0.0,
        build_positions=torch.zeros(2, 3, **dd),
        dd=dd,
        shift_raw=shift_raw,
        lattice=10.0 * torch.eye(3, **dd),
        periodic_axes=torch.tensor([True, True, True], device=DEVICE),
    )

    assert nbl.overflow is True
    assert nbl.shift.tolist() == [[0, 0, 0]]


def test_shift_at_both_ends_of_its_dtype_is_stored() -> None:
    """``int16`` holds -32768 but not +32768, so the range check must
    compare each bound on its own rather than the magnitude."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    idx = torch.tensor([0, 0], dtype=torch.int32, device=DEVICE)
    pairs = _Pairs(idx_i=idx, idx_j=idx + 1, n_found=2)
    shift_raw = torch.tensor([[-32768, 0, 0], [0, 32767, 0]], device=DEVICE)

    nbl = _neighbor_list(
        pairs,
        cutoff=3.0,
        skin=0.0,
        build_positions=torch.zeros(2, 3, **dd),
        dd=dd,
        shift_raw=shift_raw,
        lattice=10.0 * torch.eye(3, **dd),
        periodic_axes=torch.tensor([True, True, True], device=DEVICE),
    )

    assert nbl.shift.tolist() == [[-32768, 0, 0], [0, 32767, 0]]


def test_shift_just_beyond_either_end_of_its_dtype_raises() -> None:
    """One cell past ``int16``'s range, on either side, must raise
    instead of wrapping around."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    idx = torch.tensor([0], dtype=torch.int32, device=DEVICE)
    pairs = _Pairs(idx_i=idx, idx_j=idx + 1, n_found=1)

    for shift_raw in (
        torch.tensor([[-32769, 0, 0]], device=DEVICE),
        torch.tensor([[0, 0, 32768]], device=DEVICE),
    ):
        with pytest.raises(ValueError, match="lies outside"):
            _neighbor_list(
                pairs,
                cutoff=3.0,
                skin=0.0,
                build_positions=torch.zeros(2, 3, **dd),
                dd=dd,
                shift_raw=shift_raw,
                lattice=10.0 * torch.eye(3, **dd),
                periodic_axes=torch.tensor([True, True, True], device=DEVICE),
            )


def test_ghost_shifts_are_int16_only_when_their_differences_fit() -> None:
    """A ghost's shift is its image shift plus its atom's fold, narrowed
    to ``int16`` only while the difference of two cannot overflow."""
    owner = torch.tensor([0, 1, 1], device=DEVICE)
    ghost_shift = torch.tensor([[0, 0, 0], [0, 0, 0], [1, 0, 0]], device=DEVICE)

    near_fold = torch.tensor([[2, 0, 0], [-3, 0, 0]], device=DEVICE)
    near = _ghost_shift_in_original_coordinates(ghost_shift, owner, near_fold)
    assert near.dtype == torch.int16
    assert near.tolist() == [[2, 0, 0], [-3, 0, 0], [-2, 0, 0]]

    far_fold = torch.tensor([[20000, 0, 0], [20000, 0, 0]], device=DEVICE)
    far = _ghost_shift_in_original_coordinates(ghost_shift, owner, far_fold)
    assert far.dtype == torch.int64
    assert far.tolist() == [[20000, 0, 0], [20000, 0, 0], [20001, 0, 0]]


@pytest.mark.parametrize(
    "lattice, periodic_axes",
    [
        (_CUBIC_CELL, [True, True, True]),
        (_TRICLINIC_CELL, [True, True, True]),
        (_CUBIC_CELL, [True, True, False]),
    ],
)
def test_periodic_pairs_are_invariant_under_whole_cell_translations(
    lattice: Tensor, periodic_axes: list[bool]
) -> None:
    """Translating individual atoms by whole lattice vectors along the
    periodic axes describes the *same* crystal, so it must give the same
    pairs at the same distances -- for a skewed cell and for a slab, not
    just a cube, and with offsets big enough that the un-wrapped ghost
    pool could never have reached them."""
    torch.manual_seed(11)
    nat = 5
    lattice = lattice.to(DEVICE)
    periodic = torch.tensor(periodic_axes, device=DEVICE)

    fractional = torch.rand(nat, 3, dtype=torch.double, device=DEVICE)
    inside = fractional @ lattice

    offsets = torch.randint(-3, 4, (nat, 3), device=DEVICE)
    offsets = torch.where(periodic, offsets, torch.zeros_like(offsets))
    outside = inside + offsets.to(torch.double) @ lattice

    def build(positions: Tensor) -> NeighborList:
        return build_neighborlist(
            hydrogens(positions, lattice=lattice, periodic=periodic),
            cutoff=5.0,
            tile=4,
        )

    want = periodic_pair_geometry(build(inside), inside, lattice)
    got = periodic_pair_geometry(build(outside), outside, lattice)

    assert len(want) > 0, "the in-cell reference found no pairs at all"
    assert got == want


def test_periodic_cn_is_invariant_under_whole_cell_translations() -> None:
    """The same invariance one level up, through the physics: `cn_d3`
    must not notice which periodic image an atom was written in.

    The cutoff deliberately avoids ``_TRICLINIC_CELL``'s own vector
    lengths (7.0, 6.61, 6.11). A self-image sitting at *exactly* the
    cutoff is a knife-edge that no floating-point implementation can
    resolve translation-invariantly -- the pair's distance is
    reconstructed from coordinates of different magnitude in the two
    builds -- and pinning that would be testing rounding, not the
    wrapping this covers.
    """
    torch.manual_seed(12)
    nat = 6
    cutoff = 5.5
    lattice = _TRICLINIC_CELL.to(DEVICE)
    periodic = torch.tensor([True, True, True], device=DEVICE)

    numbers = torch.randint(1, 30, (nat,), device=DEVICE)
    inside = torch.rand(nat, 3, dtype=torch.double, device=DEVICE) @ lattice
    offsets = torch.randint(-2, 3, (nat, 3), device=DEVICE)
    outside = inside + offsets.to(torch.double) @ lattice

    model = cn_d3.replace(cutoff=cutoff)

    def coordination(positions: Tensor) -> Tensor:
        nbl = build_neighborlist(
            hydrogens(positions, lattice=lattice, periodic=periodic), cutoff
        )
        structure = Structure(
            numbers=numbers, positions=positions, lattice=lattice
        )
        return model(structure, pairs=nbl)

    assert torch.allclose(coordination(outside), coordination(inside))


def test_periodic_padding_targets_the_phantom_row() -> None:
    """Padded slots in a periodic list must still carry `idx_i == idx_j
    == nat` and `mask == False`."""
    torch.manual_seed(7)
    nat = 10
    lattice = _CUBIC_CELL.to(DEVICE)
    periodic = torch.tensor([True, True, True], device=DEVICE)
    positions = torch.rand(nat, 3, dtype=torch.double, device=DEVICE) @ lattice

    nbl = build_neighborlist(
        hydrogens(positions, lattice=lattice, periodic=periodic),
        cutoff=6.0,
        tile=4,
    )
    padded = ~nbl.mask

    assert bool(padded.any()), "test needs a list with padded slots"
    assert bool((nbl.idx_i[padded] == nat).all())
    assert bool((nbl.idx_j[padded] == nat).all())
    assert bool((nbl.shift[padded] == 0).all())
    assert bool((nbl.idx_i[nbl.mask] != nat).all())
    assert bool((nbl.idx_j[nbl.mask] != nat).all())


def test_periodic_explicit_capacity_keeps_the_leading_entries() -> None:
    """A periodic list with an explicit `capacity` holds the same leading
    entries as the auto-sized one: all of them, then padding, for a larger
    capacity, and the first `capacity` of them, flagged as `overflow`, for
    a smaller one."""
    torch.manual_seed(7)
    nat = 10
    lattice = _CUBIC_CELL.to(DEVICE)
    periodic = torch.tensor([True, True, True], device=DEVICE)
    positions = torch.rand(nat, 3, dtype=torch.double, device=DEVICE) @ lattice
    structure = hydrogens(positions, lattice=lattice, periodic=periodic)

    full = build_neighborlist(structure, cutoff=6.0, tile=4)
    npair = int(full.mask.sum())

    for capacity in (npair + 37, npair // 2):
        nbl = build_neighborlist(
            structure, cutoff=6.0, tile=4, capacity=capacity
        )
        kept = min(npair, capacity)

        assert nbl.idx_i.shape[0] == capacity
        assert nbl.shift.shape == (capacity, 3)
        assert nbl.overflow is (capacity < npair)
        assert int(nbl.mask.sum()) == kept
        assert torch.equal(nbl.idx_i[:kept], full.idx_i[:kept])
        assert torch.equal(nbl.idx_j[:kept], full.idx_j[:kept])
        assert torch.equal(nbl.shift[:kept], full.shift[:kept])
        assert bool((nbl.idx_i[kept:] == nat).all())
        assert bool((nbl.shift[kept:] == 0).all())


def test_padding_atoms_of_a_single_cell_have_no_pairs() -> None:
    """Padding atoms (`numbers == 0`) of a single periodic structure must
    get no pairs, not even with images of real atoms: the sparse CN must
    then match the dense one."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    numbers = torch.tensor([8, 1, 1, 0, 0], device=DEVICE)
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.8, 0.0, 0.0],
            [-0.5, 1.7, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        **dd,
    )
    lattice = 8.0 * torch.eye(3, **dd)
    structure = Structure(numbers=numbers, positions=positions, lattice=lattice)

    nbl = build_neighborlist(structure, cn_d3.cutoff)
    real_pairs = nbl.idx_i[nbl.mask], nbl.idx_j[nbl.mask]

    assert bool((real_pairs[0] < 3).all())
    assert bool((real_pairs[1] < 3).all())
    assert torch.allclose(cn_d3(structure, nbl), cn_d3(structure))


def test_periodic_stale_sees_an_atom_rewrapped_into_the_cell() -> None:
    """An atom that an MD code re-wraps into the cell has moved by a whole
    lattice vector in the caller's coordinates. The stored shifts are
    relative to those coordinates, so the list must read as stale: kept,
    its pair across the boundary would be measured a box length apart."""
    edge = 10.0
    lattice = edge * torch.eye(3, dtype=torch.double, device=DEVICE)
    periodic = torch.tensor([True, True, True], device=DEVICE)

    # 0.4 apart across x = 0
    positions = torch.tensor(
        [[0.2, 5.0, 5.0], [edge - 0.2, 5.0, 5.0]],
        dtype=torch.double,
        device=DEVICE,
    )
    nbl = build_neighborlist(
        hydrogens(positions, lattice=lattice, periodic=periodic),
        cutoff=3.0,
        skin=1.0,
    )

    rewrapped = positions.clone()
    rewrapped[0, 0] = edge - 0.1  # physically moved by -0.3, across x = 0
    assert bool(nbl.stale(hydrogens(rewrapped, lattice, periodic)))

    drifted = positions.clone()
    drifted[0, 0] -= 0.3  # the same physical move, left unwrapped
    assert not bool(nbl.stale(hydrogens(drifted, lattice, periodic)))


def test_periodic_large_lattice_reproduces_molecular_pairs() -> None:
    """A lattice much larger than the cutoff must reproduce the
    non-periodic pair set exactly, pair for pair, every shift zero."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    positions = get_structure("mb16_43", "01", **dd).positions

    cutoff = 6.0
    lattice = 500.0 * torch.eye(3, **dd)
    periodic = torch.tensor([True, True, True], device=DEVICE)

    periodic_nbl = build_neighborlist(
        hydrogens(positions, lattice=lattice, periodic=periodic), cutoff, tile=4
    )
    molecular_nbl = build_neighborlist(hydrogens(positions), cutoff, tile=4)

    got = periodic_neighborlist_pair_set(periodic_nbl)
    want = {(i, j, 0, 0, 0) for i, j in neighborlist_pair_set(molecular_nbl)}

    assert got == want


def test_neighborlist_has_no_public_lattice_attribute() -> None:
    """`NeighborList` carries no gradient, including no captured lattice:
    consumers receive the lattice as a call-time argument instead (see
    `tad_mctc.ncoord.common`), never by reading it off the list."""
    lattice = _CUBIC_CELL
    periodic = torch.tensor([True, True, True])
    positions = torch.rand(5, 3, dtype=torch.double) @ lattice

    nbl = build_neighborlist(
        hydrogens(positions, lattice=lattice, periodic=periodic),
        cutoff=4.0,
        tile=4,
    )
    assert not hasattr(nbl, "lattice")


def test_stale_reacts_to_a_strained_cell() -> None:
    """A strained cell changes every pair distance even when every atom
    stays put in fractional coordinates, so `stale` must flag it; an
    unchanged cell with unchanged positions must not."""
    lattice = _CUBIC_CELL
    periodic = torch.tensor([True, True, True])
    fractional = torch.rand(6, 3, dtype=torch.double)
    positions = fractional @ lattice

    nbl = build_neighborlist(
        hydrogens(positions, lattice=lattice, periodic=periodic),
        cutoff=4.0,
        tile=4,
        skin=1.0,
    )

    assert not bool(nbl.stale(hydrogens(positions, lattice, periodic)))

    strain = 1.05
    strained_lattice = strain * lattice
    strained_positions = fractional @ strained_lattice
    strained = hydrogens(strained_positions, strained_lattice, periodic)
    assert bool(nbl.stale(strained))


def test_stale_sees_a_cell_strained_in_place() -> None:
    """A cell optimiser may update the lattice in place. The list must
    compare against its own copy of the build-time lattice, not against
    the caller's tensor."""
    lattice = _CUBIC_CELL.clone()
    periodic = torch.tensor([True, True, True])
    positions = torch.rand(6, 3, dtype=torch.double) @ lattice

    structure = hydrogens(positions, lattice=lattice, periodic=periodic)
    nbl = build_neighborlist(structure, cutoff=4.0, tile=4, skin=1.0)

    lattice *= 1.05
    assert bool(nbl.stale(structure))


def test_stale_without_lattice_raises_for_a_periodic_list() -> None:
    """A periodic list's staleness cannot be decided without the current
    cell, so a `Structure` without a lattice must raise rather than
    silently assume the build-time cell still applies."""
    lattice = _CUBIC_CELL
    periodic = torch.tensor([True, True, True])
    positions = torch.rand(5, 3, dtype=torch.double) @ lattice

    nbl = build_neighborlist(
        hydrogens(positions, lattice=lattice, periodic=periodic),
        cutoff=4.0,
        tile=4,
        skin=1.0,
    )

    with pytest.raises(ValueError, match="periodic"):
        nbl.stale(hydrogens(positions))


def test_negative_cutoff_raises() -> None:
    """There is no physically meaningful negative real-space cutoff --
    `cutoff < 0` must raise immediately rather than silently produce a
    tile-layout-dependent, inconsistent result.
    """
    positions = torch.rand(10, 3, dtype=torch.double)

    with pytest.raises(ValueError):
        build_neighborlist(hydrogens(positions), cutoff=-1.0)


def test_negative_cutoff_in_build_neighborlists_raises() -> None:
    """Same as `test_negative_cutoff_raises`, but for the multi-cutoff
    entry point and naming the offending cutoff among several valid
    ones."""
    positions = torch.rand(10, 3, dtype=torch.double)

    with pytest.raises(ValueError):
        build_neighborlists(hydrogens(positions), (2.0, -1.0))


def test_negative_skin_raises() -> None:
    """`skin < 0` is just as physically meaningless as a negative cutoff
    and must raise immediately as well."""
    positions = torch.rand(10, 3, dtype=torch.double)

    with pytest.raises(ValueError):
        build_neighborlist(hydrogens(positions), cutoff=1.0, skin=-1.0)


def _batched_build_capturing_tiles(
    monkeypatch: pytest.MonkeyPatch,
    structure: Structure,
    cutoff: float,
) -> tuple[NeighborList, int]:
    """Run a batched `build_neighborlist` and report both the list and
    the number of bins the search grid it built actually had.

    The bin count is what the cost of a batched build scales with, and
    the public API exposes no handle on it, so it is captured from the
    `Tiles` the build constructs rather than recomputed here from the
    same formula the implementation uses.
    """
    from tad_mctc.neighbor import list as list_module

    captured: list[int] = []
    original = list_module.Tiles

    def recording_tiles(*args, **kwargs):  # type: ignore[no-untyped-def]
        tiles = original(*args, **kwargs)
        captured.append(tiles.ncells)
        return tiles

    monkeypatch.setattr(list_module, "Tiles", recording_tiles)
    nbl = build_neighborlist(structure, cutoff)

    assert len(captured) == 1, "expected exactly one Tiles build"
    return nbl, captured[0]


@pytest.mark.parametrize("outlier", [0.0, 1.0e3, 1.0e6])
def test_batch_grid_size_is_independent_of_a_far_outlier(
    monkeypatch: pytest.MonkeyPatch, outlier: float
) -> None:
    """A batched build must not let one distant atom blow up the search
    grid. The inter-system gap the separation strategy guarantees is
    proportional to the search radius, so when the radius is small next
    to the batch's own spread, the bin grid forced to stay below that
    gap grows without bound -- linearly in how far the outlier sits.

    The outlier here is displaced along axis 1, which the separation
    strategy never shifts along, so it has no business affecting the
    grid at all.
    """
    torch.manual_seed(0)
    n_systems, atoms_per_system = 50, 3
    nat = n_systems * atoms_per_system

    positions = torch.randn(nat, 3, dtype=torch.double)
    positions[0, 1] = outlier
    batch = torch.arange(n_systems).repeat_interleave(atoms_per_system)

    # Systems of equal size, so atom `i` of system `b` is
    # `b * atoms_per_system + i`, both in the batched list and in `batch`.
    structure = hydrogens(positions.reshape(n_systems, atoms_per_system, 3))
    nbl, ncells = _batched_build_capturing_tiles(
        monkeypatch, structure, cutoff=4.0
    )

    # Generous: one bin per atom plus a couple per system is already far
    # more than a sane grid needs. A grid bounded only by the inter-system
    # gap reaches ~6.1e6 here.
    assert ncells <= 4 * (nat + n_systems)

    # And the pairs must still be right -- a cheaper grid is worthless if
    # it costs correctness.
    distances = torch.cdist(positions, positions)
    want = {
        (i, j)
        for i in range(nat)
        for j in range(i + 1, nat)
        if batch[i] == batch[j] and distances[i, j] <= 4.0
    }
    assert neighborlist_pair_set(nbl) == want


def test_float32_batch_keeps_pairs_of_systems_shifted_far_out() -> None:
    """The systems of a batch are shifted apart for the tile screen, by
    about 2e5 Bohr each here, so the last ones sit near 4e7 Bohr, where
    float32 coordinates are 4 Bohr apart. Each pair's ends round apart
    past the cutoff there unless the shift is taken in float64."""
    n_systems = 200
    positions = torch.zeros(n_systems, 2, 3, dtype=torch.float32, device=DEVICE)
    positions[:, 0, 0] = 1.9
    positions[:, 1, 0] = 1.9 + 4.9
    positions[:, :, 1] = 0.3
    # One wide system sets the shift between neighbouring systems.
    positions[0, 1, 0] = 1e5
    structure = Structure(
        numbers=torch.ones(n_systems, 2, dtype=torch.long, device=DEVICE),
        positions=positions,
    )

    nbl = build_neighborlist(structure, 5.0, tile=1)

    # Every system but the wide one holds one pair.
    assert int(nbl.mask.sum()) == n_systems - 1


def test_batch_with_zero_search_radius_raises() -> None:
    """A batched search keeps systems apart by shifting them along one
    axis and relying on a guaranteed gap proportional to the search
    radius. At `cutoff == 0` with `skin == 0` there is no such gap, and
    the search silently pairs coincident atoms belonging to *different*
    systems -- so the degenerate request is rejected instead.
    """
    two_single_atom_systems = hydrogens(
        torch.zeros(2, 1, 3, dtype=torch.double)
    )

    with pytest.raises(ValueError, match="positive search radius"):
        build_neighborlist(two_single_atom_systems, 0.0)


def test_zero_cutoff_without_batch_still_pairs_coincident_atoms() -> None:
    """The rejection above is about *batching*, not about `cutoff == 0`
    itself: a single system at zero cutoff still correctly reports the
    pairs that sit at exactly distance zero."""
    positions = torch.zeros(2, 3, dtype=torch.double)

    nbl = build_neighborlist(hydrogens(positions), cutoff=0.0)

    assert neighborlist_pair_set(nbl) == {(0, 1)}


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_planar_half_precision_structure_matches_dense_pairs(
    dtype: torch.dtype,
) -> None:
    """A planar structure has a flat axis, whose bin width is a tiny
    floor. Dividing the cutoff by it must not overflow a half-precision
    dtype."""
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.8, 0.0, 0.0], [-0.5, 1.7, 0.0], [6.0, 6.0, 0.0]],
        dtype=dtype,
        device=DEVICE,
    )

    nbl = build_neighborlist(hydrogens(positions), cutoff=4.0)

    assert neighborlist_pair_set(nbl) == {(0, 1), (0, 2), (1, 2)}


@pytest.mark.parametrize("tile", [0, -1])
def test_non_positive_tile_raises(tile: int) -> None:
    """`tile` is a tile's atom capacity, so it must be at least one.
    Without the check it fails deep inside the build with an error that
    names internals (`ZeroDivisionError` from the `nat / tile` cell
    budget, or a negative tensor dimension) rather than the caller's
    mistake.
    """
    positions = torch.rand(10, 3, dtype=torch.double)

    with pytest.raises(ValueError):
        build_neighborlist(hydrogens(positions), cutoff=1.0, tile=tile)


@pytest.mark.cuda
def test_build_neighborlist_stays_on_input_device() -> None:
    """`build_neighborlist` (molecular and periodic) must return a list
    whose tensors all live on `positions`' device, without relying on the
    ambient default device -- this test never touches it."""
    torch.manual_seed(8)
    positions = torch.randn(20, 3, dtype=torch.double, device="cuda") * 6.0

    molecular = build_neighborlist(hydrogens(positions), cutoff=4.0, tile=4)
    assert molecular.idx_i.device.type == "cuda"
    assert molecular.idx_j.device.type == "cuda"
    assert molecular.shift.device.type == "cuda"
    assert molecular.mask.device.type == "cuda"

    lattice = 8.0 * torch.eye(3, dtype=torch.double, device="cuda")
    periodic = torch.tensor([True, True, True], device="cuda")
    periodic_nbl = build_neighborlist(
        hydrogens(positions, lattice=lattice, periodic=periodic),
        cutoff=4.0,
        tile=4,
    )
    assert periodic_nbl.idx_i.device.type == "cuda"
    assert periodic_nbl.shift.device.type == "cuda"


@pytest.mark.cuda
def test_build_neighborlists_shared_cutoffs_stay_on_input_device() -> None:
    """The multi-cutoff entry point shares the same construction path, so
    it needs the same device check."""
    torch.manual_seed(9)
    positions = torch.randn(20, 3, dtype=torch.double, device="cuda") * 6.0

    small, large = build_neighborlists(hydrogens(positions), (3.0, 6.0), tile=4)
    assert small.idx_i.device.type == "cuda"
    assert large.idx_i.device.type == "cuda"


########################################################################
# The forward half of the ghost pool, and pruning its tile pairs


def test_exactly_one_of_a_shift_and_its_negative_is_forward() -> None:
    """Every non-zero shift has exactly one forward direction, and the zero
    shift (the primary cell) counts as forward."""
    axis = torch.arange(-2, 3, device=DEVICE)
    shifts = torch.cartesian_prod(axis, axis, axis)
    is_zero = (shifts == 0).all(-1)

    forward = _is_forward(shifts)
    backward = _is_forward(-shifts)

    assert bool(forward[is_zero].all())
    assert bool((forward ^ backward)[~is_zero].all())


def test_periodic_list_stores_each_bond_once() -> None:
    """A bond from `i` to image `n` of `j` is the same bond as the one from
    `j` to image `-n` of `i`, so the list must hold only one of the two.
    For a self-image bond (`i == j`) that means only one of `+n`, `-n`."""
    torch.manual_seed(8)
    lattice = _TRICLINIC_CELL.to(DEVICE)
    periodic = torch.tensor([True, True, True], device=DEVICE)
    positions = torch.rand(6, 3, dtype=torch.double, device=DEVICE) @ lattice

    nbl = build_neighborlist(
        hydrogens(positions, lattice=lattice, periodic=periodic), cutoff=9.0
    )

    seen = set()
    idx_i = nbl.idx_i[nbl.mask].tolist()
    idx_j = nbl.idx_j[nbl.mask].tolist()
    shift = nbl.shift[nbl.mask].tolist()
    for i, j, (x, y, z) in zip(idx_i, idx_j, shift):
        bond = (i, j, x, y, z)
        mirror = (j, i, -x, -y, -z)
        assert bond not in seen and mirror not in seen
        seen.add(bond)
    assert any(i == j for i, j in zip(idx_i, idx_j)), "no self-image bond"


def test_tiles_holding_primary_finds_the_tiles_with_a_primary_atom() -> None:
    torch.manual_seed(4)
    positions = torch.rand(200, 3, device=DEVICE) * 30.0
    is_primary = torch.rand(200, device=DEVICE) < 0.05

    tiles = Tiles(positions, tile=8)
    holds = _tiles_holding_primary(tiles, is_primary)

    for t in range(tiles.ntile):
        atoms = tiles.index[t][tiles.valid[t]]
        assert bool(holds[t]) == bool(is_primary[atoms].any())
    assert 0 < int(holds.sum()) < tiles.ntile


########################################################################
# Batched structures: one list over the padded, flattened atoms


def _offset_pairs(
    pairs: set[tuple[int, ...]], offset: int
) -> set[tuple[int, ...]]:
    """`pairs` with both atom indices moved by `offset`, as for system
    `b` of a batch padded to `nat` atoms (`offset = b * nat`)."""
    return {(i + offset, j + offset, *rest) for i, j, *rest in pairs}


def test_batched_molecules_index_the_padded_flattened_atoms() -> None:
    """Atom `i` of system `b` is `b * nat + i`, and a padding atom has no
    pairs: the batched list holds exactly each system's own list, moved
    by its offset."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    cutoff = 6.0
    small = load_structure("mb16_43", "SiH4", dd)
    large = load_structure("mb16_43", "01", dd)
    batch = pack_structures([small, large])
    nat = batch.numbers.shape[-1]

    nbl = build_neighborlist(batch, cutoff)

    want = neighborlist_pair_set(build_neighborlist(small, cutoff))
    want |= _offset_pairs(
        neighborlist_pair_set(build_neighborlist(large, cutoff)), nat
    )
    assert neighborlist_pair_set(nbl) == want
    assert nbl.numbers_shape == tuple(batch.numbers.shape)


def test_batched_molecules_of_padding_only_have_no_pairs() -> None:
    """A batch holding no real atom at all gets an empty list, as a single
    structure of padding atoms does, instead of failing on an empty
    reduction."""
    numbers = torch.zeros((2, 3), dtype=torch.long, device=DEVICE)
    positions = torch.rand((2, 3, 3), dtype=torch.double, device=DEVICE)
    structure = Structure(numbers=numbers, positions=positions)

    nbl = build_neighborlist(structure, 5.0)

    assert neighborlist_pair_set(nbl) == set()


def test_batched_cells_of_padding_only_have_no_pairs() -> None:
    """A batch of cells holding no real atom at all gets an empty list:
    its ghost pool is empty, like the atoms of a molecular batch."""
    numbers = torch.zeros((2, 3), dtype=torch.long, device=DEVICE)
    positions = torch.rand((2, 3, 3), dtype=torch.double, device=DEVICE)
    lattice = 10.0 * torch.eye(3, dtype=torch.double, device=DEVICE)
    structure = Structure(numbers=numbers, positions=positions, lattice=lattice)

    nbl = build_neighborlist(structure, 5.0)

    assert periodic_neighborlist_pair_set(nbl) == set()


def test_batched_cells_keep_their_own_periodicity() -> None:
    """A batch of a bulk cell and a slab searches each with its own
    lattice and `periodic` mask: the slab gets no images along its open
    axis, and no pair crosses between systems."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    cutoff = 6.0
    bulk = load_structure("other", "periodic_cubic", dd)
    slab = bulk.replace(
        periodic=torch.tensor([True, True, False], device=DEVICE)
    )
    batch = pack_structures([bulk, slab])
    nat = batch.numbers.shape[-1]

    nbl = build_neighborlist(batch, cutoff)

    want = periodic_neighborlist_pair_set(build_neighborlist(bulk, cutoff))
    want |= _offset_pairs(
        periodic_neighborlist_pair_set(build_neighborlist(slab, cutoff)), nat
    )
    assert periodic_neighborlist_pair_set(nbl) == want


@pytest.mark.parametrize("pair_budget", [1, 60, 2_000_000])
def test_batched_cells_match_their_own_lists(
    pair_budget: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cells of different atom counts and periodicity (bulk, slab, wire,
    triclinic), padded to one width and searched together, hold exactly
    the pairs of each cell's own list, moved by its offset -- at two
    cutoffs and with skin, whose search radius is what the shared ghost
    pool is sized for. The pair budget makes the search run as one chunk
    or as one chunk per cell, with the same result."""
    monkeypatch.setattr(
        "tad_mctc.neighbor.list._PAIRS_PER_CELL_SEARCH", pair_budget
    )
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    bulk = load_structure("other", "periodic_cubic", dd)
    cells = [
        bulk,
        bulk.replace(periodic=torch.tensor([True, True, False], device=DEVICE)),
        bulk.replace(
            periodic=torch.tensor([True, False, False], device=DEVICE)
        ),
        load_structure("other", "periodic_triclinic", dd),
        load_structure("other", "periodic_one_atom", dd),
    ]
    batch = pack_structures(cells)
    nat = batch.numbers.shape[-1]
    cutoffs, skin = (3.0, 6.0), 0.5

    batched = build_neighborlists(batch, cutoffs, skin=skin)

    for nbl, cutoff in zip(batched, cutoffs):
        want: set[tuple[int, ...]] = set()
        for b, cell in enumerate(cells):
            own = build_neighborlist(cell, cutoff, skin=skin)
            want |= _offset_pairs(periodic_neighborlist_pair_set(own), b * nat)
        assert periodic_neighborlist_pair_set(nbl) == want


def test_ghost_pool_of_a_batch_gives_each_cell_only_its_own_images() -> None:
    """The shift table of a batch is sized for its most demanding cell, but
    a small cell next to large ones must not hand them its many image
    rings: the batch's pool holds exactly the ghosts of the cells' own
    pools."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    # 4 rings for the 5 Bohr cell, 3 for the 8 Bohr one
    cutoff = 20.0
    small = load_structure("other", "periodic_one_atom", dd)
    large = load_structure("other", "periodic_cubic", dd)
    batch = pack_structures([small, large])
    assert batch.lattice is not None and batch.periodic is not None
    is_real = batch.numbers != 0
    periodics = batch.periodic.expand(2, 3)

    wrapped, _ = wrap_to_central_cell(
        batch.positions, batch.lattice, periodics.unsqueeze(-2)
    )
    _, owner, _, system = _ghost_pool(
        is_real, wrapped, batch.lattice, periodics, cutoff
    )

    for b in range(2):
        own_pool, _, _, _ = _ghost_pool(
            is_real[b : b + 1],
            wrapped[b : b + 1],
            batch.lattice[b : b + 1],
            periodics[b : b + 1],
            cutoff,
        )
        assert int((system == b).sum()) == own_pool.shape[0]
        # atoms of the large cell come after the padded small cell
        atoms = owner[system == b]
        assert int(atoms.min()) >= b * batch.numbers.shape[-1]


def test_ghost_pool_matches_a_loop_over_systems_atoms_and_shifts() -> None:
    """The pool of a padded batch is, in this order, every real atom of
    every system at every forward shift within that system's own rings."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    cutoff = 20.0
    small = load_structure("other", "periodic_one_atom", dd)
    large = load_structure("other", "periodic_cubic", dd)
    batch = pack_structures([large, small, large])
    assert batch.lattice is not None and batch.periodic is not None
    n_systems, nat = batch.numbers.shape
    is_real = batch.numbers != 0
    periodics = batch.periodic.expand(n_systems, 3)

    ghost_positions, owner, shift, system = _ghost_pool(
        is_real, batch.positions, batch.lattice, periodics, cutoff
    )

    shifts = build_periodic_shifts(batch.lattice, periodics, cutoff).shifts
    shifts = shifts[_is_forward(shifts)]
    rings = count_image_rings_mctclib(batch.lattice, periodics, cutoff)
    expected_positions, expected_owner = [], []
    expected_shift, expected_system = [], []
    for b in range(n_systems):
        for i in range(nat):
            if not is_real[b, i]:
                continue
            for s in shifts:
                if bool((s.abs() > rings[b]).any()):
                    continue
                expected_positions.append(
                    batch.positions[b, i] + s.to(dd["dtype"]) @ batch.lattice[b]
                )
                expected_owner.append(b * nat + i)
                expected_shift.append(s)
                expected_system.append(b)

    assert torch.equal(ghost_positions, torch.stack(expected_positions))
    assert owner.tolist() == expected_owner
    assert torch.equal(shift, torch.stack(expected_shift))
    assert system.tolist() == expected_system


def test_ghost_pool_holds_the_primary_cell_and_half_of_the_images() -> None:
    """Of every pair of opposite images, the pool holds only the forward
    one: its size is the primary cell plus half of its other images."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    cutoff = 12.0
    cell = load_structure("other", "periodic_cubic", dd)
    assert cell.lattice is not None and cell.periodic is not None
    nat = cell.numbers.shape[-1]

    ghost_positions, _, shift, _ = _ghost_pool(
        torch.ones(1, nat, dtype=torch.bool, device=DEVICE),
        cell.positions.unsqueeze(0),
        cell.lattice.unsqueeze(0),
        cell.periodic.unsqueeze(0),
        cutoff,
    )
    n_shifts = build_periodic_shifts(cell.lattice, cell.periodic, cutoff)
    n_images = nat * (n_shifts.shifts.shape[0] - 1)
    assert ghost_positions.shape[0] == nat + n_images // 2
    assert bool(_is_forward(shift).all())


def test_cells_are_split_into_chunks_by_estimated_pairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Three equal cells of about 40 estimated pairs each: a budget of 100
    takes two per chunk, a budget below one cell still takes one, and a
    large budget takes all."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    cell = load_structure("other", "periodic_cubic", dd)
    batch = pack_structures([cell, cell, cell])
    lattice = batch.lattice
    assert lattice is not None

    def chunks(budget: int) -> list[tuple[int, int]]:
        monkeypatch.setattr(
            "tad_mctc.neighbor.list._PAIRS_PER_CELL_SEARCH", budget
        )
        return _split_cells_by_pair_budget(batch.numbers, lattice, 6.5)

    assert chunks(100) == [(0, 2), (2, 3)]
    assert chunks(1) == [(0, 1), (1, 2), (2, 3)]
    assert chunks(10**9) == [(0, 3)]


def test_batched_cells_with_zero_search_radius_raise() -> None:
    """As for a batch of molecules: with no search radius there is no gap
    that keeps the cells apart."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    bulk = load_structure("other", "periodic_cubic", dd)
    batch = pack_structures([bulk, bulk])

    with pytest.raises(ValueError, match="positive search radius"):
        build_neighborlist(batch, cutoff=0.0)


def test_cell_of_zero_volume_raises() -> None:
    """A slab whose open axis has a zero lattice row instead of a
    placeholder vector is rejected with a message naming the fix, not an
    error from inverting the singular cell."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    slab = Structure(
        numbers=torch.tensor([1, 1], device=DEVICE),
        positions=torch.tensor([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]], **dd),
        lattice=torch.tensor(
            [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 0.0]], **dd
        ),
        periodic=torch.tensor([True, True, False], device=DEVICE),
    )

    with pytest.raises(RuntimeError, match="placeholder vector"):
        build_neighborlist(slab, cutoff=4.0)


def test_batched_cell_of_zero_volume_raises() -> None:
    """As above, in a batch: the cell is rejected before its volume is
    used to estimate the batch's pair count."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    bulk = load_structure("other", "periodic_cubic", dd)
    slab = Structure(
        numbers=torch.tensor([1, 1], device=DEVICE),
        positions=torch.tensor([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]], **dd),
        lattice=torch.tensor(
            [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 0.0]], **dd
        ),
        periodic=torch.tensor([True, True, False], device=DEVICE),
    )
    batch = pack_structures([bulk, slab])

    with pytest.raises(RuntimeError, match="placeholder vector"):
        build_neighborlist(batch, cutoff=4.0)


def test_batched_list_is_stale_only_for_the_batch_as_a_whole() -> None:
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    bulk = load_structure("other", "periodic_cubic", dd)
    slab = bulk.replace(
        periodic=torch.tensor([True, True, False], device=DEVICE)
    )
    batch = pack_structures([bulk, slab])
    assert batch.lattice is not None

    nbl = build_neighborlist(batch, cutoff=4.0, skin=1.0)
    assert not bool(nbl.stale(batch))

    strained_lattice = batch.lattice.clone()
    strained_lattice[1] *= 1.01
    assert bool(nbl.stale(batch.replace(lattice=strained_lattice)))


def test_to_without_a_change_returns_the_list_itself() -> None:
    """A copy to the device and dtype the list already has is not a copy."""
    positions = torch.randn(10, 3, dtype=torch.float64)
    nbl = build_neighborlist(hydrogens(positions), cutoff=3.0, tile=4)

    assert nbl.to() is nbl
    assert nbl.to(dtype=torch.float64) is nbl


def test_type_changes_the_floating_dtype() -> None:
    """`type` is `to(dtype=...)`: the floating slots follow, the index
    slots stay."""
    positions = torch.randn(10, 3, dtype=torch.float64)
    nbl = build_neighborlist(hydrogens(positions), cutoff=3.0, tile=4)

    converted = nbl.type(torch.float32)

    assert converted.dtype == torch.float32
    assert converted.idx_i.dtype == torch.int32
