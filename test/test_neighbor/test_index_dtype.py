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
Test the ``int32`` storage of the neighbour list's atom indices.

``idx_i``/``idx_j`` are stored as ``int32`` (9 instead of 17 bytes per
slot). From PyTorch 2.8 the consumers use ``int32`` slices directly; before,
they widen them to ``int64`` one chunk at a time, because older PyTorch
rejects a narrower index in the backward pass. Either way the results must
not differ by a single bit from an ``int64`` list, under every transform.
"""

from __future__ import annotations

import types
from functools import partial

import pytest
import torch

from tad_mctc._version import __tversion__
from tad_mctc.io.structure import Structure, pack_structures
from tad_mctc.ncoord import cn_d3
from tad_mctc.ncoord.common import (
    _pad_atoms,
    _sparse_pair_contributions,
    resolve_table,
    sum_over_neighborlist,
)
from tad_mctc.neighbor import _distance_kernels, _native, gather_index
from tad_mctc.neighbor._tiles import Tiles, tile_pairs
from tad_mctc.neighbor.list import (
    _IDX_DTYPE,
    NeighborList,
    _check_index_range,
    build_neighborlist,
)
from tad_mctc.typing import DD, Tensor

from ..conftest import DEVICE
from ..utils import (
    hydrogens,
    jacfwd,
    jacrev,
    load_structure,
    run_compiled_or_skip,
)

_INT32_MAX = 2**31 - 1
DD_DOUBLE: DD = {"device": DEVICE, "dtype": torch.double}


def _assert_same(actual: Tensor, expected: Tensor) -> None:
    """Bit-for-bit on the CPU. CUDA scatter-adds use float atomics whose
    order varies from run to run, so there two runs agree to rounding only."""
    if DEVICE is None or DEVICE.type == "cpu":
        assert torch.equal(actual, expected)
    else:
        assert torch.allclose(actual, expected, rtol=1e-12, atol=1e-12)


def _molecule() -> Structure:
    return load_structure("mb16_43", "01", DD_DOUBLE)


def _cell() -> Structure:
    return load_structure("other", "periodic_triclinic", DD_DOUBLE)


def _int64_view(nbl: NeighborList) -> types.SimpleNamespace:
    """What the list looked like before: ``int64`` indices, duck-typed for
    :func:`sum_over_neighborlist`, which reads only these four fields."""
    return types.SimpleNamespace(
        idx_i=nbl.idx_i.long(),
        idx_j=nbl.idx_j.long(),
        mask=nbl.mask,
        shift=nbl.shift,
    )


def _cn_from(nbl: object, structure: Structure, positions: Tensor) -> Tensor:
    """The sparse CN of ``positions``, summed over ``nbl`` by the library's
    own walk, so only the index dtype differs between the callers."""
    atoms = _pad_atoms(
        structure.numbers,
        positions,
        rcov=resolve_table(cn_d3.rcov, positions),
        en=None,
        lattice=structure.lattice,
        atoms_per_system=structure.numbers.shape[-1],
    )
    contributions = partial(
        _sparse_pair_contributions,
        atoms=atoms,
        count=cn_d3.count,
        pair_weight=None,
        cutoff=cn_d3.cutoff,
    )
    return sum_over_neighborlist(nbl, contributions, positions)  # type: ignore[arg-type]


# ---------------------------------------------------------------- storage


def test_a_build_stores_int32() -> None:
    molecule = build_neighborlist(_molecule(), cutoff=6.0)
    cell = build_neighborlist(_cell(), cutoff=6.0)

    for nbl in (molecule, cell):
        assert nbl.idx_i.dtype == torch.int32 == _IDX_DTYPE
        assert nbl.idx_j.dtype == torch.int32
        assert nbl.mask.dtype == torch.bool
    assert molecule.shift.dtype == torch.int16


def test_real_entries_keep_the_stored_dtype() -> None:
    nbl = build_neighborlist(_molecule(), cutoff=6.0)

    idx_i, idx_j, _ = nbl.real_entries()

    assert idx_i.dtype == torch.int32
    assert idx_j.dtype == torch.int32
    assert idx_i.shape[0] == int(nbl.mask.sum())


def test_bytes_per_slot_are_nine_and_fifteen() -> None:
    for structure, expected in ((_molecule(), 9), (_cell(), 15)):
        nbl = build_neighborlist(structure, cutoff=6.0)
        stored = [nbl.idx_i, nbl.idx_j, nbl.mask]
        if nbl.periodic:
            stored.append(nbl.shift)
        total = sum(t.element_size() * t.nelement() for t in stored)
        assert total == expected * nbl.idx_i.shape[0]


# ------------------------------------------- widening, bit for bit


@pytest.mark.parametrize("structure", [_molecule, _cell])
def test_cn_matches_int64_indices_exactly(structure: object) -> None:
    st = structure()  # type: ignore[operator]
    nbl = build_neighborlist(st, cutoff=cn_d3.cutoff)

    with_int32 = _cn_from(nbl, st, st.positions)
    with_int64 = _cn_from(_int64_view(nbl), st, st.positions)

    _assert_same(with_int32, with_int64)
    _assert_same(with_int32, cn_d3(st, pairs=nbl))


@pytest.mark.parametrize("structure", [_molecule, _cell])
@pytest.mark.parametrize("transform", ["jacrev", "jacfwd", "vmap"])
def test_transforms_match_int64_indices_exactly(
    structure: object, transform: str
) -> None:
    st = structure()  # type: ignore[operator]
    nbl = build_neighborlist(st, cutoff=cn_d3.cutoff)
    view = _int64_view(nbl)

    def run(source: object) -> Tensor:
        def f(p: Tensor) -> Tensor:
            return _cn_from(source, st, p)

        if transform == "jacrev":
            return jacrev(f)(st.positions)  # type: ignore[no-any-return]
        if transform == "jacfwd":
            return jacfwd(f)(st.positions)  # type: ignore[no-any-return]
        batch = torch.stack([st.positions, 1.01 * st.positions])
        return torch.func.vmap(f)(batch)  # type: ignore[no-any-return]

    _assert_same(run(nbl), run(view))


@pytest.mark.parametrize("structure", [_molecule, _cell])
@pytest.mark.parametrize("wrap_jacrev", [False, True])
def test_compiled_matches_int64_indices(
    structure: object, wrap_jacrev: bool
) -> None:
    st = structure()  # type: ignore[operator]
    nbl = build_neighborlist(st, cutoff=cn_d3.cutoff)

    def f(p: Tensor) -> Tensor:
        return _cn_from(nbl, st, p)

    def reference(p: Tensor) -> Tensor:
        return _cn_from(_int64_view(nbl), st, p)

    fn, ref = (jacrev(f), jacrev(reference)) if wrap_jacrev else (f, reference)
    compiled = run_compiled_or_skip(fn, st.positions)

    assert torch.allclose(compiled, ref(st.positions), rtol=1e-12, atol=1e-12)


def test_gradient_through_the_public_model_matches_int64() -> None:
    """The public ``cn_d3(structure, pairs=nbl)`` differentiates through the
    chunk widening exactly like the int64 walk."""
    st = _molecule()
    nbl = build_neighborlist(st, cutoff=cn_d3.cutoff)

    def public(p: Tensor) -> Tensor:
        return cn_d3(st.replace(positions=p), pairs=nbl)

    def widened(p: Tensor) -> Tensor:
        return _cn_from(_int64_view(nbl), st, p)

    _assert_same(jacrev(public)(st.positions), jacrev(widened)(st.positions))


def test_several_chunks_give_the_same_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Widening per chunk is independent of the chunk size."""
    st = _molecule()
    nbl = build_neighborlist(st, cutoff=cn_d3.cutoff)
    one_chunk = cn_d3(st, pairs=nbl)

    monkeypatch.setattr("tad_mctc.ncoord.common._CHUNK_SIZE_CPU", 1000)
    assert nbl.idx_i.shape[0] > 3000
    many = cn_d3(st, pairs=nbl)

    assert torch.allclose(one_chunk, many, rtol=1e-13, atol=1e-13)


# ------------------------------------------------------- version gate


def test_gather_index_passes_int32_through_when_the_backward_accepts_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_distance_kernels, "_INT32_INDEX_BACKWARD", True)
    index = torch.arange(5, dtype=torch.int32)
    assert gather_index(index) is index


def test_gather_index_widens_when_the_backward_needs_int64(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_distance_kernels, "_INT32_INDEX_BACKWARD", False)
    index = torch.arange(5, dtype=torch.int32)
    gathered = gather_index(index)
    assert gathered.dtype == torch.int64
    assert torch.equal(gathered, index.long())


def test_gather_index_follows_the_pytorch_version() -> None:
    index = torch.arange(5, dtype=torch.int32)

    gathered = gather_index(index)

    if __tversion__ >= (2, 8, 0):
        assert gathered is index
    else:
        assert gathered.dtype == torch.int64
        assert torch.equal(gathered, index.long())
    assert gather_index(index.long()).dtype == torch.int64


@pytest.mark.parametrize("structure", [_molecule, _cell])
def test_forced_widening_gives_the_same_bits(
    structure: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pre-2.8 path (widen every chunk) and the 2.8+ path (``int32``
    slices) agree exactly, value and gradient."""
    st = structure()  # type: ignore[operator]
    nbl = build_neighborlist(st, cutoff=cn_d3.cutoff)

    def f(p: Tensor) -> Tensor:
        return cn_d3(st.replace(positions=p), pairs=nbl)

    direct = f(st.positions), jacrev(f)(st.positions)
    monkeypatch.setattr(_distance_kernels, "_INT32_INDEX_BACKWARD", False)
    widened = f(st.positions), jacrev(f)(st.positions)

    _assert_same(direct[0], widened[0])
    _assert_same(direct[1], widened[1])


def _saved_index_copies(nbl: NeighborList, st: Structure, mode: str) -> int:
    """Bytes of integer tensors that autograd keeps for the backward pass
    of the sparse CN and that are not views of the list's own storage."""
    stored = (nbl.idx_i, nbl.idx_j, nbl.shift)
    own = {t.untyped_storage().data_ptr() for t in stored}
    copies: dict[int, int] = {}

    def pack(t: Tensor) -> Tensor:
        storage = t.untyped_storage()
        if not t.is_floating_point() and t.dtype != torch.bool:
            if storage.data_ptr() not in own:
                copies[storage.data_ptr()] = storage.nbytes()
        return t

    positions = st.positions.clone().requires_grad_(True)
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        cn = cn_d3(st.replace(positions=positions), pairs=nbl, mode=mode)
    torch.autograd.grad(cn.sum(), positions)
    return sum(copies.values())


@pytest.mark.skipif(
    __tversion__ < (2, 8, 0), reason="Before 2.8 the chunks are widened"
)
@pytest.mark.parametrize("mode", ["graph", "recompute"])
def test_the_backward_pass_keeps_no_index_copies(mode: str) -> None:
    """The memory regression of widening: every widened chunk was kept
    until the backward pass, 16 bytes per pair for the whole list."""
    st = _molecule()
    nbl = build_neighborlist(st, cutoff=cn_d3.cutoff)

    assert _saved_index_copies(nbl, st, mode) == 0


@pytest.mark.parametrize("mode", ["graph", "recompute"])
def test_widening_keeps_one_int64_copy_of_the_list(
    mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The accounting above sees the copies when there are some."""
    monkeypatch.setattr(_distance_kernels, "_INT32_INDEX_BACKWARD", False)
    st = _molecule()
    nbl = build_neighborlist(st, cutoff=cn_d3.cutoff)

    assert _saved_index_copies(nbl, st, mode) == 16 * nbl.idx_i.shape[0]


# ----------------------------------------------------------------- batches


def _molecule_batch() -> Structure:
    return pack_structures(
        [
            load_structure("mb16_43", "SiH4", DD_DOUBLE),
            load_structure("mb16_43", "01", DD_DOUBLE),
            load_structure("mb16_43", "SiH4", DD_DOUBLE),
        ]
    )


def _cells() -> list[Structure]:
    return [
        load_structure("other", "periodic_cubic", DD_DOUBLE),
        load_structure("other", "periodic_triclinic", DD_DOUBLE),
        load_structure("other", "periodic_one_atom", DD_DOUBLE),
    ]


def _canonical(
    i: int, j: int, shift: tuple[int, ...]
) -> tuple[int, int, tuple[int, ...]]:
    """One of the two orientations of a pair, ``(i, j, +shift)`` and
    ``(j, i, -shift)``: which atom is ``idx_i`` is not specified."""
    reverse = (j, i, tuple(-x for x in shift))
    return min((i, j, shift), reverse)


def _pairs_of(nbl: NeighborList) -> set[tuple[int, int, tuple[int, ...]]]:
    idx_i, idx_j, shift = nbl.real_entries()
    return {
        _canonical(int(i), int(j), tuple(int(x) for x in s))
        for i, j, s in zip(idx_i, idx_j, shift)
    }


@pytest.mark.parametrize("pair_budget", [1, 2_000_000])
def test_batched_cells_with_offsets_stay_int32(
    pair_budget: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pair budget of 1 searches one cell per chunk, so the second and
    third cell are renumbered by a non-zero offset (``first * nat``)."""
    monkeypatch.setattr(
        "tad_mctc.neighbor.list._PAIRS_PER_CELL_SEARCH", pair_budget
    )
    cells = _cells()
    batch = pack_structures(cells)
    nat = batch.numbers.shape[-1]

    nbl = build_neighborlist(batch, cutoff=4.0, skin=0.5)

    assert nbl.idx_i.dtype == torch.int32
    assert nbl.idx_j.dtype == torch.int32
    assert int(nbl.idx_i.max()) == batch.numbers.numel()  # the phantom

    want: set[tuple[int, int, tuple[int, ...]]] = set()
    for b, cell in enumerate(cells):
        own = build_neighborlist(cell, cutoff=4.0, skin=0.5)
        want |= {
            _canonical(i + b * nat, j + b * nat, shift)
            for i, j, shift in _pairs_of(own)
            if i < cell.numbers.shape[-1] and j < cell.numbers.shape[-1]
        }
    assert _pairs_of(nbl) == want


def test_batched_molecules_match_their_own_lists() -> None:
    batch = _molecule_batch()
    nat = batch.numbers.shape[-1]

    nbl = build_neighborlist(batch, cutoff=8.0)

    assert nbl.idx_i.dtype == torch.int32
    assert int(nbl.idx_i.max()) == batch.numbers.numel()
    got = {(i, j) for i, j, _ in _pairs_of(nbl)}
    want: set[tuple[int, int]] = set()
    for b in range(batch.numbers.shape[0]):
        real = batch.numbers[b] != 0
        own = build_neighborlist(
            Structure(
                numbers=batch.numbers[b][real],
                positions=batch.positions[b][real],
            ),
            cutoff=8.0,
        )
        atoms = real.nonzero().squeeze(-1)
        want |= {
            (int(atoms[i]) + b * nat, int(atoms[j]) + b * nat)
            for i, j, _ in _pairs_of(own)
        }
    assert got == want


def test_batched_cn_matches_the_single_systems() -> None:
    batch = _molecule_batch()
    nbl = build_neighborlist(batch, cutoff=cn_d3.cutoff)

    cn = cn_d3(batch, pairs=nbl)

    for b in range(batch.numbers.shape[0]):
        real = batch.numbers[b] != 0
        single = Structure(
            numbers=batch.numbers[b][real], positions=batch.positions[b][real]
        )
        alone = cn_d3(single, pairs=build_neighborlist(single, cn_d3.cutoff))
        assert torch.allclose(cn[b][real], alone, rtol=1e-12, atol=1e-12)


# ------------------------------------------------------------------ guard


def test_check_index_range_is_a_pure_size_check() -> None:
    _check_index_range(0)
    _check_index_range(_INT32_MAX)

    with pytest.raises(ValueError, match=str(_INT32_MAX + 1)) as info:
        _check_index_range(_INT32_MAX + 1)
    assert str(_INT32_MAX) in str(info.value)

    with pytest.raises(ValueError, match="ghost pool"):
        _check_index_range(2**40, "ghost pool")


def test_a_build_beyond_the_index_range_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The limit is lowered instead of allocating a huge system. The 16
    atoms plus the phantom index exceed it."""
    monkeypatch.setattr("tad_mctc.neighbor.list._IDX_MAX", 10)
    st = _molecule()
    assert st.numbers.shape[-1] > 10

    with pytest.raises(ValueError, match="int32 maximum"):
        build_neighborlist(st, cutoff=6.0)


def test_a_batch_beyond_the_index_range_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each system fits, but the flattened batch (``b * nat + i``, plus
    the phantom) does not."""
    batch = _molecule_batch()
    nat = batch.numbers.shape[-1]
    monkeypatch.setattr("tad_mctc.neighbor.list._IDX_MAX", 2 * nat)

    with pytest.raises(ValueError, match=str(batch.numbers.numel())):
        build_neighborlist(batch, cutoff=6.0)


def test_a_ghost_pool_beyond_the_index_range_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Few atoms, many images: the atom count passes, the ghost pool of the
    periodic search does not."""
    cell = _cell()
    n_atoms = int(cell.numbers.numel())
    monkeypatch.setattr("tad_mctc.neighbor.list._IDX_MAX", n_atoms + 1)

    with pytest.raises(ValueError, match="ghost pool"):
        build_neighborlist(cell, cutoff=12.0)


def test_the_limit_itself_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    st = _molecule()
    monkeypatch.setattr(
        "tad_mctc.neighbor.list._IDX_MAX", int(st.numbers.numel())
    )

    nbl = build_neighborlist(st, cutoff=6.0)

    assert int(nbl.idx_i.max()) == int(st.numbers.numel())


@pytest.mark.native
def test_the_native_kernel_refuses_a_padding_value_beyond_int32() -> None:
    if not _native.is_available():
        pytest.skip("native extension is not available")
    positions = torch.randn(40, 3, dtype=torch.double, device="cpu") * 3.0
    tiles = Tiles(positions, tile=8)
    tile_a, tile_b = tile_pairs(tiles, 4.0)

    with pytest.raises(RuntimeError, match="int32"):
        _native.atom_pairs_within_thresholds_native(
            tiles.index,
            tiles.valid,
            tile_a,
            tile_b,
            positions,
            (4.0,),
            None,
            (4096, _INT32_MAX + 1),
            4096,
        )


@pytest.mark.native
def test_the_native_kernel_writes_int32() -> None:
    if not _native.is_available():
        pytest.skip("native extension is not available")
    positions = torch.randn(40, 3, dtype=torch.double, device="cpu") * 3.0
    tiles = Tiles(positions, tile=8)
    tile_a, tile_b = tile_pairs(tiles, 4.0)

    ((idx_i, idx_j, n_found),) = _native.atom_pairs_within_thresholds_native(  # type: ignore[misc]
        tiles.index,
        tiles.valid,
        tile_a,
        tile_b,
        positions,
        (4.0,),
        None,
        (4096, 40),
        4096,
    )

    assert idx_i.dtype == torch.int32 == idx_j.dtype
    assert n_found > 0
    assert bool((idx_i[n_found:] == 40).all())


# ------------------------------------------------------ int64 still loads


def test_an_int64_list_is_loaded_as_int32() -> None:
    built = build_neighborlist(_molecule(), cutoff=6.0)

    loaded = NeighborList.create(
        idx_i=built.idx_i.long(),
        idx_j=built.idx_j.long(),
        shift=built.shift,
        mask=built.mask,
        build_positions=built.build_positions,
        cutoff=built.cutoff,
        skin=built.skin,
        overflow=False,
    )

    assert loaded.idx_i.dtype == torch.int32
    assert torch.equal(loaded.idx_i, built.idx_i)
    assert torch.equal(loaded.idx_j, built.idx_j)
    loaded.check_compatible(_molecule(), 6.0)


def test_an_int64_list_beyond_the_range_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built = build_neighborlist(_molecule(), cutoff=6.0)
    monkeypatch.setattr("tad_mctc.neighbor.list._IDX_MAX", 5)

    with pytest.raises(ValueError, match="int32 maximum"):
        NeighborList.create(
            idx_i=built.idx_i.long(),
            idx_j=built.idx_j.long(),
            shift=built.shift,
            mask=built.mask,
            build_positions=built.build_positions,
            cutoff=built.cutoff,
            skin=built.skin,
            overflow=False,
        )


def test_another_index_dtype_is_refused() -> None:
    built = build_neighborlist(_molecule(), cutoff=6.0)

    with pytest.raises(ValueError, match="int32"):
        NeighborList.create(
            idx_i=built.idx_i.short(),
            idx_j=built.idx_j,
            shift=built.shift,
            mask=built.mask,
            build_positions=built.build_positions,
            cutoff=built.cutoff,
            skin=built.skin,
            overflow=False,
        )


def test_hydrogens_helper_still_builds_int32() -> None:
    positions = torch.randn(10, 3, dtype=torch.float64)
    nbl = build_neighborlist(hydrogens(positions), cutoff=3.0, tile=4)

    assert nbl.idx_i.dtype == torch.int32
