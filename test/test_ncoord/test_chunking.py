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
Chunked evaluation: the sparse kernel walks its neighbour list in chunks
far larger than any test system. These tests shrink the chunk size so a small system crosses many
chunks and compare against one chunk. `SMALL_CHUNK` does not divide any
list or grid here, so the last chunk is a short one.

The sparse path's `"recompute"` mode is chunked the same way; its
gradient checks live in `test_sparse_molecular.py`.
"""

from __future__ import annotations

import pytest
import torch
from torch.func import vmap

from tad_mctc.autograd import dgradcheck
from tad_mctc.io.structure import Structure, pack_structures
from tad_mctc.ncoord import cn_d3, cn_d4, cn_eeq_en
from tad_mctc.ncoord import common as common_module
from tad_mctc.ncoord.common import CNModel, sum_over_neighborlist
from tad_mctc.neighbor.list import NeighborList, build_neighborlist
from tad_mctc.tools.testing import requires_compile
from tad_mctc.typing import DD, Tensor

from ..conftest import DEVICE
from ..utils import (
    compile_fullgraph,
    jacrev,
    load_structure,
)
from .samples import bulk_and_slab

DD_DOUBLE: DD = {"device": DEVICE, "dtype": torch.double}

SMALL_CHUNK = 37
ONE_CHUNK = 2**40

# No pair weight, a symmetric one (D4) and an antisymmetric one (EEQ-EN).
MODELS = [cn_d3, cn_d4, cn_eeq_en]
MODEL_IDS = ["d3", "d4", "eeq_en"]


def _set_chunk_size(monkeypatch: pytest.MonkeyPatch, size: int) -> None:
    monkeypatch.setattr(common_module, "_CHUNK_SIZE_CPU", size)
    monkeypatch.setattr(common_module, "_CHUNK_SIZE_GPU", size)


def _value_and_gradients(
    model: CNModel, structure: Structure, pairs: NeighborList | None
) -> list[Tensor]:
    """The coordination number and its gradients with respect to the
    positions and, for a cell, the lattice."""
    positions = structure.positions.clone().requires_grad_(True)
    inputs = [positions]
    lattice = None
    if structure.lattice is not None:
        lattice = structure.lattice.clone().requires_grad_(True)
        inputs.append(lattice)

    cn = model(structure.replace(positions=positions, lattice=lattice), pairs)

    # Uneven weights, so every atom's coordination number enters the
    # gradient differently.
    weights = torch.linspace(0.5, 1.5, cn.numel(), **DD_DOUBLE)
    energy = (cn * weights.reshape(cn.shape)).sum()
    gradients = torch.autograd.grad(energy, inputs)

    return [cn.detach(), *gradients]


def _assert_chunks_match_one_chunk(
    monkeypatch: pytest.MonkeyPatch,
    model: CNModel,
    structure: Structure,
    pairs: NeighborList | None,
) -> None:
    _set_chunk_size(monkeypatch, ONE_CHUNK)
    one_chunk = _value_and_gradients(model, structure, pairs)

    _set_chunk_size(monkeypatch, SMALL_CHUNK)
    chunked = _value_and_gradients(model, structure, pairs)

    for expected, actual in zip(one_chunk, chunked):
        assert torch.allclose(actual, expected, atol=1e-12, rtol=0)


def _padded_molecule_batch() -> Structure:
    return pack_structures(
        [
            load_structure("mb16_43", "01", DD_DOUBLE),
            load_structure("mb16_43", "SiH4", DD_DOUBLE),
        ]
    )


########################################################################
# Sparse path


@pytest.mark.parametrize("model", MODELS, ids=MODEL_IDS)
def test_sparse_molecule(
    monkeypatch: pytest.MonkeyPatch, model: CNModel
) -> None:
    structure = load_structure("mb16_43", "01", DD_DOUBLE)
    nbl = build_neighborlist(structure, cutoff=model.cutoff)
    _assert_chunks_match_one_chunk(monkeypatch, model, structure, nbl)


@pytest.mark.parametrize("model", MODELS, ids=MODEL_IDS)
def test_sparse_padded_batch(
    monkeypatch: pytest.MonkeyPatch, model: CNModel
) -> None:
    structure = _padded_molecule_batch()
    nbl = build_neighborlist(structure, cutoff=model.cutoff)
    _assert_chunks_match_one_chunk(monkeypatch, model, structure, nbl)


@pytest.mark.parametrize("model", MODELS, ids=MODEL_IDS)
def test_sparse_shared_cell(
    monkeypatch: pytest.MonkeyPatch, model: CNModel
) -> None:
    structure = load_structure("other", "periodic_triclinic", DD_DOUBLE)
    nbl = build_neighborlist(structure, cutoff=model.cutoff)
    _assert_chunks_match_one_chunk(monkeypatch, model, structure, nbl)


@pytest.mark.parametrize("model", MODELS, ids=MODEL_IDS)
def test_sparse_per_system_cells(
    monkeypatch: pytest.MonkeyPatch, model: CNModel
) -> None:
    structure = bulk_and_slab(DD_DOUBLE)
    nbl = build_neighborlist(structure, cutoff=model.cutoff)
    _assert_chunks_match_one_chunk(monkeypatch, model, structure, nbl)


########################################################################
# Transforms and gradient checks through several chunks


def test_sparse_vmap_jacrev_matches_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_chunk_size(monkeypatch, SMALL_CHUNK)
    structure = load_structure("mb16_43", "01", DD_DOUBLE)
    nbl = build_neighborlist(structure, cutoff=cn_d4.cutoff, skin=1.0)

    def f(positions: Tensor) -> Tensor:
        return cn_d4(structure.replace(positions=positions), nbl)

    batch = torch.stack([structure.positions + 0.01 * k for k in range(3)])
    batched = vmap(jacrev(f))(batch)
    looped = torch.stack([jacrev(f)(positions) for positions in batch])

    assert torch.allclose(batched, looped, atol=1e-12, rtol=0)


@pytest.mark.grad
def test_sparse_gradcheck(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_chunk_size(monkeypatch, SMALL_CHUNK)
    structure = load_structure("mb16_43", "SiH4", DD_DOUBLE)
    nbl = build_neighborlist(structure, cutoff=cn_eeq_en.cutoff, skin=1.0)

    def f(positions: Tensor) -> Tensor:
        return cn_eeq_en(structure.replace(positions=positions), nbl)

    positions = structure.positions.clone().requires_grad_(True)
    assert dgradcheck(f, positions)


########################################################################
# Compilation
#
# Under `torch.compile` the chunk loop would be unrolled, one copy of the
# pair kernel per chunk. The sparse kernel therefore evaluates the list as
# one chunk while compiling; the test counts how often the pair kernel is
# traced.


@requires_compile
def test_sparse_compiles_as_one_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_chunk_size(monkeypatch, SMALL_CHUNK)
    structure = load_structure("mb16_43", "01", DD_DOUBLE)
    nbl = build_neighborlist(structure, cutoff=cn_d3.cutoff)

    traced_chunks: list[int] = []
    pair_contributions = common_module._sparse_pair_contributions

    def counting_pair_contributions(*args, **kwargs):  # type: ignore
        traced_chunks.append(1)
        return pair_contributions(*args, **kwargs)

    monkeypatch.setattr(
        common_module, "_sparse_pair_contributions", counting_pair_contributions
    )

    # A fresh `Structure`, not `structure.replace`: Dynamo in older
    # PyTorch cannot trace `dataclasses.replace`.
    def f(positions: Tensor) -> Tensor:
        return cn_d3(
            Structure(numbers=structure.numbers, positions=positions), nbl
        )

    torch._dynamo.reset()  # pylint: disable=protected-access
    compiled = compile_fullgraph(f)(structure.positions)

    assert len(traced_chunks) == 1
    assert torch.allclose(compiled, f(structure.positions), atol=1e-12)


def test_sum_over_neighborlist_is_reusable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The walk sums any per-pair quantity, chunked, without the CN model:
    counting one per pair gives each atom's number of neighbours."""
    structure = load_structure("mb16_43", "01", DD_DOUBLE)
    nbl = build_neighborlist(structure, cutoff=cn_d3.cutoff)
    positions = structure.positions

    def one_per_pair(
        idx_i: Tensor, idx_j: Tensor, mask: Tensor, shift: Tensor
    ) -> tuple[Tensor, Tensor]:
        ones = mask.to(positions.dtype)
        return ones, ones

    nat = positions.shape[0]
    real = nbl.mask
    expected = torch.bincount(nbl.idx_i[real], minlength=nat) + torch.bincount(
        nbl.idx_j[real], minlength=nat
    )

    for size in (ONE_CHUNK, SMALL_CHUNK):
        _set_chunk_size(monkeypatch, size)
        for mode in ("graph", "recompute"):
            total = sum_over_neighborlist(
                nbl, one_per_pair, positions, mode=mode
            )
            assert torch.equal(total, expected.to(total.dtype))


def test_chunk_size_depends_on_the_device() -> None:
    """The CPU and the other devices each have their own chunk size."""
    assert common_module._chunk_size(torch.empty(1)) == (
        common_module._CHUNK_SIZE_CPU
    )
    assert common_module._chunk_size(torch.empty(1, device="meta")) == (
        common_module._CHUNK_SIZE_GPU
    )


def test_graph_mode_under_tracing_takes_one_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """While `torch.compile` traces, the graph mode leaves the chunk loop
    out, so a small chunk size no longer splits the list."""
    structure = load_structure("mb16_43", "01", DD_DOUBLE)
    nbl = build_neighborlist(structure, cn_d3.cutoff)

    _set_chunk_size(monkeypatch, ONE_CHUNK)
    one_chunk = _value_and_gradients(cn_d3, structure, nbl)

    _set_chunk_size(monkeypatch, 1)
    monkeypatch.setattr(common_module, "is_compiling", lambda: True)
    traced = _value_and_gradients(cn_d3, structure, nbl)

    for expected, actual in zip(one_chunk, traced):
        assert torch.allclose(actual, expected, atol=1e-12, rtol=0)
