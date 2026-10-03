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
The sparse molecular quadrant: `CNModel` with `pairs=nbl`, a list
built for a molecule, checked against the dense all-pairs path.

Covers a cutoff that actually truncates, arbitrary pair weights, nested
derivatives, `vmap` of `jacrev`, the `"recompute"` mode, and a list moved
between devices. Agreement with the Fortran references is checked for
every evaluation path in `test_reference.py` and `test_grad/`, and
`vmap` and `torch.compile` in `test_transforms.py` and `test_compile.py`;
the checks shared with periodic lists live in `test_sparse.py`.
"""

from __future__ import annotations

import pytest
import torch
from torch.func import jacrev, vmap

from tad_mctc.autograd import dgradcheck, dgradgradcheck
from tad_mctc.data.structures import get_structure
from tad_mctc.io.structure import Structure
from tad_mctc.ncoord import (
    CNModel,
    cn_d3,
    cn_d4,
    cn_eeq,
    cn_eeq_en,
    cn_eeqbc,
    cn_eeqbc_en,
    cn_gfn2,
)
from tad_mctc.ncoord import common as common_module
from tad_mctc.ncoord.common import NeighborListMode
from tad_mctc.ncoord.count import erf_count, exp_count
from tad_mctc.neighbor.list import build_neighborlist
from tad_mctc.typing import DD, Callable, Tensor

from ..conftest import DEVICE
from ..utils import hydrogens

########################################################################
# A cutoff that truncates
#
# On a small molecule -- like every sample in `test_reference.py` -- every
# pair sits inside the default cutoff, so comparing the two paths there
# would pass even if the sparse path's cutoff mask were missing entirely.
#
# A real, merely *large* molecule does not fix this: every counting
# function here decays to numerical zero well before any realistic
# cutoff, so masking a real bonded geometry (checked with mstore's
# 123-atom `polyala_12` polyalanine chain, whose 42 Bohr maximum pairwise
# distance does exceed every `cn_*` preset's own 25-30 Bohr default)
# changes the coordination number by at most 0.001 -- a completely broken
# cutoff mask would still agree with the unmasked reference to ~1e-12.
# These tests instead use a synthetic cloud whose density (0.01
# atoms/Bohr**3) is deliberately higher than any real molecule -- a
# condensed-phase stand-in -- so that an 8 Bohr cutoff excludes pairs that
# still contribute meaningfully, and build both paths at that same
# cutoff via `CNModel.replace`.

TRUNCATING_CUTOFF = 8.0

truncated_variants: dict[str, CNModel] = {
    "d3": cn_d3.replace(cutoff=TRUNCATING_CUTOFF),
    "gfn2": cn_gfn2.replace(cutoff=TRUNCATING_CUTOFF),
    "eeq": cn_eeq.replace(cutoff=TRUNCATING_CUTOFF),
    "d4": cn_d4.replace(cutoff=TRUNCATING_CUTOFF),
    "eeq_en": cn_eeq_en.replace(cutoff=TRUNCATING_CUTOFF),
    "eeqbc": cn_eeqbc.replace(cutoff=TRUNCATING_CUTOFF),
    "eeqbc_en": cn_eeqbc_en.replace(cutoff=TRUNCATING_CUTOFF),
}


def _condensed_positions(nat: int, dtype: torch.dtype) -> Tensor:
    """A fixed-seed, roughly condensed-density cloud of `nat` points."""
    generator = torch.Generator(device=DEVICE).manual_seed(0)
    density = 0.01  # atoms per Bohr**3, roughly condensed phase
    box_edge = (nat / density) ** (1.0 / 3.0)
    return (
        torch.rand(nat, 3, dtype=dtype, device=DEVICE, generator=generator)
        * box_edge
    )


def _condensed_cloud(nat: int, dtype: torch.dtype, seed: int) -> Structure:
    """`nat` random elements from H to Ar on `_condensed_positions`."""
    numbers = torch.randint(
        1,
        18,
        (nat,),
        device=DEVICE,
        generator=torch.Generator(device=DEVICE).manual_seed(seed),
    )
    return Structure(
        numbers=numbers, positions=_condensed_positions(nat, dtype)
    )


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_truncation_bites_at_condensed_density(dtype: torch.dtype) -> None:
    """Sanity check for the fixture itself: at this size and cutoff,
    truncating really does change the coordination number, so the tests
    below are not accidentally trivial. A cutoff far larger than the box
    stands in for "untruncated", since `CNModel.cutoff` is always a real
    mask."""
    structure = _condensed_cloud(500, dtype, seed=1)

    truncated = CNModel(count=exp_count, cutoff=TRUNCATING_CUTOFF)
    untruncated = CNModel(count=exp_count, cutoff=1.0e6)

    diff = (truncated(structure) - untruncated(structure)).abs()
    assert diff.max() > 0.1


@pytest.mark.parametrize("name", list(truncated_variants))
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_sparse_matches_dense_masked_at_same_cutoff(
    dtype: torch.dtype, name: str
) -> None:
    """All seven variants, dense vs. sparse, both masked at the same
    `TRUNCATING_CUTOFF`, on a system large enough for the cutoff to
    genuinely discard pairs. Catches a `pair_weight` that is wrong for
    one variant only, and, together with
    `test_truncation_bites_at_condensed_density` above, confirms the
    dense reference is truncated too, not left unmasked."""
    structure = _condensed_cloud(500, dtype, seed=2)
    model = truncated_variants[name]

    dense = model(structure)

    nbl = build_neighborlist(structure, cutoff=TRUNCATING_CUTOFF, tile=16)
    sparse = model(structure, pairs=nbl)

    # Both paths sum the same pairs but in a different order (a dense
    # row-sum vs. a scattered `index_add`), so float32 agreement is
    # bounded by float32 rounding over ~500 atoms' worth of terms, not by
    # the 1e-12 that float64 reaches.
    tol = 1e-12 if dtype == torch.double else 5e-4
    assert pytest.approx(dense.cpu(), abs=tol) == sparse.cpu()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_sparse_matches_dense_with_skin(dtype: torch.dtype) -> None:
    """A list built with extra `skin` holds pairs beyond the cutoff. The
    model must mask them out itself, so the sparse result still matches
    the dense one masked at the model's own cutoff, not the list's larger
    search radius."""
    structure = _condensed_cloud(300, dtype, seed=0)
    model = CNModel(count=exp_count, cutoff=TRUNCATING_CUTOFF)

    dense = model(structure)

    nbl_no_skin = build_neighborlist(
        structure, cutoff=TRUNCATING_CUTOFF, tile=16, skin=0.0
    )
    nbl_with_skin = build_neighborlist(
        structure, cutoff=TRUNCATING_CUTOFF, tile=16, skin=2.0
    )

    sparse_no_skin = model(structure, pairs=nbl_no_skin)
    sparse_with_skin = model(structure, pairs=nbl_with_skin)

    tol = 1e-12 if dtype == torch.double else 5e-4
    assert pytest.approx(dense.cpu(), abs=tol) == sparse_no_skin.cpu()
    assert pytest.approx(dense.cpu(), abs=tol) == sparse_with_skin.cpu()


########################################################################
# Pair weights


def test_matches_dense_for_arbitrary_pair_weight() -> None:
    """Both orientations of every pair are evaluated, so any elementwise
    `pair_weight`, symmetric or not, matches the dense path exactly -- not
    only the antisymmetric weights the presets use."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    def arbitrary_weight(en_i: Tensor, en_j: Tensor) -> Tensor:
        return 1.0 + 0.1 * en_i - 0.3 * en_j

    model = CNModel(count=erf_count, cutoff=25.0, pair_weight=arbitrary_weight)
    structure = get_structure("mb16_43", "01", **dd)

    nbl = build_neighborlist(structure, model.cutoff, tile=4)
    sparse = model(structure, pairs=nbl)

    assert torch.allclose(model(structure), sparse, atol=1e-12, rtol=0)


########################################################################
# Transforms with one fixed list
#
# Every check contracts the coordination number with a fixed random weight
# vector before differentiating. Contracting with `.sum()` instead would be
# wrong: the total coordination number is translationally invariant, so
# `grad(cn.sum())` is identically zero and every check below would pass
# against a badly broken implementation just as easily as a correct one.

torch.manual_seed(0)

NAT = 16
NUMBERS = torch.randint(1, 18, (NAT,))
POSITIONS = torch.rand(NAT, 3, dtype=torch.float64) * 9.0
WEIGHT = torch.rand(NAT, dtype=torch.float64)
STRUCTURE = Structure(numbers=NUMBERS, positions=POSITIONS)
MODEL = CNModel(count=exp_count, cutoff=25.0)

# An explicit, small capacity keeps the nested-derivative checks fast: the default auto-bucketed capacity rounds up to 4096, and
# every consumption op below pays for that many pairs regardless of how
# few of NAT=16 atoms' ~120 possible pairs are real.
NBL = build_neighborlist(
    hydrogens(POSITIONS), cutoff=MODEL.cutoff, tile=8, capacity=128
)


def f_dense(positions: torch.Tensor) -> torch.Tensor:
    cn = MODEL(STRUCTURE.replace(positions=positions))
    return (cn * WEIGHT).sum()


def f_sparse(positions: torch.Tensor, mode: str = "graph") -> torch.Tensor:
    structure = STRUCTURE.replace(positions=positions)
    cn = MODEL(structure, pairs=NBL, mode=mode)
    return (cn * WEIGHT).sum()


def test_value_matches_dense() -> None:
    """Sanity check before differentiating anything."""
    assert (
        pytest.approx(f_dense(POSITIONS).item(), abs=1e-12)
        == f_sparse(POSITIONS).item()
    )


@pytest.mark.parametrize("order,tolerance", [(1, 1e-12), (2, 1e-11)])
def test_jacrev_matches_dense_to_order(order: int, tolerance: float) -> None:
    """`jacrev` nested `order` times, sparse vs. dense."""
    dense, sparse = f_dense, f_sparse
    for _ in range(order):
        dense = jacrev(dense)
        sparse = jacrev(sparse)

    dense_value = dense(POSITIONS)
    sparse_value = sparse(POSITIONS)

    assert torch.isfinite(
        sparse_value
    ).all()  # pyright: ignore[reportArgumentType]
    assert (
        pytest.approx(dense_value.cpu(), abs=tolerance) == sparse_value.cpu()
    )  # pyright: ignore[reportAttributeAccessIssue]


def _third_derivative_contraction(
    f: Callable[[Tensor], Tensor],
) -> tuple[Tensor, Tensor, Tensor]:
    """The first three derivatives of `f` at `POSITIONS`, each contracted
    with `WEIGHT` before the next differentiation so that no full
    third-order tensor is built. Classic `autograd.grad` with
    `create_graph=True`."""
    p = POSITIONS.clone().requires_grad_(True)

    (first,) = torch.autograd.grad(f(p), p, create_graph=True)
    (second,) = torch.autograd.grad(
        (first * WEIGHT.unsqueeze(-1)).sum(), p, create_graph=True
    )
    (third,) = torch.autograd.grad((second * WEIGHT.unsqueeze(-1)).sum(), p)
    return first, second, third


def test_autograd_create_graph_three_deep_matches_dense() -> None:
    """Three nested derivatives through the padded list stay finite, are
    not identically zero (which would match a broken implementation just
    as well as a correct one), and agree with the dense path."""
    sparse = _third_derivative_contraction(f_sparse)
    dense = _third_derivative_contraction(f_dense)

    for sparse_order, dense_order in zip(sparse, dense):
        assert torch.isfinite(sparse_order).all()
        torch.testing.assert_close(
            sparse_order, dense_order, atol=1e-10, rtol=0
        )
    assert sparse[-1].abs().max() > 1e-6


def test_vmap_jacrev_matches_explicit_loop() -> None:
    """`vmap(jacrev(f_sparse))` over a batch of geometries must match an
    explicit Python loop calling `jacrev(f_sparse)` once per geometry."""
    batch = torch.stack([POSITIONS + 0.01 * k for k in range(4)])

    batched = vmap(jacrev(f_sparse))(batch)
    looped = torch.stack([jacrev(f_sparse)(p) for p in batch])

    assert pytest.approx(looped.cpu(), abs=1e-12) == batched.cpu()


########################################################################
# The "recompute" mode
#
# `mode="recompute"` checkpoints the pair list in chunks to bound backward
# memory. It supports first- and second-order derivatives, but cannot be
# composed with `vmap`.


def test_vmap_over_recompute_mode_raises() -> None:
    """`torch.utils.checkpoint` cannot be composed with `vmap`:
    `mode="recompute"` must fail loudly under `vmap`, not silently produce
    a wrong gradient."""
    batch = torch.stack([POSITIONS + 0.01 * k for k in range(3)])

    def f_recompute(positions: torch.Tensor) -> torch.Tensor:
        return f_sparse(positions, mode="recompute")

    # The message differs across torch versions: older ones fail inside
    # `_NoopSaveInputs`, newer ones reject saved tensor hooks up front.
    with pytest.raises(
        RuntimeError, match="_NoopSaveInputs|saved tensor hooks"
    ):
        vmap(jacrev(f_recompute))(batch)


@pytest.mark.grad
def test_gradcheck_recompute(monkeypatch: pytest.MonkeyPatch) -> None:
    """`dgradcheck` (numerical vs. analytical) through `mode="recompute"`,
    the checkpointed, chunked pair loop. The default chunk sizes are far
    larger than this fixture's ~120 real pairs, so the un-patched defaults
    would checkpoint the whole list as a single chunk and never touch the
    multi-chunk `index_add` accumulation; patching them down forces
    several checkpointed chunks to accumulate into one gradient, which is
    the part actually worth distrusting."""
    monkeypatch.setattr(common_module, "_CHUNK_SIZE_CPU", 16)
    monkeypatch.setattr(common_module, "_CHUNK_SIZE_GPU", 16)
    p = POSITIONS.clone().requires_grad_(True)

    def func(pos: torch.Tensor) -> torch.Tensor:
        return f_sparse(pos, mode="recompute")

    assert dgradcheck(func, p)


@pytest.mark.grad
def test_gradgradcheck_recompute(monkeypatch: pytest.MonkeyPatch) -> None:
    """`dgradgradcheck` (double backward) through `mode="recompute"`.
    `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`
    recomputes each chunk's forward pass under backward rather than
    saving its activations, and that recomputation builds a graph when
    invoked under `create_graph=True`, which is what a second backward
    needs."""
    monkeypatch.setattr(common_module, "_CHUNK_SIZE_CPU", 16)
    monkeypatch.setattr(common_module, "_CHUNK_SIZE_GPU", 16)
    p = POSITIONS.clone().requires_grad_(True)

    def func(pos: torch.Tensor) -> torch.Tensor:
        return f_sparse(pos, mode="recompute")

    assert dgradgradcheck(func, p)


########################################################################
# A list without pairs
#
# An isolated atom, a dissociated dimer, or a batch in which no system has
# a pair inside the cutoff gives a list with no slots at all. The
# coordination numbers are then zero, but they must still be part of the
# graph of the positions, with a zero gradient like the dense path's.

EMPTY_CUTOFF = 5.0
EMPTY_DD: DD = {"device": DEVICE, "dtype": torch.double}

empty_structures: dict[str, Structure] = {
    "dissociated dimer": hydrogens(
        torch.tensor([[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]], **EMPTY_DD)
    ),
    "single atom": hydrogens(torch.tensor([[0.0, 0.0, 0.0]], **EMPTY_DD)),
    "batch without pairs": hydrogens(
        torch.tensor(
            [
                [[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 30.0, 0.0]],
            ],
            **EMPTY_DD,
        )
    ),
}


@pytest.mark.parametrize("mode", ["graph", "recompute"])
@pytest.mark.parametrize("name", list(empty_structures))
def test_list_without_pairs_has_zero_gradient(
    name: str, mode: NeighborListMode
) -> None:
    """First and second derivatives exist and are zero, as on the dense
    path."""
    structure = empty_structures[name]
    nbl = build_neighborlist(structure, EMPTY_CUTOFF)
    assert nbl.idx_i.shape[0] == 0

    model = cn_d3.replace(cutoff=EMPTY_CUTOFF)
    positions = structure.positions.clone().requires_grad_(True)
    structure = structure.replace(positions=positions)

    cn = model(structure, pairs=nbl, mode=mode)
    (first,) = torch.autograd.grad((cn**2).sum(), positions, create_graph=True)
    (second,) = torch.autograd.grad(first.sum(), positions)

    dense = model(structure)
    (dense_first,) = torch.autograd.grad((dense**2).sum(), positions)

    assert (cn == 0).all()
    torch.testing.assert_close(first, dense_first, atol=0, rtol=0)
    assert (second == 0).all()


########################################################################
# GPU


@pytest.mark.cuda
def test_cpu_built_list_moved_to_cuda_matches_cuda_built() -> None:
    """A `NeighborList` built on CPU and moved with `.to(cuda)` must give
    the same coordination number as one built on CUDA directly."""
    model = CNModel(count=erf_count, cutoff=25.0)

    cpu = get_structure("mb16_43", "01", dtype=torch.double)
    cuda = cpu.to(torch.device("cuda"))

    nbl_moved = build_neighborlist(cpu, model.cutoff, tile=4).to(
        torch.device("cuda")
    )
    nbl_cuda = build_neighborlist(cuda, model.cutoff, tile=4)

    cn_moved = model(cuda, pairs=nbl_moved)
    cn_cuda = model(cuda, pairs=nbl_cuda)

    assert cn_moved.device.type == "cuda"
    assert torch.allclose(cn_moved, cn_cuda, atol=1e-12, rtol=0)
