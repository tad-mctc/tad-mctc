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
Transform conformance of `tad_mctc.ncoord`: the presets (`cn_d3`, `cn_d4`,
`cn_eeq`, `cn_eeq_en`, `cn_eeqbc`, `cn_eeqbc_en`, `cn_gfn2`), the counting
functions (`exp_count`, `erf_count`, `gfn2_count`) and
`cut_coordination_number`.

The presets run on a 3-atom molecule and on a padded batch of two molecules,
through three evaluation paths: dense (all pairs), `NeighborList` with
``mode="graph"`` and `NeighborList` with ``mode="recompute"``. The
neighbour lists are built outside of every transform. The coordination
numbers are checked with respect to the positions.

Checks: `vmap` without fallback, reverse mode up to third order, forward
mode, `torch.compile(fullgraph=True)` and finite derivatives up to third
order on the padded batch.

Exclusions:

- ``mode="recompute"`` under `vmap` and under forward mode (`jacfwd` is
  built on `vmap`): it wraps each chunk in
  `torch.utils.checkpoint.checkpoint`, which does not support `vmap` (see
  `CNModel.__call__`).
- Periodic structures: covered with finite differences in
  `test/test_ncoord/test_transforms.py` and `test_compile.py`; the
  conformance suite stays on molecules.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from torch.func import vmap

from tad_mctc.autograd import (
    dgradcheck,
    dgradgradcheck,
    dgradgradgradcheck,
    jacfwd_matches_jacrev,
    no_vmap_fallback,
)
from tad_mctc.io.structure import Structure
from tad_mctc.ncoord import (
    cn_d3,
    cn_d4,
    cn_eeq,
    cn_eeq_en,
    cn_eeqbc,
    cn_eeqbc_en,
    cn_gfn2,
    erf_count,
    exp_count,
    gfn2_count,
)
from tad_mctc.ncoord.common import CNModel, cut_coordination_number
from tad_mctc.neighbor.list import NeighborList, build_neighborlist
from tad_mctc.tree import stack

from ..conftest import DEVICE
from ..utils import (
    DYNAMO_SUPPORTED,
    DYNAMO_UNSUPPORTED_REASON,
    compile_fullgraph,
)

DD = {"device": DEVICE, "dtype": torch.float64}

# nondeterministic `index_add` on CUDA, see `test_ncoord/_paths.py`
NONDET_TOL = 1e-12

Evaluation = Callable[[torch.Tensor], torch.Tensor]

PRESETS = [
    pytest.param(cn_d3, id="cn_d3"),
    pytest.param(cn_d4, id="cn_d4"),
    pytest.param(cn_eeq, id="cn_eeq"),
    pytest.param(cn_eeq_en, id="cn_eeq_en"),
    pytest.param(cn_eeqbc, id="cn_eeqbc"),
    pytest.param(cn_eeqbc_en, id="cn_eeqbc_en"),
    pytest.param(cn_gfn2, id="cn_gfn2"),
]

PATHS = ["dense", "graph", "recompute"]
# the paths that support `vmap`, which `jacfwd` is built on
VMAP_PATHS = ["dense", "graph"]


# -- systems --------------------------------------------------------------


def _molecule() -> Structure:
    return Structure(
        numbers=torch.tensor([8, 1, 1], device=DEVICE),
        positions=torch.tensor(
            [[0.0, 0.0, 0.2], [0.0, 1.4, -1.0], [0.0, -1.4, -1.0]], **DD  # type: ignore[arg-type]
        ),
    )


def _padded_batch() -> Structure:
    """Water and H2, the second padded by one atom (``numbers == 0``)."""
    return Structure(
        numbers=torch.tensor([[8, 1, 1], [1, 1, 0]], device=DEVICE),
        positions=torch.tensor(
            [
                [[0.0, 0.0, 0.2], [0.0, 1.4, -1.0], [0.0, -1.4, -1.0]],
                [[0.1, 0.0, 0.0], [0.0, 0.0, 1.8], [0.0, 0.0, 0.0]],
            ],
            **DD,  # type: ignore[arg-type]
        ),
    )


def _systems() -> list[Structure]:
    """The padded batch as two single systems of equal ``nat``."""
    batch = _padded_batch()
    return [
        Structure(numbers=batch.numbers[i], positions=batch.positions[i])
        for i in range(2)
    ]


def _list(model: CNModel, structure: Structure) -> NeighborList:
    return build_neighborlist(structure, model.cutoff, capacity=64)


def _evaluation(model: CNModel, structure: Structure, path: str) -> Evaluation:
    """``positions -> cn`` through one evaluation path."""
    if path == "dense":

        def dense(positions: torch.Tensor) -> torch.Tensor:
            return model(structure.replace(positions=positions))

        return dense

    nbl = _list(model, structure)

    def sparse(positions: torch.Tensor) -> torch.Tensor:
        return model(structure.replace(positions=positions), nbl, mode=path)  # type: ignore[arg-type]

    return sparse


# -- vmap -----------------------------------------------------------------


@pytest.mark.parametrize("model", PRESETS)
def test_vmap_dense_matches_loop(model: CNModel) -> None:
    systems = _systems()
    numbers = torch.stack([s.numbers for s in systems])
    positions = torch.stack([s.positions for s in systems])

    def cn(n: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return model(Structure(numbers=n, positions=p))

    with no_vmap_fallback():
        batched = vmap(cn)(numbers, positions)

    looped = torch.stack([cn(n, p) for n, p in zip(numbers, positions)])
    assert torch.allclose(batched, looped, atol=1e-12)
    assert torch.allclose(batched, model(_padded_batch()), atol=1e-12)


@pytest.mark.parametrize("model", PRESETS)
def test_vmap_graph_matches_loop(model: CNModel) -> None:
    systems = _systems()
    nbls = [_list(model, s) for s in systems]

    def cn(s: Structure, nbl: NeighborList) -> torch.Tensor:
        return model(s, nbl, mode="graph")

    with no_vmap_fallback():
        batched = vmap(cn)(stack(systems), stack(nbls))

    looped = torch.stack([cn(s, n) for s, n in zip(systems, nbls)])
    assert torch.allclose(batched, looped, atol=1e-12)


# -- reverse mode up to third order ---------------------------------------


@pytest.mark.grad
@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("model", PRESETS)
def test_gradcheck_orders(model: CNModel, path: str) -> None:
    f = _evaluation(model, _molecule(), path)

    def positions() -> torch.Tensor:
        return _molecule().positions.requires_grad_()

    assert dgradcheck(f, positions(), nondet_tol=NONDET_TOL)
    assert dgradgradcheck(f, positions(), nondet_tol=NONDET_TOL)
    assert dgradgradgradcheck(f, positions(), nondet_tol=NONDET_TOL)


# -- forward mode ---------------------------------------------------------


@pytest.mark.parametrize("path", VMAP_PATHS)
@pytest.mark.parametrize("model", PRESETS)
def test_forward_matches_reverse(model: CNModel, path: str) -> None:
    f = _evaluation(model, _molecule(), path)

    assert jacfwd_matches_jacrev(f, _molecule().positions)


# -- compile --------------------------------------------------------------


@pytest.mark.filterwarnings(
    "ignore:remat_using_tags_for_fwd_loss_bwd_graph:UserWarning"
)
@pytest.mark.skipif(not DYNAMO_SUPPORTED, reason=DYNAMO_UNSUPPORTED_REASON)
@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("model", PRESETS)
def test_compile_matches_eager(model: CNModel, path: str) -> None:
    f = _evaluation(model, _molecule(), path)
    positions = _molecule().positions

    # the closures share one code object: start from a clean cache, or
    # Dynamo's recompilation limit is reached
    torch._dynamo.reset()  # pylint: disable=protected-access
    compiled = compile_fullgraph(f)
    assert torch.allclose(compiled(positions), f(positions), atol=1e-12)


# -- padding --------------------------------------------------------------


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("model", PRESETS)
def test_derivatives_finite_with_padding(model: CNModel, path: str) -> None:
    structure = _padded_batch()
    f = _evaluation(model, structure, path)

    positions = structure.positions.clone().requires_grad_()
    out = f(positions).sum()
    for _ in range(3):
        (grad,) = torch.autograd.grad(out, positions, create_graph=True)
        assert torch.isfinite(grad).all()
        out = grad.sum()


# -- counting functions and the cap ---------------------------------------


def _r() -> torch.Tensor:
    return torch.tensor([0.5, 1.4, 2.0, 3.5, 6.0], **DD)  # type: ignore[arg-type]


def _r0() -> torch.Tensor:
    return torch.tensor([1.1, 1.8, 1.8, 2.4, 3.0], **DD)  # type: ignore[arg-type]


def _exp(r: torch.Tensor) -> torch.Tensor:
    return exp_count(r, _r0())


def _erf(r: torch.Tensor) -> torch.Tensor:
    return erf_count(r, _r0())


def _gfn2(r: torch.Tensor) -> torch.Tensor:
    return gfn2_count(r, _r0())


def _cut(cn: torch.Tensor) -> torch.Tensor:
    return cut_coordination_number(cn, 4.0)


COUNTING = [
    pytest.param(_exp, id="exp_count"),
    pytest.param(_erf, id="erf_count"),
    pytest.param(_gfn2, id="gfn2_count"),
    pytest.param(_cut, id="cut_coordination_number"),
]


@pytest.mark.parametrize("f", COUNTING)
def test_counting_vmap_matches_loop(f: Evaluation) -> None:
    batch = torch.stack([_r(), _r() * 1.1])

    with no_vmap_fallback():
        batched = vmap(f)(batch)

    looped = torch.stack([f(x) for x in batch])
    assert torch.allclose(batched, looped)


@pytest.mark.grad
@pytest.mark.parametrize("f", COUNTING)
def test_counting_gradcheck_orders(f: Evaluation) -> None:
    assert dgradcheck(f, _r().requires_grad_())
    assert dgradgradcheck(f, _r().requires_grad_())
    assert dgradgradgradcheck(f, _r().requires_grad_())


@pytest.mark.parametrize("f", COUNTING)
def test_counting_forward_matches_reverse(f: Evaluation) -> None:
    assert jacfwd_matches_jacrev(f, _r())


@pytest.mark.skipif(not DYNAMO_SUPPORTED, reason=DYNAMO_UNSUPPORTED_REASON)
@pytest.mark.parametrize("f", COUNTING)
def test_counting_compile_matches_eager(f: Evaluation) -> None:
    torch._dynamo.reset()  # pylint: disable=protected-access
    compiled = compile_fullgraph(f)
    assert torch.allclose(compiled(_r()), f(_r()))
