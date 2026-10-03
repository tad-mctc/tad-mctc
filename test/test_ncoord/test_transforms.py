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
`vmap` and `jacrev` of every coordination-number variant, over positions
and over the lattice, through every evaluation path that can be traced
(see `_paths.py`). The dense path cannot be traced on a periodic
structure, since it builds its periodic shifts inside the call, so cells
are only run through the `precomputed` and `sparse` paths. The case lists
below spell out which path runs on which sample. `torch.compile` is in
`test_compile.py`.

Each check compares against something that needs no reference: a plain
Python loop for `vmap`, finite differences for `jacrev`. Every check runs
on single structures, on batches, and on a bulk cell batched with a slab;
a check has one helper and one thin test per kind of structure. The
one-atom cell has thin tests of its own, for the variants that are not
identically zero on it.
"""

from __future__ import annotations

import pytest
import torch

from tad_mctc.autograd import jacrev_matches_finite_diff, vmap_matches_loop
from tad_mctc.io.structure import Structure
from tad_mctc.ncoord.common import CNModel
from tad_mctc.typing import DD, Tensor

from ..conftest import DEVICE
from ..utils import load_batch, load_structure
from ._paths import (
    Bind,
    bind_dense,
    bind_precomputed,
    bind_sparse,
)
from ._variants import VARIANTS
from .samples import bulk_and_slab

DD_DOUBLE: DD = {"device": DEVICE, "dtype": torch.double}

SINGLE_CASES = [
    pytest.param(bind_dense, ("mb16_43", "SiH4"), id="dense-SiH4"),
    pytest.param(bind_dense, ("mb16_43", "01"), id="dense-01"),
    pytest.param(
        bind_precomputed,
        ("other", "periodic_triclinic"),
        id="precomputed-periodic_triclinic",
    ),
    pytest.param(bind_sparse, ("mb16_43", "SiH4"), id="sparse-SiH4"),
    pytest.param(bind_sparse, ("mb16_43", "01"), id="sparse-01"),
    pytest.param(
        bind_sparse,
        ("other", "periodic_triclinic"),
        id="sparse-periodic_triclinic",
    ),
]

CELL_CASES = [
    pytest.param(
        bind_precomputed,
        ("other", "periodic_triclinic"),
        id="precomputed-periodic_triclinic",
    ),
    pytest.param(
        bind_sparse,
        ("other", "periodic_triclinic"),
        id="sparse-periodic_triclinic",
    ),
]

# A cell of a single atom, which interacts only with its own images. The
# EN-weighted variants weigh a pair by the difference of the two atoms'
# electronegativities, which is zero for an atom and its own image, so
# their CN is exactly zero there and every check would pass trivially.
# They are left out for this cell; `CELL_CASES` covers them.
ONE_ATOM_CELL_CASES = [
    pytest.param(
        bind_precomputed,
        ("other", "periodic_one_atom"),
        id="precomputed-periodic_one_atom",
    ),
    pytest.param(
        bind_sparse,
        ("other", "periodic_one_atom"),
        id="sparse-periodic_one_atom",
    ),
]
ONE_ATOM_VARIANTS = ["cn_d3", "cn_d4", "cn_gfn2", "cn_eeq", "cn_eeqbc"]

MOLECULE_PAIR = (("mb16_43", "01"), ("mb16_43", "SiH4"))
CELL_PAIR = (("other", "periodic_triclinic"), ("other", "periodic_one_atom"))

BATCH_CASES = [
    pytest.param(bind_dense, MOLECULE_PAIR, id="dense-01+SiH4"),
    pytest.param(
        bind_precomputed,
        CELL_PAIR,
        id="precomputed-periodic_triclinic+periodic_one_atom",
    ),
    pytest.param(bind_sparse, MOLECULE_PAIR, id="sparse-01+SiH4"),
    pytest.param(
        bind_sparse,
        CELL_PAIR,
        id="sparse-periodic_triclinic+periodic_one_atom",
    ),
]

CELL_BATCH_CASES = [
    pytest.param(
        bind_precomputed,
        CELL_PAIR,
        id="precomputed-periodic_triclinic+periodic_one_atom",
    ),
    pytest.param(
        bind_sparse,
        CELL_PAIR,
        id="sparse-periodic_triclinic+periodic_one_atom",
    ),
]

# The paths that accept a batch of cells and can be traced.
bulk_and_slab_paths = pytest.mark.parametrize(
    "bind",
    [bind_precomputed, bind_sparse],
    ids=["precomputed", "sparse"],
)


########################################################################
# The checks


def _check_vmap_over_positions(
    model: CNModel, bind: Bind, structure: Structure
) -> None:
    """`vmap` over a batch of `positions`, with everything else fixed,
    matches a plain Python loop."""
    cn = bind(model, structure)

    def f(positions: Tensor) -> Tensor:
        return cn(structure.replace(positions=positions))

    batch = torch.stack(
        [
            structure.positions,
            structure.positions + 0.01,
            structure.positions - 0.01,
        ]
    )
    assert vmap_matches_loop(f, batch)


def _check_vmap_over_lattices(
    model: CNModel, bind: Bind, structure: Structure
) -> None:
    """`vmap` over a batch of `lattice`s, with the periodic shifts or the
    neighbour list built once for the unscaled one, matches a plain Python
    loop. Scaling
    a lattice up only moves images out of the cutoff, so what was built
    for the smallest lattice covers the others."""
    assert structure.lattice is not None
    cn = bind(model, structure)

    def f(lattice: Tensor) -> Tensor:
        return cn(structure.replace(lattice=lattice))

    batch = torch.stack(
        [structure.lattice * scale for scale in (1.0, 1.01, 1.02)]
    )
    assert vmap_matches_loop(f, batch)


def _check_jacrev_wrt_lattice(
    model: CNModel, bind: Bind, structure: Structure
) -> None:
    """`jacrev` with respect to `lattice`, through the `shifts @ lattice`
    term, matches a finite-difference Jacobian."""
    assert structure.lattice is not None
    cn = bind(model, structure)

    def f(lattice: Tensor) -> Tensor:
        return cn(structure.replace(lattice=lattice))

    # Off the cell as loaded: the one-atom cell is 5 Bohr wide, so its
    # fifth image lies at exactly `cn_d3`'s 25 Bohr cutoff. There the hard
    # cutoff makes the CN a step, which finite differences see and the
    # derivative rightly ignores. Scaling up only removes images, so the
    # shifts or the list built for the loaded cell still cover it.
    lattice = 1.013 * structure.lattice
    assert jacrev_matches_finite_diff(f, lattice)


########################################################################
# vmap over positions


@pytest.mark.parametrize("variant_name", list(VARIANTS))
@pytest.mark.parametrize("bind,source", SINGLE_CASES)
def test_vmap_over_positions(
    variant_name: str, bind: Bind, source: tuple[str, str]
) -> None:
    structure = load_structure(*source, DD_DOUBLE)
    _check_vmap_over_positions(VARIANTS[variant_name].call, bind, structure)


@pytest.mark.parametrize("variant_name", ONE_ATOM_VARIANTS)
@pytest.mark.parametrize("bind,source", ONE_ATOM_CELL_CASES)
def test_vmap_over_positions_one_atom_cell(
    variant_name: str, bind: Bind, source: tuple[str, str]
) -> None:
    structure = load_structure(*source, DD_DOUBLE)
    _check_vmap_over_positions(VARIANTS[variant_name].call, bind, structure)


@pytest.mark.parametrize("variant_name", list(VARIANTS))
@pytest.mark.parametrize("bind,pair", BATCH_CASES)
def test_vmap_over_positions_batch(
    variant_name: str,
    bind: Bind,
    pair: tuple[tuple[str, str], tuple[str, str]],
) -> None:
    structure = load_batch(pair, DD_DOUBLE)
    _check_vmap_over_positions(VARIANTS[variant_name].call, bind, structure)


@pytest.mark.parametrize("variant_name", list(VARIANTS))
@bulk_and_slab_paths
def test_vmap_over_positions_bulk_and_slab(
    variant_name: str, bind: Bind
) -> None:
    structure = bulk_and_slab(DD_DOUBLE)
    _check_vmap_over_positions(VARIANTS[variant_name].call, bind, structure)


########################################################################
# vmap over lattices


@pytest.mark.parametrize("variant_name", list(VARIANTS))
@pytest.mark.parametrize("bind,source", CELL_CASES)
def test_vmap_over_lattices(
    variant_name: str, bind: Bind, source: tuple[str, str]
) -> None:
    structure = load_structure(*source, DD_DOUBLE)
    _check_vmap_over_lattices(VARIANTS[variant_name].call, bind, structure)


@pytest.mark.parametrize("variant_name", ONE_ATOM_VARIANTS)
@pytest.mark.parametrize("bind,source", ONE_ATOM_CELL_CASES)
def test_vmap_over_lattices_one_atom_cell(
    variant_name: str, bind: Bind, source: tuple[str, str]
) -> None:
    structure = load_structure(*source, DD_DOUBLE)
    _check_vmap_over_lattices(VARIANTS[variant_name].call, bind, structure)


@pytest.mark.parametrize("variant_name", list(VARIANTS))
@pytest.mark.parametrize("bind,pair", CELL_BATCH_CASES)
def test_vmap_over_lattices_batch(
    variant_name: str,
    bind: Bind,
    pair: tuple[tuple[str, str], tuple[str, str]],
) -> None:
    structure = load_batch(pair, DD_DOUBLE)
    _check_vmap_over_lattices(VARIANTS[variant_name].call, bind, structure)


@pytest.mark.parametrize("variant_name", list(VARIANTS))
@bulk_and_slab_paths
def test_vmap_over_lattices_bulk_and_slab(
    variant_name: str, bind: Bind
) -> None:
    structure = bulk_and_slab(DD_DOUBLE)
    _check_vmap_over_lattices(VARIANTS[variant_name].call, bind, structure)


########################################################################
# jacrev with respect to the lattice


@pytest.mark.parametrize("variant_name", list(VARIANTS))
@pytest.mark.parametrize("bind,source", CELL_CASES)
def test_jacrev_wrt_lattice(
    variant_name: str, bind: Bind, source: tuple[str, str]
) -> None:
    structure = load_structure(*source, DD_DOUBLE)
    _check_jacrev_wrt_lattice(VARIANTS[variant_name].call, bind, structure)


@pytest.mark.parametrize("variant_name", ONE_ATOM_VARIANTS)
@pytest.mark.parametrize("bind,source", ONE_ATOM_CELL_CASES)
def test_jacrev_wrt_lattice_one_atom_cell(
    variant_name: str, bind: Bind, source: tuple[str, str]
) -> None:
    structure = load_structure(*source, DD_DOUBLE)
    _check_jacrev_wrt_lattice(VARIANTS[variant_name].call, bind, structure)


@pytest.mark.parametrize("variant_name", list(VARIANTS))
@pytest.mark.parametrize("bind,pair", CELL_BATCH_CASES)
def test_jacrev_wrt_lattice_batch(
    variant_name: str,
    bind: Bind,
    pair: tuple[tuple[str, str], tuple[str, str]],
) -> None:
    structure = load_batch(pair, DD_DOUBLE)
    _check_jacrev_wrt_lattice(VARIANTS[variant_name].call, bind, structure)


@pytest.mark.parametrize("variant_name", list(VARIANTS))
@bulk_and_slab_paths
def test_jacrev_wrt_lattice_bulk_and_slab(
    variant_name: str, bind: Bind
) -> None:
    structure = bulk_and_slab(DD_DOUBLE)
    _check_jacrev_wrt_lattice(VARIANTS[variant_name].call, bind, structure)
