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
The evaluation paths of a `CNModel`, for the tests that run the same
checks over every path: the Fortran references (`test_reference.py`,
`test_grad/`), the cell geometry (`test_periodic_cells.py`) and the
transforms (`test_transforms.py`, `test_compile.py`).

An *evaluation path* is one way to compute the coordination number:

- ``dense``: ``model(structure)``, all pairs; for a periodic structure it
  builds its periodic shifts itself on every call.
- ``precomputed``: ``model(structure, pairs=shifts)``, all pairs over
  periodic shifts built ahead of time. Periodic structures only.
- ``sparse``: ``model(structure, pairs=nbl)``, over a pre-built
  neighbour list.

Crossed with the two geometries, the paths give the four *quadrants* the
path-specific tests are split into: `test_dense_molecular.py`,
`test_dense_periodic.py` (``dense`` and ``precomputed``),
`test_sparse_molecular.py` and `test_sparse_periodic.py`.

Each ``bind_*`` function builds what its path needs (periodic shifts or a
neighbour list) once, from the structure it is given, and returns the
evaluation alone. Tests then differentiate or transform only the
evaluation, the way callers of the library are meant to. The tests list
which path runs on which samples themselves: the dense path cannot be
traced (``vmap``, ``torch.compile``) on a periodic structure, since it
builds its shifts inside the call, and the precomputed path takes only
periodic structures.
"""

from __future__ import annotations

from tad_mctc.io.structure import Structure
from tad_mctc.ncoord.common import CNModel
from tad_mctc.neighbor.images import build_periodic_shifts
from tad_mctc.neighbor.list import build_neighborlist
from tad_mctc.typing import Callable, Tensor

Evaluation = Callable[[Structure], Tensor]
"""A coordination-number evaluation with everything data-dependent
already built, taking a `Structure` with the same atoms."""

Bind = Callable[[CNModel, Structure], Evaluation]
"""One evaluation path: builds what the path needs for a model and a
structure, and returns the evaluation."""

SPARSE_FLOAT_ABS_TOL = 5e-5
"""Lower bound on the absolute `float32` tolerance of the sparse path
against the references. The neighbour list is summed with a scattered
`index_add`, one `float32` addition per pair, where the dense paths
reduce pairwise along a row; over a periodic cell's hundreds of image
pairs per atom this rounds ~10x worse (up to ~1e-5, measured)."""

SPARSE_NONDET_TOL = 1e-12
"""How far repeated backward passes of the sparse path may differ
(`gradcheck`'s `nondet_tol`). On CUDA, `index_add` accumulates with
atomic adds in varying order, so repeated double-backward passes differ
in the last bits (below 1e-14 in `float64`, measured). The dense paths
are deterministic."""


def bind_dense(model: CNModel, structure: Structure) -> Evaluation:
    """The all-pairs path, which needs nothing built ahead."""
    return model


def bind_precomputed(model: CNModel, structure: Structure) -> Evaluation:
    """All pairs over periodic shifts built for the cell `structure`, or
    shared by its batch of cells, sized for the most demanding system."""
    assert structure.lattice is not None and structure.periodic is not None
    shifts = build_periodic_shifts(
        structure.lattice, structure.periodic, cutoff=model.cutoff
    )

    def evaluate(target: Structure) -> Tensor:
        return model(target, pairs=shifts)

    return evaluate


def bind_sparse(model: CNModel, structure: Structure) -> Evaluation:
    """The pairs of a neighbour list built for `structure`, single or
    batched."""
    nbl = build_neighborlist(structure, model.cutoff)

    def evaluate(target: Structure) -> Tensor:
        return model(target, pairs=nbl)

    return evaluate
