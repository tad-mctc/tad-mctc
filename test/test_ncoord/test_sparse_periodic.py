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
The sparse periodic quadrant: `CNModel` with `pairs=nbl`, a list
built for a cell, whose pairs carry image shifts that `structure.lattice`
turns into translations at call time.

Covers agreement with the dense periodic path, for the reference cells
and for a batch mixing a bulk cell with a slab. Cell geometry that every
evaluation path must handle alike lives in `test_periodic_cells.py`.
Agreement with the Fortran references is checked for every evaluation path
in `test_reference.py` and `test_grad/`, and the transforms in
`test_transforms.py` and `test_compile.py`; the checks shared with
molecular lists live in `test_sparse.py`.
"""

from __future__ import annotations

import pytest
import torch

from tad_mctc.io.structure import pack_structures
from tad_mctc.ncoord import erf_count
from tad_mctc.ncoord.common import CNModel
from tad_mctc.neighbor.list import build_neighborlist
from tad_mctc.typing import DD

from ..conftest import DEVICE
from ..utils import load_structure
from ._variants import VARIANTS
from .samples import CELL_REFS, bulk_and_slab, source_id

########################################################################
# Agreement with the dense periodic path


@pytest.mark.parametrize("variant_name", list(VARIANTS))
@pytest.mark.parametrize("source", CELL_REFS, ids=source_id)
def test_matches_dense_periodic(
    variant_name: str, source: tuple[str, str]
) -> None:
    """Sparse and dense periodic paths agree on every periodic sample --
    catches a `pair_weight` or cutoff-masking bug specific to one path
    that the reference comparison might not exercise identically."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    model = VARIANTS[variant_name].call

    structure = load_structure(*source, dd)
    nbl = build_neighborlist(structure, model.cutoff)
    sparse = model(structure, pairs=nbl)

    assert torch.allclose(sparse, model(structure), atol=1e-11, rtol=0)


########################################################################
# Batched cells: one list, each pair translated by its own system's cell


def test_batch_matches_dense() -> None:
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    model = CNModel(count=erf_count, cutoff=25.0)
    batch = bulk_and_slab(dd)

    nbl = build_neighborlist(batch, model.cutoff)
    sparse = model(batch, pairs=nbl)

    assert torch.allclose(sparse, model(batch), atol=1e-11, rtol=0)


def test_batch_sharing_one_cell_matches_dense() -> None:
    """One `(1, 3, 3)` cell for every system of a batch is shared, not
    indexed per system."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    cell = load_structure("other", "periodic_cubic", dd)
    batch = pack_structures([cell, cell, cell])
    assert cell.lattice is not None and cell.periodic is not None
    shared = batch.replace(
        lattice=cell.lattice.unsqueeze(0), periodic=cell.periodic
    )
    model = CNModel(count=erf_count, cutoff=25.0)

    nbl = build_neighborlist(shared, model.cutoff)
    sparse = model(shared, pairs=nbl)

    assert torch.allclose(sparse, model(shared), atol=1e-11, rtol=0)
