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
`torch.compile(fullgraph=True)` of the coordination number, in positions
and, for a periodic structure, in the lattice: no data-dependent shape or
control flow may reach the evaluation, since the periodic shifts or the
neighbour list are built ahead of the call (see `_paths.py`).

Compiling costs ~2 s per case, so only two models are compiled: `cn_d3`,
on every path that can be traced, and `cn_eeq`, on one, for the `cn_max`
cap. The cap is applied once after the pair sum, whichever path summed.
`vmap` and `jacrev` are in `test_transforms.py`.
"""

from __future__ import annotations

import pytest
import torch

from tad_mctc.io.structure import Structure
from tad_mctc.ncoord import cn_d3, cn_eeq
from tad_mctc.ncoord.common import CNModel
from tad_mctc.typing import DD, Tensor

from ..conftest import DEVICE
from ..utils import (
    DYNAMO_SUPPORTED,
    DYNAMO_UNSUPPORTED_REASON,
    compile_fullgraph,
    load_structure,
)
from ._paths import (
    Bind,
    bind_dense,
    bind_precomputed,
    bind_precomputed_batch,
    bind_sparse,
)
from .samples import bulk_and_slab

DD_DOUBLE: DD = {"device": DEVICE, "dtype": torch.double}


def _assert_compiles_like_eager(
    model: CNModel, bind: Bind, structure: Structure
) -> None:
    cn = bind(model, structure)

    # A fresh `Structure`, not `structure.replace`: Dynamo in older
    # PyTorch cannot trace `dataclasses.replace`.
    def f(positions: Tensor, lattice: Tensor | None = None) -> Tensor:
        return cn(
            Structure(
                numbers=structure.numbers,
                positions=positions,
                lattice=lattice,
                periodic=structure.periodic,
            )
        )

    args: tuple[Tensor, ...] = (structure.positions,)
    if structure.lattice is not None:
        args = (structure.positions, structure.lattice)

    torch._dynamo.reset()  # pylint: disable=protected-access
    compiled_value = compile_fullgraph(f)(*args)

    assert torch.allclose(compiled_value, f(*args), atol=1e-12, rtol=0)


# Every path that can be traced on each sample. The dense path builds the
# periodic shifts of a cell inside the call, so it takes only molecules.
@pytest.mark.skipif(not DYNAMO_SUPPORTED, reason=DYNAMO_UNSUPPORTED_REASON)
@pytest.mark.parametrize(
    "bind,source",
    [
        pytest.param(bind_dense, ("mb16_43", "SiH4"), id="dense-SiH4"),
        pytest.param(
            bind_precomputed,
            ("other", "periodic_triclinic"),
            id="precomputed-periodic_triclinic",
        ),
        pytest.param(bind_sparse, ("mb16_43", "SiH4"), id="sparse-SiH4"),
        pytest.param(
            bind_sparse,
            ("other", "periodic_triclinic"),
            id="sparse-periodic_triclinic",
        ),
    ],
)
def test_compiles_fullgraph(bind: Bind, source: tuple[str, str]) -> None:
    structure = load_structure(*source, DD_DOUBLE)
    _assert_compiles_like_eager(cn_d3, bind, structure)


# The paths that accept a batch of cells and can be traced.
@pytest.mark.skipif(not DYNAMO_SUPPORTED, reason=DYNAMO_UNSUPPORTED_REASON)
@pytest.mark.parametrize(
    "bind",
    [bind_precomputed_batch, bind_sparse],
    ids=["precomputed", "sparse"],
)
def test_compiles_fullgraph_bulk_and_slab(bind: Bind) -> None:
    _assert_compiles_like_eager(cn_d3, bind, bulk_and_slab(DD_DOUBLE))


@pytest.mark.skipif(not DYNAMO_SUPPORTED, reason=DYNAMO_UNSUPPORTED_REASON)
def test_compiles_fullgraph_with_cn_max() -> None:
    """The capped `cn_eeq` preset: `cn_max` is a plain float here, so the
    cap must trace without a data-dependent branch."""
    structure = load_structure("mb16_43", "SiH4", DD_DOUBLE)
    _assert_compiles_like_eager(cn_eeq, bind_dense, structure)
