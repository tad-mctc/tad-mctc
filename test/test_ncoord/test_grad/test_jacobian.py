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
Jacobian of every coordination-number variant w.r.t. positions, via
`jacrev`, against finite differences and against mctc-lib's own Fortran
derivative -- for molecules and periodic cells alike.
"""

from __future__ import annotations

import pytest
import torch

from tad_mctc.autograd import bjacrev, jacrev, numgrad
from tad_mctc.convert import tensor_to_numpy
from tad_mctc.io.structure import Structure
from tad_mctc.typing import DD, Tensor

from .._variants import VARIANTS
from ...conftest import DEVICE
from ...utils import load_batch, load_structure
from ..samples import BATCH_PAIRS, REPRESENTATIVES, is_periodic, pair_id, refs

@pytest.mark.parametrize("variant_name", list(VARIANTS))
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("source", REPRESENTATIVES, ids=lambda s: s[1])
def test_single(
    variant_name: str, dtype: torch.dtype, source: tuple[str, str]
) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}
    tol = torch.finfo(dtype).eps ** 0.5 * 50
    variant = VARIANTS[variant_name]

    structure = load_structure(*source, dd)

    # numerical gradient as ref
    numdr = numgrad(variant.call, structure)

    def wrapper(pos: Tensor) -> Tensor:
        return variant.call(structure.replace(positions=pos))

    pos = structure.positions.detach().clone().requires_grad_(True)
    jac: Tensor = jacrev(wrapper)(pos)
    assert pytest.approx(numdr.cpu(), abs=tol) == tensor_to_numpy(jac)


@pytest.mark.parametrize("variant_name", list(VARIANTS))
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("pair", BATCH_PAIRS, ids=pair_id)
def test_batch(
    variant_name: str,
    dtype: torch.dtype,
    pair: tuple[tuple[str, str], tuple[str, str]],
) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}
    tol = torch.finfo(dtype).eps ** 0.5 * 50
    variant = VARIANTS[variant_name]

    structure = load_batch(pair, dd)

    # numerical gradient as ref
    numdr = numgrad(variant.call, structure)

    def wrapper(pos: Tensor) -> Tensor:
        return variant.call(structure.replace(positions=pos))

    # Plain `jacrev` over the whole batch rather than `bjacrev`: vmapping
    # a periodic batch through `__call__` would hit its data-dependent
    # shift-table build. Structures in a batch do not interact, so only
    # the diagonal blocks of the (batch, nat, batch, nat, 3) Jacobian are
    # nonzero; keep those.
    pos = structure.positions.detach().clone().requires_grad_(True)
    full: Tensor = jacrev(wrapper)(pos)
    batch = torch.arange(pos.shape[0], device=pos.device)
    jac = full[batch, :, batch]
    assert pytest.approx(numdr.cpu(), abs=tol) == tensor_to_numpy(jac)

    off_diagonal = full.clone()
    off_diagonal[batch, :, batch] = 0.0
    assert (off_diagonal == 0).all()


@pytest.mark.parametrize("variant_name", list(VARIANTS))
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize(
    "pair",
    [pair for pair in BATCH_PAIRS if not is_periodic(pair[0])],
    ids=pair_id,
)
def test_batch_vmap(
    variant_name: str,
    dtype: torch.dtype,
    pair: tuple[tuple[str, str], tuple[str, str]],
) -> None:
    """`bjacrev`, i.e. `jacrev` vmapped over the batch, for molecules only:
    for a periodic batch, `__call__` builds its shift table inside the
    vmap, which is data-dependent. The vmap route for periodic structures
    is `with_precomputed_shifts` (see `test_precomputed_shifts.py`)."""
    dd: DD = {"device": DEVICE, "dtype": dtype}
    tol = torch.finfo(dtype).eps ** 0.5 * 50
    variant = VARIANTS[variant_name]

    structure = load_batch(pair, dd)

    # numerical gradient as ref
    numdr = numgrad(variant.call, structure)

    def wrapper(numbers: Tensor, pos: Tensor) -> Tensor:
        return variant.call(Structure(numbers=numbers, positions=pos))

    pos = structure.positions.detach().clone().requires_grad_(True)
    jac: Tensor = bjacrev(wrapper, argnums=1)(structure.numbers, pos)
    assert pytest.approx(numdr.cpu(), abs=tol) == tensor_to_numpy(jac)


@pytest.mark.parametrize("variant_name", list(VARIANTS))
@pytest.mark.parametrize("source", REPRESENTATIVES, ids=lambda s: s[1])
def test_matches_fortran_reference(
    variant_name: str, source: tuple[str, str]
) -> None:
    """`jacrev` against mctc-lib's own Fortran-computed derivative."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    tol = torch.finfo(torch.double).eps ** 0.5 * 50
    variant = VARIANTS[variant_name]

    structure = load_structure(*source, dd)
    ref = refs[source][variant.dref_key].to(**dd)  # type: ignore[literal-required]

    def wrapper(pos: Tensor) -> Tensor:
        return variant.call(structure.replace(positions=pos))

    pos = structure.positions.detach().clone().requires_grad_(True)
    jac: Tensor = jacrev(wrapper)(pos)
    assert pytest.approx(ref.cpu(), abs=tol) == tensor_to_numpy(jac)
