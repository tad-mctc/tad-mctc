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
`jacrev`, against mctc-lib's own Fortran derivative, for molecules and
periodic cells alike, over every evaluation path that accepts the sample
(see `_paths.py`); and `vmap(jacrev)` over a batch against finite differences,
on the dense path. The Jacobian of a batch on every path is checked by
`test_gradcheck_batch` in `test_autodiff.py`.
"""

from __future__ import annotations

import pytest
import torch
from torch.func import jacrev, vmap

from tad_mctc.autograd import numgrad
from tad_mctc.convert import tensor_to_numpy
from tad_mctc.io.structure import Structure
from tad_mctc.typing import DD, Tensor

from ...conftest import DEVICE
from ...utils import load_batch, load_structure
from .._paths import Bind, bind_dense, bind_precomputed, bind_sparse
from .._variants import VARIANTS
from ..samples import (
    LARGE_CRYSTALS,
    MOLECULE_PAIRS,
    MOLECULE_REFS,
    SMALL_CELL_REFS,
    pair_id,
    refs,
    source_id,
)


@pytest.mark.parametrize("variant_name", list(VARIANTS))
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("pair", MOLECULE_PAIRS, ids=pair_id)
def test_batch_vmap(
    variant_name: str,
    dtype: torch.dtype,
    pair: tuple[tuple[str, str], tuple[str, str]],
) -> None:
    """`jacrev` vmapped over the batch, for molecules only:
    for a periodic batch, `__call__` builds its shift table inside the
    vmap, which is data-dependent. The vmap route for periodic structures
    is a precomputed shift table passed as `pairs` (see
    `test_dense_periodic.py`)."""
    dd: DD = {"device": DEVICE, "dtype": dtype}
    tol = torch.finfo(dtype).eps ** 0.5 * 50
    variant = VARIANTS[variant_name]

    structure = load_batch(pair, dd)

    # numerical gradient as ref
    numdr = numgrad(variant.call, structure)

    def wrapper(numbers: Tensor, pos: Tensor) -> Tensor:
        return variant.call(Structure(numbers=numbers, positions=pos))

    pos = structure.positions.detach().clone().requires_grad_(True)
    jac: Tensor = vmap(jacrev(wrapper, argnums=1))(structure.numbers, pos)
    assert pytest.approx(numdr.cpu(), abs=tol) == tensor_to_numpy(jac)


def _reference_atol(dtype: torch.dtype) -> float:
    """
    Absolute tolerance of a Jacobian against mctc-lib's derivative.

    Measured over every reference sample, variant and evaluation path, on
    CPU and CUDA: the largest deviation is ~2e-14 in `float64` and ~7e-6
    in `float32`, for Jacobian entries up to ~0.5. Unlike the
    finite-difference checks above, this reference is exact, so the
    tolerance stays close to those values; `rtol=0` because most entries
    are near zero.
    """
    return 1e-11 if dtype == torch.double else 2e-5


# Every reference sample on CPU. On GPU only two small ones: the dense
# periodic Jacobian of the larger cells takes up to ~2 GB each, and the
# device handling is the same for every sample. The large crystals skip
# the all-pairs paths (see `LARGE_CRYSTALS`).
if DEVICE is None:
    _MOLECULES = MOLECULE_REFS
    _SMALL_CELLS = SMALL_CELL_REFS
    _LARGE_CELLS = LARGE_CRYSTALS
else:
    _MOLECULES = [("mb16_43", "SiH4")]
    _SMALL_CELLS = [("other", "periodic_triclinic")]
    _LARGE_CELLS = []


def _check_matches_fortran_reference(
    bind: Bind, source: tuple[str, str], variant_name: str, dtype: torch.dtype
) -> None:
    """`jacrev` against mctc-lib's own Fortran-computed derivative -- for
    periodic cells this includes the `shift @ lattice` term."""
    dd: DD = {"device": DEVICE, "dtype": dtype}
    variant = VARIANTS[variant_name]

    structure = load_structure(*source, dd)
    ref = refs[source][variant.dref_key].to(**dd)  # type: ignore[literal-required]
    cn = bind(variant.call, structure)

    def wrapper(pos: Tensor) -> Tensor:
        return cn(structure.replace(positions=pos))

    pos = structure.positions.detach().clone().requires_grad_(True)
    jac: Tensor = jacrev(wrapper)(pos)
    atol = _reference_atol(dtype)
    torch.testing.assert_close(jac, ref, atol=atol, rtol=0)


@pytest.mark.parametrize("variant_name", list(VARIANTS))
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("source", _MOLECULES + _SMALL_CELLS, ids=source_id)
def test_matches_fortran_reference_dense(
    variant_name: str, dtype: torch.dtype, source: tuple[str, str]
) -> None:
    _check_matches_fortran_reference(bind_dense, source, variant_name, dtype)


@pytest.mark.parametrize("variant_name", list(VARIANTS))
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("source", _SMALL_CELLS, ids=source_id)
def test_matches_fortran_reference_precomputed(
    variant_name: str, dtype: torch.dtype, source: tuple[str, str]
) -> None:
    _check_matches_fortran_reference(
        bind_precomputed, source, variant_name, dtype
    )


@pytest.mark.parametrize("variant_name", list(VARIANTS))
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize(
    "source", _MOLECULES + _SMALL_CELLS + _LARGE_CELLS, ids=source_id
)
def test_matches_fortran_reference_sparse(
    variant_name: str, dtype: torch.dtype, source: tuple[str, str]
) -> None:
    _check_matches_fortran_reference(bind_sparse, source, variant_name, dtype)
