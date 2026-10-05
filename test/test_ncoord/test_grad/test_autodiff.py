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
Autograd `gradgradcheck` of every coordination-number variant w.r.t.
positions, and `gradcheck`/`gradgradcheck` of batches, for molecules and
periodic cells alike, through every evaluation path that accepts the
sample (see `_paths.py`). The first derivative of a single structure is
checked against mctc-lib's own derivative instead, which is exact (see
`test_jacobian.py`).
"""

from __future__ import annotations

import pytest
import torch

from tad_mctc.autograd import (
    dgradcheck,
    dgradgradcheck,
    positions_gradchecker,
)
from tad_mctc.io.structure import Structure
from tad_mctc.typing import DD, Callable, Tensor

from ...conftest import DEVICE
from ...utils import load_batch, load_structure
from .._paths import (
    SPARSE_NONDET_TOL,
    Bind,
    bind_dense,
    bind_precomputed,
    bind_sparse,
)
from .._variants import VARIANTS
from ..samples import (
    CELL_PAIRS,
    MOLECULE_PAIRS,
    REPRESENTATIVE_CELLS,
    REPRESENTATIVE_MOLECULES,
    pair_id,
    source_id,
)

tol = 1e-8

DD_DOUBLE: DD = {"device": DEVICE, "dtype": torch.double}


def gradchecker(
    bind: Bind, structure: Structure, variant_name: str
) -> tuple[Callable[[Tensor], Tensor], Tensor]:
    """The CN on the path `bind` sets up, as a function of positions
    alone, and the positions to differentiate it at."""
    cn_function = bind(VARIANTS[variant_name].call, structure)
    return positions_gradchecker(cn_function, structure)


########################################################################
# Second derivative of single structures


def _check_gradgrad(
    bind: Bind, source: tuple[str, str], variant_name: str, nondet_tol: float
) -> None:
    structure = load_structure(*source, DD_DOUBLE)
    func, diffvars = gradchecker(bind, structure, variant_name)
    assert dgradgradcheck(func, diffvars, atol=tol, nondet_tol=nondet_tol)


@pytest.mark.grad
@pytest.mark.parametrize(
    "source", REPRESENTATIVE_MOLECULES + REPRESENTATIVE_CELLS, ids=source_id
)
@pytest.mark.parametrize("variant_name", list(VARIANTS))
def test_gradgradcheck_dense(
    source: tuple[str, str], variant_name: str
) -> None:
    _check_gradgrad(bind_dense, source, variant_name, nondet_tol=0.0)


@pytest.mark.grad
@pytest.mark.parametrize("source", REPRESENTATIVE_CELLS, ids=source_id)
@pytest.mark.parametrize("variant_name", list(VARIANTS))
def test_gradgradcheck_precomputed(
    source: tuple[str, str], variant_name: str
) -> None:
    _check_gradgrad(bind_precomputed, source, variant_name, nondet_tol=0.0)


@pytest.mark.grad
@pytest.mark.parametrize(
    "source", REPRESENTATIVE_MOLECULES + REPRESENTATIVE_CELLS, ids=source_id
)
@pytest.mark.parametrize("variant_name", list(VARIANTS))
def test_gradgradcheck_sparse(
    source: tuple[str, str], variant_name: str
) -> None:
    _check_gradgrad(bind_sparse, source, variant_name, SPARSE_NONDET_TOL)


########################################################################
# First and second derivatives of batches


def _check_batch_grad(
    bind: Bind,
    pair: tuple[tuple[str, str], tuple[str, str]],
    variant_name: str,
    nondet_tol: float,
) -> None:
    """The whole Jacobian of a batch, so this also checks that the
    structures of a batch do not affect each other."""
    structure = load_batch(pair, DD_DOUBLE)
    func, diffvars = gradchecker(bind, structure, variant_name)
    assert dgradcheck(func, diffvars, atol=tol, nondet_tol=nondet_tol)


def _check_batch_gradgrad(
    bind: Bind,
    pair: tuple[tuple[str, str], tuple[str, str]],
    variant_name: str,
    nondet_tol: float,
) -> None:
    structure = load_batch(pair, DD_DOUBLE)
    func, diffvars = gradchecker(bind, structure, variant_name)
    assert dgradgradcheck(func, diffvars, atol=tol, nondet_tol=nondet_tol)


@pytest.mark.grad
@pytest.mark.parametrize("pair", MOLECULE_PAIRS + CELL_PAIRS, ids=pair_id)
@pytest.mark.parametrize("variant_name", list(VARIANTS))
def test_gradcheck_batch_dense(
    pair: tuple[tuple[str, str], tuple[str, str]], variant_name: str
) -> None:
    _check_batch_grad(bind_dense, pair, variant_name, nondet_tol=0.0)


@pytest.mark.grad
@pytest.mark.parametrize("pair", CELL_PAIRS, ids=pair_id)
@pytest.mark.parametrize("variant_name", list(VARIANTS))
def test_gradcheck_batch_precomputed(
    pair: tuple[tuple[str, str], tuple[str, str]], variant_name: str
) -> None:
    _check_batch_grad(bind_precomputed, pair, variant_name, nondet_tol=0.0)


@pytest.mark.grad
@pytest.mark.parametrize("pair", MOLECULE_PAIRS + CELL_PAIRS, ids=pair_id)
@pytest.mark.parametrize("variant_name", list(VARIANTS))
def test_gradcheck_batch_sparse(
    pair: tuple[tuple[str, str], tuple[str, str]], variant_name: str
) -> None:
    _check_batch_grad(bind_sparse, pair, variant_name, SPARSE_NONDET_TOL)


@pytest.mark.grad
@pytest.mark.parametrize("pair", MOLECULE_PAIRS + CELL_PAIRS, ids=pair_id)
@pytest.mark.parametrize("variant_name", list(VARIANTS))
def test_gradgradcheck_batch_dense(
    pair: tuple[tuple[str, str], tuple[str, str]], variant_name: str
) -> None:
    _check_batch_gradgrad(bind_dense, pair, variant_name, nondet_tol=0.0)


@pytest.mark.grad
@pytest.mark.parametrize("pair", CELL_PAIRS, ids=pair_id)
@pytest.mark.parametrize("variant_name", list(VARIANTS))
def test_gradgradcheck_batch_precomputed(
    pair: tuple[tuple[str, str], tuple[str, str]], variant_name: str
) -> None:
    _check_batch_gradgrad(bind_precomputed, pair, variant_name, nondet_tol=0.0)


@pytest.mark.grad
@pytest.mark.parametrize("pair", MOLECULE_PAIRS + CELL_PAIRS, ids=pair_id)
@pytest.mark.parametrize("variant_name", list(VARIANTS))
def test_gradgradcheck_batch_sparse(
    pair: tuple[tuple[str, str], tuple[str, str]], variant_name: str
) -> None:
    _check_batch_gradgrad(bind_sparse, pair, variant_name, SPARSE_NONDET_TOL)
