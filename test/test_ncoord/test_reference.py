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
Coordination numbers against the mctc-lib Fortran reference, for every
variant and every reference sample -- molecules and periodic cells alike
-- through every evaluation path that accepts the sample (see
`_paths.py`): the dense path and the neighbour list take both, the
precomputed periodic shifts only cells.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tad_mctc.batch import pack
from tad_mctc.data.structures import get_structure
from tad_mctc.ncoord import cn_eeq
from tad_mctc.typing import DD, Tensor

from ..conftest import DEVICE
from ..utils import load_batch, load_structure
from ._paths import (
    SPARSE_FLOAT_ABS_TOL,
    Bind,
    bind_dense,
    bind_precomputed,
    bind_precomputed_batch,
    bind_sparse,
)
from ._variants import VARIANTS
from .samples import (
    CELL_PAIRS,
    CELL_REFS,
    MOLECULE_PAIRS,
    MOLECULE_REFS,
    pair_id,
    refs,
    source_id,
)

ATOL_DOUBLE = 1e-11
"""Double-precision tolerance for every sample but `CUTOFF_SOURCE`."""

CUTOFF_SOURCE = ("other", "periodic_one_atom")
"""A 5 Bohr cubic cell with the default 25 Bohr cutoff: 30 images sit at
exactly the cutoff, each adding ~2.5e-10 to the CN, and `distance <= cutoff`
flips on a last-bit difference. CI saw this sample off by 5e-10 once
(py313, torch 2.6.0 and 2.7.1) and could not be reproduced locally."""

ATOL_CUTOFF_SOURCE = 1e-8
"""Covers a few images flipping (30 x 2.5e-10 = 7.5e-9 at most)."""


def _assert_close(
    ref: Tensor,
    cn: Tensor,
    dtype: torch.dtype,
    abs_tol: float | None,
    atol: float = ATOL_DOUBLE,
) -> None:
    """
    In double precision, every sample agrees with Fortran to ~1e-14, so
    compare with `rtol=0` and a tight `atol`: a relative tolerance hides
    ~1e-5 absolute errors on CNs of order 1, which once masked a reference
    generated from an unwrapped periodic cell (``x23/acetic``).

    In single precision, the default `pytest.approx` (1e-6 relative)
    suffices for d3/d4/eeq. `gfn2` multiplies two sigmoids and `eeqbc`
    takes an extra `r0**norm_exp`, so they (and the EN-weighted variants)
    compound more roundoff, up to ~1e-5, and pass `abs_tol` instead.
    """
    if dtype == torch.double:
        # `assert_close` over `allclose`: on failure it reports the max
        # abs/rel deviation and the offending index instead of a bare
        # `False`, which is what a rare, tolerance-boundary mismatch
        # needs to be diagnosed rather than re-guessed at.
        torch.testing.assert_close(cn, ref, atol=atol, rtol=0)
    elif abs_tol is None:
        assert pytest.approx(ref.cpu()) == cn.cpu()
    else:
        assert pytest.approx(ref.cpu(), abs=abs_tol) == cn.cpu()


def _ref(source: tuple[str, str], key: str, dd: DD) -> Tensor:
    """`refs[source][key]`, moved to `dd`. `key` is a plain `str` (each
    `Variant.ref_key`), not one of `Refs`' literal keys, hence the ignore."""
    return refs[source][key].to(**dd)  # type: ignore[literal-required]


def _float_abs_tol(*tols: float | None) -> float | None:
    """The largest of the given `float32` tolerances, or `None` if none
    is set."""
    set_tols = [tol for tol in tols if tol is not None]
    return max(set_tols) if set_tols else None


def test_every_reference_file_is_listed() -> None:
    """Each reference file is a sample of exactly one list, and the lists
    hold what their names say."""
    references = Path(__file__).resolve().parents[1] / "references"
    files = {
        (path.parent.name, path.stem) for path in references.glob("*/*.json")
    }
    assert files == set(MOLECULE_REFS) | set(CELL_REFS)

    for source in MOLECULE_REFS:
        assert get_structure(*source).lattice is None, source
    for source in CELL_REFS:
        assert get_structure(*source).lattice is not None, source


########################################################################
# Single structures


def _check_single(
    bind: Bind,
    source: tuple[str, str],
    variant_name: str,
    dtype: torch.dtype,
    path_float_abs_tol: float | None = None,
) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}
    variant = VARIANTS[variant_name]

    structure = load_structure(*source, dd)
    ref = _ref(source, variant.ref_key, dd)

    cn = bind(variant.call, structure)(structure)
    atol = ATOL_CUTOFF_SOURCE if source == CUTOFF_SOURCE else ATOL_DOUBLE
    abs_tol = _float_abs_tol(variant.abs_tol, path_float_abs_tol)
    _assert_close(ref, cn, dtype, abs_tol, atol)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("variant_name", list(VARIANTS))
@pytest.mark.parametrize("source", MOLECULE_REFS + CELL_REFS, ids=source_id)
def test_single_dense(
    variant_name: str, source: tuple[str, str], dtype: torch.dtype
) -> None:
    _check_single(bind_dense, source, variant_name, dtype)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("variant_name", list(VARIANTS))
@pytest.mark.parametrize("source", CELL_REFS, ids=source_id)
def test_single_precomputed(
    variant_name: str, source: tuple[str, str], dtype: torch.dtype
) -> None:
    _check_single(bind_precomputed, source, variant_name, dtype)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("variant_name", list(VARIANTS))
@pytest.mark.parametrize("source", MOLECULE_REFS + CELL_REFS, ids=source_id)
def test_single_sparse(
    variant_name: str, source: tuple[str, str], dtype: torch.dtype
) -> None:
    _check_single(
        bind_sparse, source, variant_name, dtype, SPARSE_FLOAT_ABS_TOL
    )


########################################################################
# Batches


def _check_batch(
    bind: Bind,
    pair: tuple[tuple[str, str], tuple[str, str]],
    variant_name: str,
    dtype: torch.dtype,
    path_float_abs_tol: float | None = None,
) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}
    variant = VARIANTS[variant_name]

    structure = load_batch(pair, dd)
    ref = pack([_ref(source, variant.ref_key, dd) for source in pair])

    cn = bind(variant.call, structure)(structure)
    atol = ATOL_CUTOFF_SOURCE if CUTOFF_SOURCE in pair else ATOL_DOUBLE
    abs_tol = _float_abs_tol(variant.abs_tol, path_float_abs_tol)
    _assert_close(ref, cn, dtype, abs_tol, atol)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("variant_name", list(VARIANTS))
@pytest.mark.parametrize("pair", MOLECULE_PAIRS + CELL_PAIRS, ids=pair_id)
def test_batch_dense(
    variant_name: str,
    pair: tuple[tuple[str, str], tuple[str, str]],
    dtype: torch.dtype,
) -> None:
    _check_batch(bind_dense, pair, variant_name, dtype)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("variant_name", list(VARIANTS))
@pytest.mark.parametrize("pair", CELL_PAIRS, ids=pair_id)
def test_batch_precomputed(
    variant_name: str,
    pair: tuple[tuple[str, str], tuple[str, str]],
    dtype: torch.dtype,
) -> None:
    _check_batch(bind_precomputed_batch, pair, variant_name, dtype)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("variant_name", list(VARIANTS))
@pytest.mark.parametrize("pair", MOLECULE_PAIRS + CELL_PAIRS, ids=pair_id)
def test_batch_sparse(
    variant_name: str,
    pair: tuple[tuple[str, str], tuple[str, str]],
    dtype: torch.dtype,
) -> None:
    _check_batch(bind_sparse, pair, variant_name, dtype, SPARSE_FLOAT_ABS_TOL)


########################################################################
# Coordination number cap


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("cn_max", [49, 51.0, torch.tensor(49)])
def test_single_cnmax(dtype: torch.dtype, cn_max: int | float | Tensor) -> None:
    """A cap far above every CN in the sample leaves it unchanged."""
    dd: DD = {"device": DEVICE, "dtype": dtype}
    source = ("mb16_43", "01")

    structure = load_structure(*source, dd)
    ref = _ref(source, "cn_eeq", dd)

    cn = cn_eeq.replace(cn_max=cn_max)(structure)
    _assert_close(ref, cn, dtype, abs_tol=1e-5)
