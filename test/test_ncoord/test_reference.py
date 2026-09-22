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
variant and every `refs` entry -- molecules and periodic cells alike,
through the model's plain `__call__`, which picks the periodic path from
`structure.lattice` on its own.
"""

from __future__ import annotations

import pytest
import torch

from tad_mctc.batch import pack
from tad_mctc.ncoord import cn_eeq
from tad_mctc.typing import DD, Tensor

from ._variants import VARIANTS
from ..conftest import DEVICE
from ..utils import load_batch, load_structure
from .samples import BATCH_PAIRS, pair_id, refs


def _assert_close(
    ref: Tensor, cn: Tensor, dtype: torch.dtype, abs_tol: float | None
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
        assert torch.allclose(cn, ref, atol=1e-11, rtol=0)
    elif abs_tol is None:
        assert pytest.approx(ref.cpu()) == cn.cpu()
    else:
        assert pytest.approx(ref.cpu(), abs=abs_tol) == cn.cpu()


def _ref(source: tuple[str, str], key: str, dd: DD) -> Tensor:
    """`refs[source][key]`, moved to `dd`. `key` is a plain `str` (each
    `Variant.ref_key`), not one of `Refs`' literal keys, hence the ignore."""
    return refs[source][key].to(**dd)  # type: ignore[literal-required]


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("variant_name", list(VARIANTS))
@pytest.mark.parametrize("source", list(refs), ids=lambda s: s[1])
def test_single(
    variant_name: str, source: tuple[str, str], dtype: torch.dtype
) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}
    variant = VARIANTS[variant_name]

    structure = load_structure(*source, dd)
    ref = _ref(source, variant.ref_key, dd)

    cn = variant.call(structure)
    _assert_close(ref, cn, dtype, variant.abs_tol)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("variant_name", list(VARIANTS))
@pytest.mark.parametrize("pair", BATCH_PAIRS, ids=pair_id)
def test_batch(
    variant_name: str,
    pair: tuple[tuple[str, str], tuple[str, str]],
    dtype: torch.dtype,
) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}
    variant = VARIANTS[variant_name]

    structure = load_batch(pair, dd)
    ref = pack([_ref(source, variant.ref_key, dd) for source in pair])

    cn = variant.call(structure)
    _assert_close(ref, cn, dtype, variant.abs_tol)


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
