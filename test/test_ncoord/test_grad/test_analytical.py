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
Analytical `cn_d3_gradient` against two independent ground truths: finite
differences of `cn_d3` itself, and mctc-lib's Fortran-computed `dcn_d3dr`
reference (`samples.py`'s `refs`). `cn_d3` is the only counting-function
variant covered here because it is the only one with both an analytical
gradient (`cn_d3_gradient`) and a stored derivative reference -- the other
variants' jacobians are covered against finite differences only, in
`test_jacobian.py`, via `torch.func.jacrev`.
"""

from __future__ import annotations

import pytest
import torch

from tad_mctc.autograd import numgrad
from tad_mctc.batch import pack, zero_masked_pairs
from tad_mctc.data import radii
from tad_mctc.io.structure import Structure
from tad_mctc.ncoord import cn_d3, cn_d3_gradient
from tad_mctc.ncoord import defaults
from tad_mctc.typing import DD, CNFunc, CNGradFunction, Tensor

from ...conftest import DEVICE
from ...utils import load_sample
from ..samples import refs

# Molecules only: `cn_d3_gradient` has no lattice argument.
sample_list: list[tuple[str, str]] = [
    ("mb16_43", "SiH4"),
    ("heavy28", "pbh4_bih3"),
    ("mb16_43", "01"),
]


def _fortran_ref(source: tuple[str, str], nat: int, dd: DD) -> Tensor:
    """The Fortran `dcn_d3dr` reference, reshaped to `cn_d3_gradient`'s
    ``(nat, nat, 3)`` layout. Only `cn_d3` has a closed-form gradient, so
    this needs no `Variant` table."""
    return refs[source]["dcn_d3dr"].to(**dd).reshape(nat, nat, 3)


@pytest.mark.parametrize("function", [(cn_d3, cn_d3_gradient)])
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("collection,record", sample_list)
def test_single(
    function: tuple[CNFunc, CNGradFunction],
    dtype: torch.dtype,
    collection: str,
    record: str,
) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}
    tol = torch.finfo(dtype).eps ** 0.5 * 50

    numbers, positions = load_sample(collection, record, dd)
    structure = Structure(numbers=numbers, positions=positions)

    numdr = zero_masked_pairs(numbers, numgrad(function[0], structure))
    dcndr = function[1](numbers, positions)
    ref = zero_masked_pairs(
        numbers, _fortran_ref((collection, record), numbers.shape[-1], dd)
    )

    assert pytest.approx(dcndr.cpu(), abs=tol) == numdr.cpu()
    assert pytest.approx(dcndr.cpu(), abs=tol) == ref.cpu()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("collection,record", sample_list)
def test_rcov_table_convention(
    dtype: torch.dtype, collection: str, record: str
) -> None:
    """``cn_d3_gradient``'s ``rcov`` must accept the same per-element-table
    convention as ``cn_d3``/``CNModel.rcov`` (indexed by atomic number, e.g.
    ``radii.COV_D3(**dd)``), not a per-atom array. A *non-default* table
    (so this cannot pass merely because both sides fall back to the same
    default) passed straight through to ``cn_d3_gradient`` must reproduce
    the gradient of ``cn_d3``'s own forward pass built with that same table (via :meth:`CNModel.replace`) — i.e. `cn_d3_gradient`
    and `cn_d3`/`CNModel.rcov` must agree on what `rcov` means."""
    dd: DD = {"device": DEVICE, "dtype": dtype}
    tol = torch.finfo(dtype).eps ** 0.5 * 50

    numbers, positions = load_sample(collection, record, dd)
    structure = Structure(numbers=numbers, positions=positions)

    table = radii.COV_D3(**dd) * 0.9
    cn_d3_scaled = cn_d3.replace(rcov=table)

    dcndr = cn_d3_gradient(numbers, positions, rcov=table)
    numdr = zero_masked_pairs(numbers, numgrad(cn_d3_scaled, structure))

    assert pytest.approx(dcndr.cpu(), abs=tol) == numdr.cpu()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("collection,record", sample_list)
def test_explicit_cutoff(
    dtype: torch.dtype, collection: str, record: str
) -> None:
    """``cutoff`` passed explicitly must take the same code path as the
    default (which resolves ``None`` to ``defaults.CUTOFF_D3`` before
    doing anything else), so passing that same value through explicitly
    must reproduce the default-cutoff gradient exactly."""
    dd: DD = {"device": DEVICE, "dtype": dtype}

    numbers, positions = load_sample(collection, record, dd)

    dcndr_default = cn_d3_gradient(numbers, positions)
    dcndr_explicit = cn_d3_gradient(
        numbers, positions, cutoff=torch.tensor(defaults.CUTOFF_D3, **dd)
    )

    assert pytest.approx(dcndr_explicit.cpu()) == dcndr_default.cpu()


@pytest.mark.parametrize("function", [(cn_d3, cn_d3_gradient)])
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("collection1,record1", [("mb16_43", "SiH4")])
@pytest.mark.parametrize("collection2,record2", sample_list)
def test_batch(
    function: tuple[CNFunc, CNGradFunction],
    dtype: torch.dtype,
    collection1: str,
    record1: str,
    collection2: str,
    record2: str,
) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}
    tol = torch.finfo(dtype).eps ** 0.5 * 50

    numbers1, positions1 = load_sample(collection1, record1, dd)
    numbers2, positions2 = load_sample(collection2, record2, dd)
    numbers = pack((numbers1, numbers2))
    positions = pack((positions1, positions2))
    structure = Structure(numbers=numbers, positions=positions)

    numdr = zero_masked_pairs(numbers, numgrad(function[0], structure))
    dcndr = function[1](numbers, positions)
    ref = zero_masked_pairs(
        numbers,
        pack(
            (
                _fortran_ref((collection1, record1), numbers1.shape[-1], dd),
                _fortran_ref((collection2, record2), numbers2.shape[-1], dd),
            )
        ),
    )

    assert pytest.approx(dcndr.cpu(), abs=tol) == numdr.cpu()
    assert pytest.approx(dcndr.cpu(), abs=tol) == ref.cpu()
