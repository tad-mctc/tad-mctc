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
The dense molecular quadrant: `CNModel.__call__` on a `Structure` without
a lattice, summing over all atom pairs.

Agreement with the Fortran references, for single molecules and padded
batches, is checked for every evaluation path in `test_reference.py` and
`test_grad/`, and the transforms in `test_transforms.py` and
`test_compile.py`. This file covers the accuracy in `float32`, the cutoff
mask and the padding of batched molecules.
"""

from __future__ import annotations

import pytest
import torch
from torch.func import jacrev

from tad_mctc.batch import pack
from tad_mctc.data import radii
from tad_mctc.data.structures import get_structure
from tad_mctc.io.structure import Structure
from tad_mctc.ncoord import cn_d3, cn_eeq_en
from tad_mctc.ncoord.common import CNModel
from tad_mctc.ncoord.count import erf_count, exp_count, gfn2_count
from tad_mctc.typing import CountingFunction

from ..conftest import DEVICE

########################################################################
# Accuracy


@pytest.mark.parametrize("model", [cn_d3, cn_eeq_en], ids=["d3", "eeq_en"])
def test_float32_matches_float64(model: CNModel) -> None:
    """The dense path takes distances from position differences, which stay
    accurate in `float32`. The expansion `|x|^2 + |y|^2 - 2 x.y` cancels
    large terms and is about ten times worse on this 210-atom peptide, so
    this bound would fail with it."""
    sample = get_structure("glu_ala", "0008").to(DEVICE)

    reference = model(sample.type(torch.double))
    single = model(sample.type(torch.float))

    assert torch.allclose(single.double(), reference, atol=2e-6, rtol=0)


########################################################################
# Cutoff


@pytest.mark.parametrize("cfunc", [erf_count, exp_count, gfn2_count])
def test_cutoff_excludes_pair(cfunc: CountingFunction) -> None:
    """A cutoff shorter than the interatomic distance excludes the pair
    entirely, giving CN == 0."""
    numbers = torch.tensor([1, 1], dtype=torch.long)
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]], dtype=torch.float64
    )

    tight = CNModel(count=cfunc, cutoff=50.0)
    cn_tight = tight(Structure(numbers=numbers, positions=positions))
    assert torch.all(cn_tight == 0.0)


def test_cutoff_excludes_pair_exp_count_strictly() -> None:
    """`exp_count` is strictly positive for any finite distance (it
    asymptotes to ``exp(-kcn) ~ 1e-7`` rather than to exactly zero), which
    allows asserting a strict inequality once the pair is inside the
    cutoff, unlike `erf_count`/`gfn2_count`, which underflow to exactly
    zero at this distance regardless of cutoff."""
    numbers = torch.tensor([1, 1], dtype=torch.long)
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]], dtype=torch.float64
    )

    tight = CNModel(count=exp_count, cutoff=50.0)
    wide = CNModel(count=exp_count, cutoff=150.0)

    structure = Structure(numbers=numbers, positions=positions)
    cn_tight = tight(structure)
    cn_wide = wide(structure)

    assert torch.all(cn_wide > cn_tight)


########################################################################
# Padding


def test_padding_rcov_jacobian_is_finite_with_element_one() -> None:
    """A batched, padded dense evaluation differentiated with respect to
    the `rcov` table must give a finite Jacobian: padding atoms look up
    element 1, whose radius is non-zero, so `r0**norm_exp` never sees a
    zero base."""
    sih4 = get_structure("mb16_43", "SiH4")
    mb = get_structure("mb16_43", "01")
    numbers = pack([sih4.numbers, mb.numbers])
    positions = pack([sih4.positions.double(), mb.positions.double()])
    structure = Structure(numbers=numbers, positions=positions)

    def cn_sum(table: torch.Tensor) -> torch.Tensor:
        model = CNModel(count=erf_count, cutoff=25.0, rcov=table)
        return model(structure).sum()

    table = radii.COV_D3(dtype=torch.double)
    jacobian = jacrev(cn_sum)(table)
    assert torch.isfinite(jacobian).all()
