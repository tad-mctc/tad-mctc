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
Test error handling in coordination number calculation.
"""

from __future__ import annotations

import pytest
import torch

from tad_mctc.io.structure import Structure
from tad_mctc.ncoord import (
    CNModel,
    cn_d3,
    cn_d3_gradient,
    cn_d4,
    cn_eeq,
    cn_eeq_en,
    cn_gfn2,
    derf_count,
    dexp_count,
    dgfn2_count,
    erf_count,
    exp_count,
    gfn2_count,
)
from tad_mctc.typing import DD, CNFunc, CNGradFunction, CountingFunction

from ..conftest import DEVICE


@pytest.mark.parametrize("function", [cn_d3, cn_d4, cn_gfn2, cn_eeq, cn_eeq_en])
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_fail(function: CNFunc, dtype: torch.dtype) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    numbers = torch.tensor([1, 1], device=DEVICE)
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], **dd)

    # `Structure.__post_init__` (`structure_check`) rejects the shape
    # mismatch before `CNModel` ever sees it.
    with pytest.raises(RuntimeError):
        wrong_positions = positions[:1]
        function(Structure(numbers=numbers, positions=wrong_positions))

    with pytest.raises(RuntimeError):
        wrong_numbers = torch.tensor([1], device=DEVICE)
        function(Structure(numbers=wrong_numbers, positions=positions))


@pytest.mark.parametrize("cfunc", [erf_count, exp_count, gfn2_count])
def test_coordination_number_custom_counting(cfunc: CountingFunction) -> None:
    numbers = torch.tensor([6, 1], dtype=torch.long)
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64
    )

    model = CNModel(count=cfunc, cutoff=5.0)
    res = model(Structure(numbers=numbers, positions=positions))
    assert torch.isfinite(res).all()


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


@pytest.mark.parametrize("function", [cn_d3_gradient])
@pytest.mark.parametrize("cfunc", [derf_count, dexp_count, dgfn2_count])
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_grad_fail(
    function: CNGradFunction, cfunc: CountingFunction, dtype: torch.dtype
) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    numbers = torch.tensor([1, 1], device=DEVICE)
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], **dd)

    # rcov is a per-element table indexed by atomic number (same convention
    # as `CNModel.rcov`), not a per-atom array; like `CNModel`, there is no
    # explicit shape validation, so a table too small for the atomic
    # numbers present fails via the gather itself. On CUDA, an
    # out-of-bounds gather is undefined behavior rather than a catchable
    # `IndexError` -- it raises an async device-side assert that poisons
    # the CUDA context for the rest of the process, so this half of the
    # check only runs on CPU.
    if DEVICE is None or DEVICE.type == "cpu":
        with pytest.raises(IndexError):
            rcov = torch.tensor([1.0], **dd)
            function(numbers, positions, dcounting_function=cfunc, rcov=rcov)

    # wrong numbers
    with pytest.raises(ValueError):
        numbers = torch.tensor([1], device=DEVICE)
        function(numbers, positions, dcounting_function=cfunc)
