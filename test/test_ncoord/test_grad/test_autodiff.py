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
Autograd `gradcheck`/`gradgradcheck` of every coordination-number variant
w.r.t. positions, for molecules and periodic cells alike.
"""

from __future__ import annotations

import pytest
import torch

from tad_mctc.autograd import dgradcheck, dgradgradcheck
from tad_mctc.io.structure import Structure
from tad_mctc.typing import DD, Callable, Tensor

from ...conftest import DEVICE
from ...utils import load_batch, load_structure
from .._variants import VARIANTS
from ..samples import BATCH_PAIRS, REPRESENTATIVES, pair_id

tol = 1e-8

DD_DOUBLE: DD = {"device": DEVICE, "dtype": torch.double}


def gradchecker(
    structure: Structure, variant_name: str
) -> tuple[Callable[[Tensor], Tensor], Tensor]:
    """The CN as a function of positions alone, and the positions to
    differentiate it at."""
    cn_function = VARIANTS[variant_name].call

    def func(pos: Tensor) -> Tensor:
        return cn_function(structure.replace(positions=pos))

    positions = structure.positions.detach().clone().requires_grad_(True)
    return func, positions


@pytest.mark.grad
@pytest.mark.parametrize("source", REPRESENTATIVES, ids=lambda s: s[1])
@pytest.mark.parametrize("variant_name", list(VARIANTS))
def test_gradcheck(source: tuple[str, str], variant_name: str) -> None:
    structure = load_structure(*source, DD_DOUBLE)
    func, diffvars = gradchecker(structure, variant_name)
    assert dgradcheck(func, diffvars, atol=tol)


@pytest.mark.grad
@pytest.mark.parametrize("source", REPRESENTATIVES, ids=lambda s: s[1])
@pytest.mark.parametrize("variant_name", list(VARIANTS))
def test_gradgradcheck(source: tuple[str, str], variant_name: str) -> None:
    structure = load_structure(*source, DD_DOUBLE)
    func, diffvars = gradchecker(structure, variant_name)
    assert dgradgradcheck(func, diffvars, atol=tol)


@pytest.mark.grad
@pytest.mark.parametrize("pair", BATCH_PAIRS, ids=pair_id)
@pytest.mark.parametrize("variant_name", list(VARIANTS))
def test_gradcheck_batch(
    pair: tuple[tuple[str, str], tuple[str, str]], variant_name: str
) -> None:
    structure = load_batch(pair, DD_DOUBLE)
    func, diffvars = gradchecker(structure, variant_name)
    assert dgradcheck(func, diffvars, atol=tol)


@pytest.mark.grad
@pytest.mark.parametrize("pair", BATCH_PAIRS, ids=pair_id)
@pytest.mark.parametrize("variant_name", list(VARIANTS))
def test_gradgradcheck_batch(
    pair: tuple[tuple[str, str], tuple[str, str]], variant_name: str
) -> None:
    structure = load_batch(pair, DD_DOUBLE)
    func, diffvars = gradchecker(structure, variant_name)
    assert dgradgradcheck(func, diffvars, atol=tol)
