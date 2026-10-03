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
Transform conformance of `tad_mctc.batch`: `real_atoms`, `real_pairs` and
`real_triples`, on a batch of padded atomic numbers.

Exclusions: no derivative checks (boolean outputs), no padding check (the
functions take the padded ``numbers`` themselves).
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from torch.func import vmap

from tad_mctc.autograd import no_vmap_fallback
from tad_mctc.batch import real_atoms, real_pairs, real_triples

from ..conftest import DEVICE
from ..utils import (
    DYNAMO_SUPPORTED,
    DYNAMO_UNSUPPORTED_REASON,
    compile_fullgraph,
)


def _numbers() -> torch.Tensor:
    return torch.tensor(
        [[8, 1, 1, 0], [6, 1, 0, 0], [7, 1, 1, 1]], device=DEVICE
    )


def _atoms(numbers: torch.Tensor) -> torch.Tensor:
    return real_atoms(numbers)


def _pairs(numbers: torch.Tensor) -> torch.Tensor:
    return real_pairs(numbers)


def _pairs_with_diagonal(numbers: torch.Tensor) -> torch.Tensor:
    return real_pairs(numbers, mask_diagonal=False)


def _triples(numbers: torch.Tensor) -> torch.Tensor:
    return real_triples(numbers)


def _triples_with_diagonal(numbers: torch.Tensor) -> torch.Tensor:
    return real_triples(numbers, mask_diagonal=False, mask_self=False)


FUNCTIONS = [
    pytest.param(_atoms, id="real_atoms"),
    pytest.param(_pairs, id="real_pairs"),
    pytest.param(_pairs_with_diagonal, id="real_pairs-diagonal"),
    pytest.param(_triples, id="real_triples"),
    pytest.param(_triples_with_diagonal, id="real_triples-diagonal"),
]


@pytest.mark.parametrize("f", FUNCTIONS)
def test_vmap_matches_loop(f: Callable[[torch.Tensor], torch.Tensor]) -> None:
    numbers = _numbers()

    with no_vmap_fallback():
        batched = vmap(f)(numbers)

    looped = torch.stack([f(n) for n in numbers])
    assert torch.equal(batched, looped)
    assert torch.equal(batched, f(numbers))


@pytest.mark.skipif(not DYNAMO_SUPPORTED, reason=DYNAMO_UNSUPPORTED_REASON)
@pytest.mark.parametrize("f", FUNCTIONS)
def test_compile_matches_eager(
    f: Callable[[torch.Tensor], torch.Tensor],
) -> None:
    numbers = _numbers()

    compiled = compile_fullgraph(f)
    assert torch.equal(compiled(numbers), f(numbers))
