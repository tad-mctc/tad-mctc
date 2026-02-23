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
Test Einstein summations.
"""

from __future__ import annotations

import pytest
import torch

from tad_mctc import math


def test_functions_existence():
    assert callable(math.einsum)
    assert callable(math.einsum_greedy)
    assert callable(math.einsum_optimal)


# These tests are pointless if `opt_einsum` is not installed. But they should
# pass anyway as they are just wrappers around the `torch.einsum` function.
def test_optimization_flags():
    operands = (torch.rand(2, 3), torch.rand(3, 4))
    equation = "ij,jk->ik"

    ref = torch.einsum(equation, *operands)
    assert pytest.approx(ref.cpu()) == math.einsum(equation, *operands).cpu()

    e_auto = math.einsum(equation, *operands, optimize="auto")
    assert pytest.approx(ref.cpu()) == e_auto.cpu()

    e_greedy = math.einsum_greedy(equation, *operands)
    assert pytest.approx(ref.cpu()) == e_greedy.cpu()
    e_greedy_2 = math.einsum(equation, *operands, optimize="greedy")
    assert pytest.approx(ref.cpu()) == e_greedy_2.cpu()

    e_optimal = math.einsum_optimal(equation, *operands)
    assert pytest.approx(ref.cpu()) == e_optimal.cpu()
    e_optimal_2 = math.einsum(equation, *operands, optimize="optimal")
    assert pytest.approx(ref.cpu()) == e_optimal_2.cpu()
