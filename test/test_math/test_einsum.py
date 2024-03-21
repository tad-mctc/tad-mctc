# This file is part of tad-mctc.
#
# SPDX-Identifier: LGPL-3.0
# Copyright (C) 2023 Marvin Friede
#
# tad-mctc is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# tad_mctc is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with tad-mctc. If not, see <https://www.gnu.org/licenses/>.
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
    assert pytest.approx(ref) == math.einsum(equation, *operands)

    e_auto = math.einsum(equation, *operands, optimize="auto")
    assert pytest.approx(ref) == e_auto

    e_greedy = math.einsum_greedy(equation, *operands)
    assert pytest.approx(ref) == e_greedy
    e_greedy_2 = math.einsum(equation, *operands, optimize="greedy")
    assert pytest.approx(ref) == e_greedy_2

    e_optimal = math.einsum_optimal(equation, *operands)
    assert pytest.approx(ref) == e_optimal
    e_optimal_2 = math.einsum(equation, *operands, optimize="optimal")
    assert pytest.approx(ref) == e_optimal_2
