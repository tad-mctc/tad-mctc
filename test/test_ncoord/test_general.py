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
Test error handling in coordination number calculation.
"""
from __future__ import annotations

import pytest
import torch

from tad_mctc._typing import Any, CountingFunction, Protocol, Tensor
from tad_mctc.ncoord import cn_d3, cn_d4, cn_eeq, erf_count, exp_count, gfn2_count


class CNFunction(Protocol):
    """
    Type annotation for coordination number function.
    """

    def __call__(
        self,
        numbers: Tensor,
        positions: Tensor,
        *,
        counting_function: CountingFunction = erf_count,
        rcov: Tensor | None = None,
        en: Tensor | None = None,
        cutoff: Tensor | None = None,
        kcn: float = 7.5,
        **kwargs: Any,
    ) -> Tensor:
        ...


@pytest.mark.parametrize("function", [cn_d3, cn_d4, cn_eeq])
@pytest.mark.parametrize("counting_function", [erf_count, exp_count, gfn2_count])
def test_fail(function: CNFunction, counting_function: CountingFunction) -> None:
    numbers = torch.tensor([1, 1])
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    # rcov wrong shape
    with pytest.raises(ValueError):
        rcov = torch.tensor([1.0])
        function(
            numbers,
            positions,
            counting_function=counting_function,
            rcov=rcov,
        )

    # wrong numbers
    with pytest.raises(ValueError):
        numbers = torch.tensor([1])
        function(numbers, positions, counting_function=counting_function)
