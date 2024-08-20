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
Coordination number: Typing
===========================

Type annotations for CNs.
"""
from __future__ import annotations

from ..typing import Any, CountingFunction, Protocol, Tensor
from .count import derf_count, erf_count

__all__ = ["CNFunction", "CNGradFunction"]


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
        """
        Calculate the coordination number of each atom in the system.
        """
        ...


class CNGradFunction(Protocol):
    """
    Type annotation for coordination number function.
    """

    def __call__(
        self,
        numbers: Tensor,
        positions: Tensor,
        *,
        dcounting_function: CountingFunction = derf_count,
        rcov: Tensor | None = None,
        en: Tensor | None = None,
        cutoff: Tensor | None = None,
        kcn: float = 7.5,
        **kwargs: Any,
    ) -> Tensor:
        """
        Calculate the coordination number gradient of each atom in the system.
        """
        ...
