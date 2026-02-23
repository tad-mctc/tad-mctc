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
Utility functions for testing.
"""

from __future__ import annotations

import numpy as np

from tad_mctc.convert import numpy_to_tensor, symmetrizef
from tad_mctc.typing import DD, Tensor

__all__ = ["_rng", "_symrng"]


def _rng(size: tuple[int, ...] | int, dd: DD) -> Tensor:
    s = (size,) if isinstance(size, int) else size
    n = np.random.rand(*s)
    return numpy_to_tensor(n, **dd)  # type: ignore[arg-type]


def _symrng(size: tuple[int, ...] | int, dd: DD) -> Tensor:
    return symmetrizef(_rng(size, dd))
