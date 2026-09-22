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
Data: Structures - Other - Atoms
==================================

Single-atom systems, split out of :mod:`tad_mctc.data.structures.other`
purely for file-level navigability -- see that package's own docstring for
why this split follows content shape rather than mstore's per-source
convention. Reached the same way as any other bespoke record, through
:func:`tad_mctc.data.structures.get_structure` with collection
``"other"``; nothing outside
``other`` should import this module directly.
"""

from __future__ import annotations

import torch

from ....convert import symbol_to_number

__all__ = ["atoms"]


atoms: dict[str, dict[str, torch.Tensor]] = {
    "H": {
        "numbers": symbol_to_number(["H"]),
        "positions": torch.tensor(
            [[0.00000000000000, 0.00000000000000, 0.00000000000000]],
        ),
    },
    "He": {
        "numbers": symbol_to_number(["He"]),
        "positions": torch.tensor(
            [[0.00000000000000, 0.00000000000000, 0.00000000000000]],
        ),
    },
    "C": {
        "numbers": symbol_to_number(["C"]),
        "positions": torch.tensor(
            [[0.00000000000000, 0.00000000000000, 0.00000000000000]],
        ),
    },
    "S": {
        "numbers": symbol_to_number(["S"]),
        "positions": torch.tensor(
            [[0.00000000000000, 0.00000000000000, 0.00000000000000]],
        ),
    },
    "Rn": {
        "numbers": symbol_to_number(["Rn"]),
        "positions": torch.tensor(
            [[0.00000000000000, 0.00000000000000, 0.00000000000000]],
        ),
    },
}
