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
Data: Structures - glu_ala
===========================

The `glu_ala_a_0001_to_2048` size ladder from
https://www.ergoscf.org/xyz/gluala.php -- 26 extended glutamine-alanine
peptide conformers, 28 to 53,250 atoms -- packed into `data.npz` by
`tools/glu_ala/convert.py`.

Record ids are the ladder's own zero-padded filenames (``"0001"`` ..
``"2048"``), not atom counts. `data.npz` is opened on first access and
only the arrays of the requested record are decompressed, so importing
this module stays cheap. Look records up through
:func:`tad_mctc.data.structures.get_structure` with collection
``"glu_ala"``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, Mapping

import numpy as np
import torch
from torch import Tensor

from ....typing import get_default_dtype

__all__ = ["glu_ala"]

_DATA_PATH = Path(__file__).parent / "data.npz"


class _GluAlaRecords(Mapping[str, Dict[str, Tensor]]):
    """
    Read-only mapping from record id to that record's `numbers` and
    `positions` (bohr), backed by `data.npz`.
    """

    def __init__(self) -> None:
        self._data: np.lib.npyio.NpzFile | None = None

    def _load(self) -> np.lib.npyio.NpzFile:
        if self._data is None:
            self._data = np.load(_DATA_PATH)

        assert self._data is not None
        return self._data

    def __getitem__(self, record: str) -> dict[str, Tensor]:
        data = self._load()
        try:
            numbers = data[f"{record}_numbers"]
            positions = data[f"{record}_positions"]
        except KeyError:
            raise KeyError(record) from None

        # `data.npz` stores compact dtypes (uint8, float32) to keep the
        # package small; convert to the library's usual types.
        return {
            "numbers": torch.from_numpy(numbers.astype(np.int64)),
            "positions": torch.from_numpy(positions).to(get_default_dtype()),
        }

    def __iter__(self) -> Iterator[str]:
        # npz keys are "<record>_numbers" and "<record>_positions"
        records = {name.rsplit("_", 1)[0] for name in self._load().files}
        return iter(sorted(records, key=int))

    def __len__(self) -> int:
        return len(self._load().files) // 2


glu_ala: Mapping[str, dict[str, Tensor]] = _GluAlaRecords()
"""The glu_ala ladder, keyed by record id in ascending size."""
