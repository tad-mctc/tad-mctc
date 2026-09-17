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
`tools/glu_ala/convert.py`. The much larger `glu_ala_b_512_to_65536`
ladder (up to 1.7 million atoms) is never packaged; see
`examples/scaling/glu_ala.py`, which downloads and reads both ladders
directly instead.

`data.npz` is read lazily and cached at module scope: only the arrays a
given :func:`get_structure` call actually names are ever decompressed.
Record ids are the ladder's own zero-padded filenames (``"0001"`` ..
``"2048"``), not atom counts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ....io.structure import Structure
from ....typing import get_default_dtype

__all__ = ["get_structure", "list_records"]

_DATA_PATH = Path(__file__).parent / "data.npz"
_data: np.lib.npyio.NpzFile | None = None


def _load() -> np.lib.npyio.NpzFile:
    global _data
    data = _data
    if data is None:
        data = np.load(_DATA_PATH)
        _data = data
    return data


def list_records() -> list[str]:
    """
    List every record id in the packaged glu_ala ladder, the source
    ladder's own zero-padded filenames.

    Returns
    -------
    list[str]
        The record ids, in the source ladder's own (ascending) order.
    """
    labels = {name.rsplit("_", 1)[0] for name in _load().files}
    return sorted(labels, key=int)


def get_structure(record: str) -> Structure:
    """
    Look up one glu_ala structure by record id.

    Parameters
    ----------
    record : str
        Record id, one of :func:`list_records`'s entries (e.g. ``"0064"``).

    Returns
    -------
    Structure
        The requested structure, positions in bohr.

    Raises
    ------
    KeyError
        If ``record`` is not found, listing the valid options.
    """
    data = _load()
    try:
        numbers = data[f"{record}_numbers"]
        positions = data[f"{record}_positions"]
    except KeyError:
        raise KeyError(
            f"Unknown glu_ala record '{record}'. Available: {list_records()}"
        ) from None

    return Structure(
        numbers=torch.from_numpy(numbers.astype(np.int64)),
        positions=torch.from_numpy(positions).to(get_default_dtype()),
    )
