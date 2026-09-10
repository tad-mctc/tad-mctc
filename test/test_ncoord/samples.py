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
Coordination number reference values, computed by the mctc-lib Fortran
library itself (see ``tools/refs`` for how and why) rather than hand-copied
from another testsuite. One JSON file per molecule lives in
``test/references/`` and is loaded here, merged with the molecular
geometries from :mod:`tad_mctc.data.molecules`.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import TypedDict

import torch

from tad_mctc.data.molecules import merge_nested_dicts, mols
from tad_mctc.typing import Molecule, Tensor

_REFERENCES_DIR = Path(__file__).resolve().parents[1] / "references"


class Refs(TypedDict):
    """Format of reference values."""

    cn_d3: Tensor
    """DFT-D3 coordination number."""

    dcn3dr: Tensor
    """Derivative of DFT-D3 coordination number w.r.t. positions."""

    cn_d4: Tensor
    """DFT-D4 coordination number"""

    dcn_d4dr: Tensor
    """Derivative of DFT-D4 coordination number w.r.t. positions."""

    cn_eeq: Tensor
    """EEQ coordination number."""

    dcn_eeqdr: Tensor
    """Derivative of EEQ coordination number w.r.t. positions."""

    cn_gfn2: Tensor
    """GFN2-xTB coordination number."""

    dcn_gfn2dr: Tensor
    """Derivative of GFN2-xTB coordination number w.r.t. positions."""

    cn_eeq_en: Tensor
    """Electronegativity-weighted EEQ coordination number."""

    dcn_eeq_endr: Tensor
    """Derivative of electronegativity-weighted EEQ coordination number
    w.r.t. positions."""

    cn_eeqbc: Tensor
    """EEQBC coordination number."""

    dcn_eeqbcdr: Tensor
    """Derivative of EEQBC coordination number w.r.t. positions."""

    cn_eeqbc_en: Tensor
    """Electronegativity-weighted EEQBC coordination number."""

    dcn_eeqbc_endr: Tensor
    """Derivative of electronegativity-weighted EEQBC coordination number
    w.r.t. positions."""


class Record(Molecule, Refs):
    """Store for molecular information and reference values."""


@lru_cache
def _load(name: str) -> dict:
    return json.loads((_REFERENCES_DIR / f"{name}.json").read_text())


def _refs(name: str) -> Refs:
    data = _load(name)
    return {
        key: torch.tensor(data[key], dtype=torch.double)
        for key in Refs.__annotations__
    }  # type: ignore[return-value]


refs: dict[str, Refs] = {
    path.stem: _refs(path.stem)
    for path in sorted(_REFERENCES_DIR.glob("*.json"))
}


samples: dict[str, Record] = merge_nested_dicts(mols, refs)
