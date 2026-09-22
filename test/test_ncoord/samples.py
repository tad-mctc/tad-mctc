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
Coordination number references computed by mctc-lib itself (see
``tools/refs``), one JSON file per structure under
``test/references/<collection>/<record>.json``. `refs` holds every file
found there, molecules and periodic cells alike, keyed by its
``(collection, record)`` pair -- the same pair `test/utils.py`'s loaders
take.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

import torch

from tad_mctc.data.structures import get_structure
from tad_mctc.typing import Tensor

_REFERENCES_DIR = Path(__file__).resolve().parents[1] / "references"


class Refs(TypedDict):
    """One CN/dCN-dr pair per counting-function variant: `cn_<variant>` is
    the coordination number, `dcn_<variant>dr` its derivative w.r.t.
    positions."""

    cn_d3: Tensor
    dcn_d3dr: Tensor
    cn_d4: Tensor
    dcn_d4dr: Tensor
    cn_eeq: Tensor
    dcn_eeqdr: Tensor
    cn_gfn2: Tensor
    dcn_gfn2dr: Tensor
    cn_eeq_en: Tensor
    dcn_eeq_endr: Tensor
    cn_eeqbc: Tensor
    dcn_eeqbcdr: Tensor
    cn_eeqbc_en: Tensor
    dcn_eeqbc_endr: Tensor


def _refs(path: Path) -> Refs:
    data = json.loads(path.read_text())
    return {
        key: torch.tensor(data[key], dtype=torch.double)
        for key in Refs.__annotations__
    }  # type: ignore[return-value]


def _source(path: Path) -> tuple[str, str]:
    """The ``(collection, record)`` pair a reference file belongs to."""
    return path.parent.name, path.stem


refs: dict[tuple[str, str], Refs] = {
    _source(path): _refs(path)
    for path in sorted(_REFERENCES_DIR.glob("*/*.json"))
}


def is_periodic(source: tuple[str, str]) -> bool:
    """Whether `source` resolves to a `Structure` with a lattice."""
    return get_structure(*source).lattice is not None


REPRESENTATIVES: list[tuple[str, str]] = [
    ("mb16_43", "SiH4"),  # small molecule
    ("mb16_43", "01"),  # mid-size molecule, many elements
    ("other", "periodic_one_atom"),  # interacts only with its own images
    ("other", "periodic_triclinic"),  # non-orthogonal cell
]
"""Samples for the tests that do not need every `refs` entry
(gradients, autograd checks)."""

BATCH_PAIRS: list[tuple[tuple[str, str], tuple[str, str]]] = [
    # different sizes, the smaller one padded
    (("mb16_43", "01"), ("mb16_43", "SiH4")),
    # different cells: the one-atom cell needs far more images, so the
    # shared shift table must cover the more demanding lattice
    (("other", "periodic_triclinic"), ("other", "periodic_one_atom")),
]
"""One molecular and one periodic pair to batch. Molecules and periodic
cells are never mixed, as `pack_structures` rejects that."""


def pair_id(pair: tuple[tuple[str, str], tuple[str, str]]) -> str:
    """Readable pytest id for one `BATCH_PAIRS` entry, e.g. ``01+SiH4``."""
    (_, record1), (_, record2) = pair
    return f"{record1}+{record2}"
