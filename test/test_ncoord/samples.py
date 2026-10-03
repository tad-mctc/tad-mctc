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
``test/references/<collection>/<record>.json``. The molecules are listed
in `MOLECULE_REFS`, the periodic cells in `CELL_REFS`, and `refs` holds
the references of both, keyed by the ``(collection, record)`` pair --
the same pair `test/utils.py`'s loaders take.

Also the hand-built periodic structures that several tests share:
`carbon_pair`, a small cell, and `bulk_and_slab`, a batch of two cells.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

import torch

from tad_mctc.io.structure import Structure, pack_structures
from tad_mctc.typing import DD, Tensor

from ..utils import load_structure

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


MOLECULE_REFS: list[tuple[str, str]] = [
    ("heavy28", "pbh4_bih3"),
    ("mb16_43", "01"),
    ("mb16_43", "02"),
    ("mb16_43", "03"),
    ("mb16_43", "SiH4"),
    ("other", "C6H5I-CH3SH"),
]
"""The reference molecules."""

LARGE_CRYSTALS: list[tuple[str, str]] = [
    ("x23", "acetic"),
    ("x23", "anthracene"),
]
"""Cells whose all-pairs Jacobian takes ~1 s each (the sparse one takes
milliseconds). Run through the all-pairs paths for every variant and
dtype, they would be a third of the whole suite, and the smaller cells
already cover the all-pairs periodic code."""

SMALL_CELL_REFS: list[tuple[str, str]] = [
    ("other", "diamond"),
    ("other", "nacl"),
    ("other", "periodic_cubic"),
    ("other", "periodic_one_atom"),
    ("other", "periodic_triclinic"),
    ("x23", "ammonia"),
]
"""The reference cells except the `LARGE_CRYSTALS`."""

CELL_REFS: list[tuple[str, str]] = SMALL_CELL_REFS + LARGE_CRYSTALS
"""The reference cells."""

refs: dict[tuple[str, str], Refs] = {
    (collection, record): _refs(_REFERENCES_DIR / collection / f"{record}.json")
    for collection, record in MOLECULE_REFS + CELL_REFS
}

REPRESENTATIVE_MOLECULES: list[tuple[str, str]] = [
    ("mb16_43", "SiH4"),  # small molecule
    ("mb16_43", "01"),  # mid-size molecule, many elements
]
"""Molecules for the tests that do not need every `refs` entry
(gradients, autograd checks)."""

REPRESENTATIVE_CELLS: list[tuple[str, str]] = [
    ("other", "periodic_one_atom"),  # interacts only with its own images
    ("other", "periodic_triclinic"),  # non-orthogonal cell
]
"""Cells for the tests that do not need every `refs` entry."""

MOLECULE_PAIRS: list[tuple[tuple[str, str], tuple[str, str]]] = [
    # different sizes, the smaller one padded
    (("mb16_43", "01"), ("mb16_43", "SiH4")),
]
"""Molecules to batch. Molecules and periodic cells are never mixed, as
`pack_structures` rejects that."""

CELL_PAIRS: list[tuple[tuple[str, str], tuple[str, str]]] = [
    # different cells: the one-atom cell needs far more images, so the
    # shared shift table must cover the more demanding lattice
    (("other", "periodic_triclinic"), ("other", "periodic_one_atom")),
]
"""Cells to batch."""


def source_id(source: tuple[str, str]) -> str:
    """Readable pytest id for one sample, its record name."""
    return source[1]


def pair_id(pair: tuple[tuple[str, str], tuple[str, str]]) -> str:
    """Readable pytest id for a pair of samples, e.g. ``01+SiH4``."""
    (_, record1), (_, record2) = pair
    return f"{record1}+{record2}"


def bulk_and_slab(dd: DD) -> Structure:
    """A batch of two cells with different lattices and periodicity, so
    each pair must pick up its own system's cell. Not a `CELL_PAIRS`
    entry: the slab has no Fortran reference."""
    bulk = load_structure("other", "periodic_cubic", dd)
    triclinic = load_structure("other", "periodic_triclinic", dd)
    slab = triclinic.replace(
        periodic=torch.tensor([True, True, False], device=dd["device"])
    )
    return pack_structures([bulk, slab])


PLACEHOLDER = 1.0
"""A short lattice vector length for non-periodic axes, as the Turbomole
`$cell` reader writes for a slab or wire. Images along such an axis would
land inside the cutoff if they were not masked out."""


def carbon_pair(dd: DD, lattice: Tensor, periodic: list[bool]) -> Structure:
    """Two carbon atoms 2 Bohr apart in `lattice`, periodic along the axes
    `periodic` marks."""
    numbers = torch.tensor([6, 6], device=dd["device"])
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.4, 1.4, 0.0]], **dd)
    return Structure(
        numbers=numbers,
        positions=positions,
        lattice=lattice,
        periodic=torch.tensor(periodic, device=dd["device"]),
    )
