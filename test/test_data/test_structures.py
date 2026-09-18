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
Test the optional `charge`/`uhf` fields on `Structure` and the mstore
records that carry them.
"""

from __future__ import annotations

import torch

from tad_mctc.data.structures import merge_nested_dicts
from tad_mctc.data.structures.mstore import mb16_43
from tad_mctc.io.structure import Structure


def test_structure_accepts_charge_and_uhf() -> None:
    structure = Structure(
        numbers=torch.tensor([3, 1]),
        positions=torch.zeros((2, 3)),
        charge=torch.tensor(1),
        uhf=torch.tensor(1),
    )

    assert structure.charge == 1
    assert structure.uhf == 1


def test_structure_charge_and_uhf_are_absent_by_default() -> None:
    # a `Structure` without `charge`/`uhf` is neutral, closed-shell: the
    # fields are `None` rather than present-and-zero.
    structure = Structure(
        numbers=torch.tensor([3, 1]),
        positions=torch.zeros((2, 3)),
    )

    assert structure.charge is None
    assert structure.uhf is None


def test_mb16_43_open_shell_records_have_uhf() -> None:
    # mstore's mindless02/07/08 are the three currently-mirrored mb16_43
    # records built with `uhf=1` upstream. `mb16_43` itself holds raw
    # per-record dicts, not `Structure` instances -- only `get_structure`
    # wraps a record -- so this stays a plain dict subscript.
    for record in ("02", "07", "08"):
        assert mb16_43[record]["uhf"] == torch.tensor(1)


def test_mb16_43_closed_shell_records_have_no_uhf_key() -> None:
    for record in ("01", "03", "H2", "LiH", "S2", "SiH4"):
        assert "uhf" not in mb16_43[record]


def test_merge_nested_dicts_carries_optional_keys() -> None:
    source: dict[str, dict[str, torch.Tensor]] = {
        "mol": {
            "numbers": torch.tensor([3, 1]),
            "positions": torch.zeros((2, 3)),
            "uhf": torch.tensor(1),
        },
    }
    target: dict[str, dict[str, int]] = {"mol": {"cn": 4}}

    merged = merge_nested_dicts(source, target)

    assert merged["mol"]["cn"] == 4
    assert merged["mol"]["uhf"] == torch.tensor(1)
