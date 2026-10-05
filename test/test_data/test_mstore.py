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
Test that every mstore dataset is reachable through `get_structure`.
"""

from __future__ import annotations

import pytest
import torch

from tad_mctc.data.structures import get_structure, list_records
from tad_mctc.data.structures.mstore import datasets


@pytest.mark.parametrize(
    "collection,record",
    [
        ("x23", "acetic"),
        ("mb16_43", "SiH4"),
        ("amylose", next(iter(datasets["amylose"]))),
        ("polyalanine", next(iter(datasets["polyalanine"]))),
        ("rc21", next(iter(datasets["rc21"]))),
    ],
)
def test_get_structure_looks_up_real_records(
    collection: str, record: str
) -> None:
    structure = get_structure(collection, record)

    assert structure.numbers.shape[-1] > 0
    assert structure.positions.shape == structure.numbers.shape + (3,)


def test_list_records_matches_dataset_keys() -> None:
    for collection in datasets:
        assert sorted(list_records(collection)) == sorted(datasets[collection])


def test_get_structure_keeps_record_charge() -> None:
    structure = get_structure("upu23", "0a", dtype=torch.float64)

    assert structure.charge is not None
    assert structure.charge.item() == -1.0


@pytest.mark.parametrize("charge", [2.0, torch.tensor(2.0)])
def test_get_structure_overrides_charge(charge: torch.Tensor | float) -> None:
    structure = get_structure("upu23", "0a", dtype=torch.float64, charge=charge)

    assert structure.charge is not None
    assert structure.charge.dtype == torch.float64
    assert structure.charge.item() == 2.0
