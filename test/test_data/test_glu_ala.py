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
Test the lazily loaded glu_ala collection through `get_structure`.
"""

from __future__ import annotations

import pytest
import torch

from tad_mctc.data.structures import get_structure, list_records


def test_list_records_ascending_by_size() -> None:
    records = list_records("glu_ala")
    assert records == sorted(records, key=int)
    assert records[0] == "0001"
    assert records[-1] == "2048"
    assert len(records) == 26


@pytest.mark.parametrize("record", ["0001", "0064", "2048"])
def test_get_structure_looks_up_real_records(record: str) -> None:
    structure = get_structure("glu_ala", record)

    assert structure.numbers.shape[-1] > 0
    assert structure.numbers.dtype == torch.long
    assert structure.positions.shape == structure.numbers.shape + (3,)
    assert structure.positions.dtype == torch.get_default_dtype()


def test_get_structure_smallest_matches_known_atom_count() -> None:
    # 28 atoms, cross-checked against the source xyz file's own header
    structure = get_structure("glu_ala", "0001")
    assert int(structure.numbers.shape[0]) == 28


def test_get_structure_raises_on_unknown_record() -> None:
    with pytest.raises(KeyError, match="in collection 'glu_ala'"):
        get_structure("glu_ala", "not-a-real-record")
