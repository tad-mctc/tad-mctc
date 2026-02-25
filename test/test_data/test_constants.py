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
Test physical constants.
"""

from __future__ import annotations

import pytest

from tad_mctc.units.codata import CODATA, get_constant


def test_codata() -> None:
    assert pytest.approx(6.62607015e-34, rel=1e-10, abs=1e-34) == CODATA.h
    assert pytest.approx(9.10938356e-31, rel=1e-10, abs=1e-31) == CODATA.me
    assert pytest.approx(299792458) == CODATA.c
    assert pytest.approx(1.380649e-23, rel=1e-10, abs=1e-23) == CODATA.kb
    assert pytest.approx(6.02214076e23) == CODATA.na

    assert pytest.approx(1.602176634e-19, rel=1e-10, abs=1e-20) == get_constant(
        "elementary charge"
    )
    assert pytest.approx(6.62607015e-34, rel=1e-10, abs=1e-34) == get_constant(
        "planck constant"
    )
