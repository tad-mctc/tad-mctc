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
Test the PyTorch version object.
"""

from __future__ import annotations

from tad_mctc._version import __tversion__


def test_tversion_compares_with_tuples() -> None:
    """`__tversion__` compares with integer tuples."""
    assert __tversion__ > (1, 0, 0)
    assert __tversion__ < (99, 0, 0)


def test_tversion_orders_prereleases_first() -> None:
    """A pre-release sorts before its release, local labels are ignored."""
    prerelease = type(__tversion__)("2.8.0a0+git7482eb2")
    assert prerelease < (2, 8, 0)
    assert prerelease >= (2, 7, 0)
    assert type(__tversion__)("2.8.0+cu128") >= (2, 8, 0)
