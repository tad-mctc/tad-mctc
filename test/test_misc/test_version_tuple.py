# This file is part of tad-mctc.
#
# SPDX-Identifier: LGPL-3.0
# Copyright (C) 2023 Marvin Friede
#
# tad-mctc is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# tad_mctc is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with tad-mctc. If not, see <https://www.gnu.org/licenses/>.
"""
Test conversion of version string to tuple.
"""
from __future__ import annotations

import pytest

from tad_mctc._version import version_tuple


def test_valid_version_strings():
    assert version_tuple("1.2.3") == (1, 2, 3), "Standard version string"
    assert version_tuple("10.20.30") == (10, 20, 30), "Multi-digit version numbers"
    assert version_tuple("1.2.3-alpha") == (1, 2, 3), "Pre-release version"
    assert version_tuple("1.2.3+build.4") == (1, 2, 3), "Build metadata"
    assert version_tuple("1.2.3_20210304") == (1, 2, 3), "Version with underscore"


def test_edge_cases():
    assert version_tuple("0.0.1") == (0, 0, 1), "Minimal version numbers"
    assert version_tuple("1.2.3.4") == (1, 2, 3), "Extra version parts"


def test_fail():
    # Missing patch number
    with pytest.raises(RuntimeError):
        version_tuple("1.2")

    # Missing minor and patch numbers
    with pytest.raises(RuntimeError):
        version_tuple("1")

    # Non-integer version part
    with pytest.raises(ValueError):
        version_tuple("1.2.x")
