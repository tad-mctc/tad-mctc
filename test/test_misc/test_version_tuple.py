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
Test conversion of version string to tuple.
"""

from __future__ import annotations

import pytest

from tad_mctc._version import version_tuple


def test_valid_version_strings():
    """Test whether valid version strings are recognized."""
    assert version_tuple("1.2.3") == (1, 2, 3), "Standard version string"
    assert version_tuple("10.20.30") == (
        10,
        20,
        30,
    ), "Multi-digit numbers"
    assert version_tuple("1.2.3-alpha") == (1, 2, 3), "Pre-release version"
    assert version_tuple("1.2.3+build.4") == (1, 2, 3), "Build metadata"
    assert version_tuple("1.2.3a0+git7482eb2") == (
        1,
        2,
        3,
    ), "Pre-release with git hash"
    assert version_tuple("1.2.3_20210304") == (
        1,
        2,
        3,
    ), "Version with underscore"


def test_edge_cases():
    """Test edge cases for version strings."""
    assert version_tuple("0.0.1") == (0, 0, 1), "Minimal version numbers"
    assert version_tuple("1.2.3.4") == (1, 2, 3), "Extra version parts"


def test_fail():
    """Test failure of unsupported version string."""
    # Missing patch number
    with pytest.raises(RuntimeError):
        version_tuple("1.2")

    # Missing minor and patch numbers
    with pytest.raises(RuntimeError):
        version_tuple("1")

    # Non-integer version part
    with pytest.raises(ValueError):
        version_tuple("1.2.x")
