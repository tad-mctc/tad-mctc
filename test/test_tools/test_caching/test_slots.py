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
Test caching.
"""

from __future__ import annotations

import pytest

from tad_mctc.tools.caching import memoize


class DummyClass:
    __slots__ = ["dummy", "__memoization_cache"]

    def __init__(self):
        self.dummy = 0

    @memoize
    def compute(self, x, y=0):
        """A simple method to test memoization."""
        return self.dummy + x + y + sum(range(1000))


def test_fail() -> None:
    class DummyClassSlots:
        __slots__ = ["dummy"]

        @memoize
        def compute(self, x, y=0):
            """A simple method to test memoization."""
            return x + y + sum(range(1000))

    with pytest.raises(AttributeError):
        obj = DummyClassSlots()
        obj.compute(10, y=20)


def test_memoization() -> None:
    obj = DummyClass()

    # Call the function with the same arguments to ensure it's cached
    first_result = obj.compute(10, y=20)
    second_result = obj.compute(10, y=20)

    assert first_result == second_result
    assert first_result == 499530
    assert id(first_result) == id(second_result)


def test_cache_separation():
    obj1 = DummyClass()
    obj2 = DummyClass()

    obj1.compute(5)
    obj2.compute(5)

    # Different objects do NOT share cache
    c1 = obj1.compute.get_cache(obj1)
    c2 = obj2.compute.get_cache(obj2)
    assert c1 != c2


def test_argument_sensitivity():
    obj = DummyClass()

    result1 = obj.compute(5)
    result2 = obj.compute(5, 1)

    # Different results due to different arguments
    assert result1 != result2


def test_clear_cache():
    obj = DummyClass()
    obj.compute(10)
    obj.compute.clear_cache(obj)

    # The cache should be empty after clearing
    assert not obj.compute.get_cache(obj)
