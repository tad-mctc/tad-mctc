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

from tad_mctc.tools.caching import memoize, memoize_all_instances


class Class:
    @memoize
    def compute(self, x, y=0):
        """A simple method to test memoization."""
        return x + y + sum(range(1000))


class ClassInstances:
    @memoize_all_instances
    def compute(self, x, y=0):
        """A simple method to test memoization."""
        return x + y + sum(range(1000))


@pytest.mark.parametrize("memoize_class", [Class, ClassInstances])
def test_memoization(memoize_class) -> None:
    obj = memoize_class()

    # Call the function with the same arguments to ensure it's cached
    first_result = obj.compute(10, y=20)
    second_result = obj.compute(10, y=20)

    assert first_result == second_result
    assert first_result == 499530
    assert id(first_result) == id(second_result)


def test_cache_separation() -> None:
    obj1 = ClassInstances()
    obj2 = ClassInstances()

    obj1.compute(5)
    obj2.compute(5)

    # Different objects DO share cache
    c1 = obj1.compute.get_cache(obj1)
    c2 = obj2.compute.get_cache(obj2)
    assert c1 == c2

    ###########################################################################

    obj3 = Class()
    obj4 = Class()

    obj3.compute(5)
    obj4.compute(5)

    # Different objects DO NOT share cache
    c3 = obj3.compute.get_cache(obj3)
    c4 = obj4.compute.get_cache(obj4)
    assert c3 != c4


@pytest.mark.parametrize("memoize_class", [Class, ClassInstances])
def test_argument_sensitivity(memoize_class) -> None:
    obj = memoize_class()

    result1 = obj.compute(5)
    result2 = obj.compute(5, 1)

    # Different results due to different arguments
    assert result1 != result2


@pytest.mark.parametrize("memoize_class", [Class, ClassInstances])
def test_clear_cache(memoize_class) -> None:
    obj = memoize_class()
    obj.compute(10)
    obj.compute.clear_cache(obj)

    # The cache should be empty after clearing
    assert not obj.compute.get_cache(obj)
