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
Test the ``memoize`` decorator against a small local class.

``Mol`` used to be the only ``@memoize`` consumer in this codebase; now that
it is gone (its cache-per-instance behavior is not part of ``Structure``,
which is a plain dict), this decorator is exercised directly against a
minimal stand-in class instead.
"""

from __future__ import annotations

from typing import Any, cast

from tad_mctc.tools import memoize


class Counter:
    """Counts how many times its memoized method actually runs its body."""

    __slots__ = ["value", "calls", "__memoization_cache"]

    def __init__(self, value: int) -> None:
        self.value = value
        self.calls = 0

    @memoize
    def doubled(self) -> int:
        self.calls += 1
        return 2 * self.value


def _cache(counter: Counter) -> dict[str, Any]:
    """
    Read back ``counter.doubled``'s cache. ``get_cache`` and ``clear`` are
    attached to the decorated function at runtime by ``@memoize``, so neither
    mypy nor pyright can see them on the declared ``Callable`` return type;
    the single `cast` here keeps every call site free of an ignore comment.
    """
    fn = cast(Any, counter.doubled)
    return fn.get_cache(counter)


def _clear_cache(counter: Counter) -> None:
    fn = cast(Any, counter.doubled)
    fn.clear(counter)


def test_cache() -> None:
    counter = Counter(21)
    assert hasattr(counter.doubled, "get_cache")

    first = counter.doubled()
    second = counter.doubled()

    assert first == second == 42
    assert counter.calls == 1, "second call should be served from the cache"

    for key in _cache(counter).keys():
        assert "doubled" in key


def test_clear_cache() -> None:
    counter = Counter(10)
    counter.doubled()
    assert _cache(counter) != {}

    _clear_cache(counter)
    assert _cache(counter) == {}

    counter.doubled()
    assert counter.calls == 2, "a cleared cache must recompute on next call"


def test_cache_keyed_by_instance() -> None:
    """Two instances of the same class must not share a cache entry."""
    a = Counter(1)
    b = Counter(2)

    assert a.doubled() == 2
    assert b.doubled() == 4
    assert a.calls == 1
    assert b.calls == 1
