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
Tools: Caching
--------------

Decorators for memoization/caching.
"""

from __future__ import annotations

from functools import wraps

from ..typing import Callable, TypeVar

__all__ = ["memoize"]

T = TypeVar("T")


def memoize(fcn: Callable[..., T]) -> Callable[..., T]:
    """
    Memoization with shared cache among all instances of the decorated function.
    This decorator allows specification of `__slots__`. It works with and
    without function arguments.

    Note that `lru_cache` can produce memory leaks for a method.

    Parameters
    ----------
    fcn : Callable[[Any], T]
        Function to memoize

    Returns
    -------
    Callable[[Any], T]
        Memoized function.
    """

    # creating the cache outside the wrapper shares it across instances
    cache = {}

    @wraps(fcn)
    def wrapper(self, *args, **kwargs):
        # create unique key for all instances in cache dictionary
        key = (id(self), fcn.__name__, args, frozenset(kwargs.items()))

        # if the result is already in the cache, return it
        if key in cache:
            return cache[key]

        # if key is not found, compute the result
        result = fcn(self, *args, **kwargs)
        cache[key] = result
        return result

    def clear():
        cache.clear()

    def get():
        return cache

    setattr(wrapper, "clear", clear)
    setattr(wrapper, "clear_cache", clear)
    setattr(wrapper, "get_cache", get)

    return wrapper
