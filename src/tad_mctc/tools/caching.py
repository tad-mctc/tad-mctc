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
from typing import Any, TypeVar

from ..typing import CacheKey, Callable

__all__ = ["memoize", "memoize_all_instances"]

T = TypeVar("T")


def memoize(fcn: Callable[..., T]) -> Callable[..., T]:  # pragma: no cover
    """
    Memoization decorator that writes the cache to the object itself, hence not
    allowing the specification of `__slots__`. It works with and without
    function arguments.

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

    @wraps(fcn)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> T:
        # name mangling of dunder attributes!
        name = f"_{self.__class__.__name__}__memoization_cache"

        if hasattr(self, "__slots__"):
            # if __slots__ are defined, `__memoization_cache` must be in them
            if "__memoization_cache" not in self.__slots__:
                raise AttributeError(
                    "Cannot use memoize with objects that specify "
                    "`__slots__`, unless the `__memoization_cache` "
                    "attribute is included in them."
                )

        if not hasattr(self, name):
            # Create a cache dictionary as an instance attribute
            setattr(self, name, {})

        cache = getattr(self, name, {})

        # Create a unique key for the cache dictionary
        key = (id(self), fcn.__name__, args, frozenset(kwargs.items()))

        # If the result is already in the cache, return it
        if key in cache:
            return cache[key]

        # If key is not found, compute the result
        result = fcn(self, *args, **kwargs)
        cache[key] = result
        return result

    def clear(self: Any) -> None:
        name = f"_{self.__class__.__name__}__memoization_cache"

        if hasattr(self, name):
            setattr(self, name, {})

    def get(self: Any) -> dict[str, Any]:
        name = f"_{self.__class__.__name__}__memoization_cache"

        if not hasattr(self, name):
            return {}

        return getattr(self, name)

    setattr(wrapper, "clear", clear)
    setattr(wrapper, "clear_cache", clear)
    setattr(wrapper, "get_cache", get)

    return wrapper


def memoize_all_instances(fcn: Callable[..., T]) -> Callable[..., T]:
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
    cache: dict[CacheKey, T] = {}

    @wraps(fcn)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> T:
        # create unique key for all instances in cache dictionary
        key = (id(self), fcn.__name__, args, frozenset(kwargs.items()))

        # if the result is already in the cache, return it
        if key in cache:
            return cache[key]

        # if key is not found, compute the result
        result = fcn(self, *args, **kwargs)
        cache[key] = result
        return result

    def clear(*_: Any) -> None:
        cache.clear()

    def get(*_: Any) -> dict[CacheKey, T]:
        return cache

    setattr(wrapper, "clear", clear)
    setattr(wrapper, "clear_cache", clear)
    setattr(wrapper, "get_cache", get)

    return wrapper
