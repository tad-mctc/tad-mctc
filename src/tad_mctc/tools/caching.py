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

import torch

from ..typing import Any, Callable, TypeVar

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
    def wrapper(self, *args, **kwargs):
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

    def clear(self):
        name = f"_{self.__class__.__name__}__memoization_cache"

        if hasattr(self, name):
            setattr(self, name, {})

    def get(self):
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

    def clear(*_):
        cache.clear()

    def get(*_):
        return cache

    setattr(wrapper, "clear", clear)
    setattr(wrapper, "clear_cache", clear)
    setattr(wrapper, "get_cache", get)

    return wrapper


def memoize_with_deps(
    *dependency_getters: Callable[..., Any]
):  # pragma: no cover
    """
    Memoization with multiple dependency-based cache invalidation. This
    decorator allows specification of `__slots__`. It works with and without
    function arguments.

    Warning
    -------
    This is an experimental feature, which can cause memory leaks!
    """

    def decorator(fcn: Callable[..., T]) -> Callable[..., T]:
        # creating the cache outside the wrapper shares it across instances
        cache = {}
        dependency_cache = {}

        @wraps(fcn)
        def wrapper(self, *args, **kwargs):
            # create unique key for all instances in cache dictionary
            key = (id(self), fcn.__name__, args, frozenset(kwargs.items()))

            # get current deps
            current_deps = tuple(getter(self) for getter in dependency_getters)
            cached_deps = dependency_cache.get(key)

            # Check if the cache has been invalidated
            cache_invalidated = False
            if cached_deps is None or len(cached_deps) != len(current_deps):
                cache_invalidated = True
            else:
                for curr, cached in zip(current_deps, cached_deps):
                    if not torch.equal(curr, cached):
                        cache_invalidated = True
                        break

            if not cache_invalidated and key in cache:
                return cache[key]

            # If result is not in cache or deps have changed, compute result
            result = fcn(self, *args, **kwargs)
            cache[key] = result
            dependency_cache[key] = current_deps
            return result

        def clear():
            cache.clear()
            dependency_cache.clear()

        def get():
            return cache

        def get_dep():
            return dependency_cache

        setattr(wrapper, "clear", clear)
        setattr(wrapper, "clear_cache", clear)
        setattr(wrapper, "get_cache", get)
        setattr(wrapper, "get_dep_cache", get_dep)

        return wrapper

    return decorator
