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
Typing: Builtins
================

This module contains all used built-in type annotations.
"""
from typing import (
    IO,
    Any,
    Callable,
    Iterable,
    Iterator,
    Literal,
    NoReturn,
    Protocol,
    TypedDict,
    TypeVar,
    overload,
    runtime_checkable,
)

__all__ = [
    "IO",
    "Any",
    "Callable",
    "Iterable",
    "Iterator",
    "Literal",
    "NoReturn",
    "Protocol",
    "TypedDict",
    "TypeVar",
    "overload",
    "runtime_checkable",
    "_wraps",
]

T = TypeVar("T")


def _wraps(
    wrapped: Callable,
    namestr: str | None = None,
    docstr: str | None = None,
    **kwargs: Any,
) -> Callable[[T], T]:
    def wrapper(fun: T) -> T:
        try:
            name = getattr(wrapped, "__name__", "<unnamed function>")
            doc = getattr(wrapped, "__doc__", "") or ""
            fun.__dict__.update(getattr(wrapped, "__dict__", {}))
            fun.__annotations__ = getattr(wrapped, "__annotations__", {})
            fun.__name__ = name if namestr is None else namestr.format(fun=name)
            fun.__module__ = getattr(wrapped, "__module__", "<unknown module>")
            fun.__doc__ = (
                doc if docstr is None else docstr.format(fun=name, doc=doc, **kwargs)
            )
            fun.__qualname__ = getattr(wrapped, "__qualname__", fun.__name__)
            fun.__wrapped__ = wrapped
        finally:
            return fun

    return wrapper
