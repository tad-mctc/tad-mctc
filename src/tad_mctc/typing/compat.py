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
Typing: Compatibility
=====================

Since typing still significantly changes across different Python versions,
all the special cases are handled here.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Protocol, TypeVar

import torch
from torch import Tensor

__all__ = [
    "CacheKey",
    "Callable",
    "CountingFunction",
    "DampingFunction",
    "Generator",
    "NotRequired",
    "PairWeightFunction",
    "PathLike",
    "Self",
    "Sequence",
    "Size",
    "Sliceable",
    "TableFunction",
    "Tensor",
    "TensorOrTensors",
    "TypeAlias",
    "TypeGuard",
    "Unpack",
    "override",
    "_wraps",
]


# Python 3.12
if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

# Python 3.11
if sys.version_info >= (3, 11):
    from typing import NotRequired, Self, Unpack
else:
    from typing_extensions import NotRequired, Self, Unpack

# Python 3.10
if sys.version_info >= (3, 10):
    from typing import TypeAlias, TypeGuard
else:
    from typing_extensions import TypeAlias, TypeGuard

# starting with Python 3.9, type hinting generics have been moved
# from the "typing" to the "collections" module
# (see PEP 585: https://peps.python.org/pep-0585/)
if sys.version_info >= (3, 9):
    from collections.abc import Callable, Generator, Sequence
else:
    from typing import Callable, Generator, Sequence

CountingFunction = Callable[..., Tensor]
"""
Signature every counting function (``exp_count``, ``erf_count``,
``gfn2_count``, and their derivatives) satisfies: ``(r, r0, *params) ->
Tensor``, where ``*params`` is each function's own set of optional
control parameters (a single ``kcn`` for ``exp_count``, ``kcn`` and
``norm_exp`` for ``erf_count``, ``ka``/``kb``/``r_shift`` for
``gfn2_count``). ``Callable[[Tensor, Tensor], Tensor]`` would reject
every call that passes one of those.
"""

# `(en_i, en_j) -> Tensor`: the weight on atom j's count as it is added to
# cn[i]. Elementwise, so it broadcasts over pairs.
PairWeightFunction = Callable[[Tensor, Tensor], Tensor]


class TableFunction(Protocol):
    """
    A per-element table (such as ``radii.COV_D3``), created on the given
    device and dtype.
    """

    def __call__(
        self, *, device: torch.device | None = None, dtype: torch.dtype
    ) -> Tensor: ...


if sys.version_info >= (3, 10):
    # "from __future__ import annotations" only affects type annotations
    # not type aliases, hence "|" is not allowed before Python 3.10

    PathLike = str | Path
    Sliceable = list[Tensor] | tuple[Tensor, ...]
    Size = list[int] | tuple[int, ...] | torch.Size
    TensorOrTensors = list[Tensor] | tuple[Tensor, ...] | Tensor
    DampingFunction = Callable[[int, Tensor, Tensor, dict[str, Tensor]], Tensor]

    CacheKey = tuple[int, str, tuple[Any, ...], frozenset[tuple[str, Any]]]
elif sys.version_info >= (3, 9):
    # in Python 3.9, "from __future__ import annotations" works with type
    # aliases but requires using `Union` from typing
    from typing import Union

    PathLike = Union[str, Path]
    Sliceable = Union[list[Tensor], tuple[Tensor, ...]]
    Size = Union[list[int], tuple[int], torch.Size]
    TensorOrTensors = Union[list[Tensor], tuple[Tensor, ...], Tensor]

    # no Union here, same as 3.10
    DampingFunction = Callable[[int, Tensor, Tensor, dict[str, Tensor]], Tensor]
    CacheKey = tuple[int, str, tuple[Any, ...], frozenset[tuple[str, Any]]]
elif sys.version_info >= (3, 8):
    # in Python 3.8, "from __future__ import annotations" only affects
    # type annotations not type aliases
    from typing import Dict, FrozenSet, List, Tuple, Union

    PathLike = Union[str, Path]
    Sliceable = Union[List[Tensor], Tuple[Tensor, ...]]
    Size = Union[List[int], Tuple[int], torch.Size]
    TensorOrTensors = Union[List[Tensor], Tuple[Tensor, ...], Tensor]
    DampingFunction = Callable[[int, Tensor, Tensor, Dict[str, Tensor]], Tensor]

    CacheKey = Tuple[int, str, Tuple[Any, ...], FrozenSet[Tuple[str, Any]]]
else:
    vinfo = sys.version_info
    raise RuntimeError(
        f"'tad_mctc' requires at least Python 3.8 (Python {vinfo.major}."
        f"{vinfo.minor}.{vinfo.micro} found)."
    )


T = TypeVar("T")


def _wraps(
    wrapped: Callable[..., Any],
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
            fun.__name__ = name if namestr is None else namestr.format(fun=name)  # type: ignore
            fun.__module__ = getattr(wrapped, "__module__", "<unknown module>")
            fun.__doc__ = (
                doc
                if docstr is None
                else docstr.format(fun=name, doc=doc, **kwargs)
            )
            fun.__qualname__ = getattr(wrapped, "__qualname__", fun.__name__)  # type: ignore
            fun.__wrapped__ = wrapped  # type: ignore
        except AttributeError:
            # Best-effort metadata propagation: some callables/objects do not
            # expose or allow setting all wrapped attributes.
            pass
        return fun

    return wrapper
