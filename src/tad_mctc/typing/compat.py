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
Typing: Compatibility
=====================

Since typing still significantly changes across different Python versions,
all the special cases are handled here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import Tensor

from .builtin import Any

__all__ = [
    "Tensor",
    "Self",
    "TypeGuard",
    "Callable",
    "Generator",
    "Sequence",
    "PathLike",
    "Sliceable",
    "Size",
    "TensorOrTensors",
    "DampingFunction",
    "CountingFunction",
]


# Python 3.11
if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

# Python 3.10
if sys.version_info >= (3, 10):
    from typing import TypeGuard
else:
    from typing_extensions import TypeGuard

# starting with Python 3.9, type hinting generics have been moved
# from the "typing" to the "collections" module
# (see PEP 585: https://peps.python.org/pep-0585/)
if sys.version_info >= (3, 9):
    from collections.abc import Callable, Generator, Sequence
else:
    from typing import Callable, Generator, Sequence


if sys.version_info >= (3, 10):
    # "from __future__ import annotations" only affects type annotations
    # not type aliases, hence "|" is not allowed before Python 3.10

    PathLike = str | Path
    Sliceable = list[Tensor] | tuple[Tensor, ...]
    Size = list[int] | tuple[int, ...] | torch.Size
    TensorOrTensors = list[Tensor] | tuple[Tensor, ...] | Tensor
    DampingFunction = Callable[[int, Tensor, Tensor, dict[str, Tensor]], Tensor]
    CountingFunction = Callable[[Tensor, Tensor, Tensor | float | int], Tensor]
elif sys.version_info >= (3, 9):
    # in Python 3.9, "from __future__ import annotations" works with type
    # aliases but requires using `Union` from typing
    from typing import Union

    PathLike = Union[str, Path]
    Sliceable = Union[list[Tensor], tuple[Tensor, ...]]
    Size = Union[list[int], tuple[int], torch.Size]
    TensorOrTensors = Union[list[Tensor], tuple[Tensor, ...], Tensor]
    CountingFunction = Callable[[Tensor, Tensor, Union[Tensor, float, int]], Tensor]

    # no Union here, same as 3.10
    DampingFunction = Callable[[int, Tensor, Tensor, dict[str, Tensor]], Tensor]
elif sys.version_info >= (3, 8):
    # in Python 3.8, "from __future__ import annotations" only affects
    # type annotations not type aliases
    from typing import Dict, List, Tuple, Union

    PathLike = Union[str, Path]
    Sliceable = Union[List[Tensor], Tuple[Tensor, ...]]
    Size = Union[List[int], Tuple[int], torch.Size]
    TensorOrTensors = Union[List[Tensor], Tuple[Tensor, ...], Tensor]
    DampingFunction = Callable[[int, Tensor, Tensor, Dict[str, Tensor]], Tensor]
    CountingFunction = Callable[[Tensor, Tensor, Union[Tensor, float, int]], Tensor]
else:
    vinfo = sys.version_info
    raise RuntimeError(
        f"'tad_mctc' requires at least Python 3.8 (Python {vinfo.major}."
        f"{vinfo.minor}.{vinfo.micro} found)."
    )
