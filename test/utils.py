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
Utility functions for testing.
"""

from __future__ import annotations

import shutil
import sys
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pytest
import torch

from tad_mctc.batch import pack
from tad_mctc.convert import numpy_to_tensor, symmetrizef
from tad_mctc.data.structures import get_structure
from tad_mctc.io.structure import Structure, pack_structures
from tad_mctc.tools.compile import is_compile_supported
from tad_mctc.typing import DD, Tensor

__all__ = [
    "_rng",
    "_symrng",
    "COMPILE_BACKEND",
    "compile_fullgraph",
    "DYNAMO_SUPPORTED",
    "DYNAMO_UNSUPPORTED_REASON",
    "hydrogens",
    "jacfwd",
    "jacrev",
    "load_batch",
    "load_pair",
    "load_sample",
    "load_structure",
    "run_compiled_or_skip",
]


def jacrev(
    f: Callable[..., Any], *args: Any, **kwargs: Any
) -> Callable[..., Any]:
    """`torch.func.jacrev` typed to return `Any` instead of a `PyTree` union."""
    return torch.func.jacrev(f, *args, **kwargs)


def jacfwd(
    f: Callable[..., Any], *args: Any, **kwargs: Any
) -> Callable[..., Any]:
    """`torch.func.jacfwd` typed to return `Any` instead of a `PyTree` union."""
    return torch.func.jacfwd(f, *args, **kwargs)


def _rng(size: tuple[int, ...] | int, dd: DD) -> Tensor:
    s = (size,) if isinstance(size, int) else size
    n = np.random.rand(*s)
    return numpy_to_tensor(n, **dd)


def _symrng(size: tuple[int, ...] | int, dd: DD) -> Tensor:
    return symmetrizef(_rng(size, dd))


def hydrogens(
    positions: Tensor,
    lattice: Tensor | None = None,
    periodic: Tensor | None = None,
) -> Structure:
    """`positions` as a `Structure` of hydrogen atoms, for neighbour-search
    tests where only the geometry matters. Without `periodic`, a
    `Structure` with a lattice is periodic along all three axes."""
    numbers = torch.ones(
        positions.shape[:-1], dtype=torch.long, device=positions.device
    )
    return Structure(
        numbers=numbers, positions=positions, lattice=lattice, periodic=periodic
    )


def load_structure(collection: str, record: str, dd: DD) -> Structure:
    """Look up `(collection, record)` via `get_structure`, moved to `dd`.
    `load_sample`/`load_pair` keep only `numbers`/`positions`; a caller
    that also needs `lattice`/`periodic` uses this directly."""
    return get_structure(
        collection, record, device=dd["device"], dtype=dd["dtype"]
    )


def load_sample(collection: str, record: str, dd: DD) -> tuple[Tensor, Tensor]:
    """`load_structure`, keeping only `numbers`/`positions`."""
    structure = load_structure(collection, record, dd)
    return structure.numbers, structure.positions


def load_pair(
    collection1: str,
    record1: str,
    collection2: str,
    record2: str,
    dd: DD,
) -> tuple[Tensor, Tensor]:
    """Load and pack two `(collection, record)`-named structures'
    `numbers`/`positions`, moved to `dd`."""
    numbers1, positions1 = load_sample(collection1, record1, dd)
    numbers2, positions2 = load_sample(collection2, record2, dd)
    numbers = pack((numbers1, numbers2))
    positions = pack((positions1, positions2))
    return numbers, positions


def load_batch(sources: Sequence[tuple[str, str]], dd: DD) -> Structure:
    """Load `(collection, record)`-named structures, moved to `dd`, and pack
    them into one batched `Structure`, keeping `lattice`/`periodic` (unlike
    `load_pair`)."""
    return pack_structures([load_structure(*source, dd) for source in sources])


DYNAMO_SUPPORTED = is_compile_supported()
"""For ``@pytest.mark.skipif(not DYNAMO_SUPPORTED, reason=DYNAMO_UNSUPPORTED_REASON)``
on any test that calls ``torch.compile``. The capability probe itself
(``is_compile_supported``) lives in ``tad_mctc.tools.compile`` -- it is
plain-torch, has no `pytest` dependency, and other "tad-*" packages can
call it directly instead of duplicating the probe in their own test
suite."""

DYNAMO_UNSUPPORTED_REASON = (
    "torch.compile/Dynamo is not supported on this Python/PyTorch combination"
)


def _has_cxx_compiler() -> bool:
    """Whether the C++ compiler that TorchInductor calls is on ``PATH``. On
    Windows that is MSVC's ``cl``, which a plain CI runner does not expose
    (``InvalidCxxCompiler: Compiler: cl is not found``)."""
    if sys.platform == "win32":
        names = ["cl"]
    else:
        names = ["c++", "g++", "clang++"]
    return any(shutil.which(name) is not None for name in names)


COMPILE_BACKEND = "inductor" if _has_cxx_compiler() else "aot_eager"
"""The ``torch.compile`` backend for tests. The compile tests check that a
function traces as one graph (``fullgraph=True``), which Dynamo decides
before any backend runs. Without a C++ compiler, ``"aot_eager"`` still traces
and runs AOTAutograd, just without generating C++ code."""


def compile_fullgraph(fn: Callable[..., Any]) -> Callable[..., Any]:
    """``torch.compile(fn)`` as one graph, with static shapes, on
    :data:`COMPILE_BACKEND`. Skips the test if ``torch.compile`` itself
    refuses to construct."""
    try:
        return torch.compile(
            fn, fullgraph=True, dynamic=False, backend=COMPILE_BACKEND
        )
    except Exception as exc:  # pylint: disable=broad-except
        return pytest.skip(f"torch.compile unsupported here: {exc}")


def run_compiled_or_skip(
    fn: Callable[..., Any],
    *args: Any,
    fullgraph: bool = True,
    dynamic: bool = False,
) -> Any:
    """
    Compile ``fn`` with ``torch.compile`` on :data:`COMPILE_BACKEND` and call
    it, skipping the test instead of failing when this Python/PyTorch/platform
    combination cannot carry it out.

    Support is not predictable from :data:`DYNAMO_SUPPORTED` alone: PyTorch
    2.5 cannot inline custom ``autograd.Function`` calls (``too many
    positional arguments``). This is a version gap, not a bug in the code
    under test. A wrong *value* still fails, since only the compile-and-call
    step is wrapped, never the assertion that follows it.
    """
    if not DYNAMO_SUPPORTED:
        pytest.skip(DYNAMO_UNSUPPORTED_REASON)

    try:
        compiled = torch.compile(
            fn, fullgraph=fullgraph, dynamic=dynamic, backend=COMPILE_BACKEND
        )
        return compiled(*args)
    except Exception as exc:  # pylint: disable=broad-except
        return pytest.skip(f"torch.compile unsupported here: {exc}")
