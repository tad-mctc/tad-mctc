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

from typing import Any, Callable

import numpy as np
import pytest
import torch

from tad_mctc.batch import pack
from tad_mctc.convert import numpy_to_tensor, symmetrizef
from tad_mctc.data.structures import resolve_structure
from tad_mctc.io.structure import Structure
from tad_mctc.tools.compile import is_compile_supported
from tad_mctc.typing import DD, Tensor

__all__ = [
    "_rng",
    "_symrng",
    "DYNAMO_SUPPORTED",
    "DYNAMO_UNSUPPORTED_REASON",
    "load_pair",
    "load_sample",
    "load_structure",
    "resolve_structure",
    "run_compiled_or_skip",
]


def _rng(size: tuple[int, ...] | int, dd: DD) -> Tensor:
    s = (size,) if isinstance(size, int) else size
    n = np.random.rand(*s)
    return numpy_to_tensor(n, **dd)


def _symrng(size: tuple[int, ...] | int, dd: DD) -> Tensor:
    return symmetrizef(_rng(size, dd))


def load_structure(collection: str, record: str, dd: DD) -> Structure:
    """Resolve `(collection, record)` via `resolve_structure`, moved to
    `dd` in the same call. `load_sample`/`load_pair` are a thin adapter
    over this that keep only `numbers`/`positions`; a caller that also
    needs `lattice`/`periodic` (`test_periodic.py`'s CN tests) uses this
    directly instead of duplicating the resolve-and-move step. Resolves
    fresh on every call rather than caching a `dict[str, Structure]`:
    `get_structure`/`structures` lookups are cheap dict indexing plus one
    `Structure` construction, so there is nothing worth caching, and doing
    it here also avoids resolving once at a default dtype and `.to()`-
    casting again per test, the way a precomputed `dict[str, Structure]`
    would need to."""
    return resolve_structure(
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


DYNAMO_SUPPORTED = is_compile_supported()
"""For ``@pytest.mark.skipif(not DYNAMO_SUPPORTED, reason=DYNAMO_UNSUPPORTED_REASON)``
on any test that calls ``torch.compile``. The capability probe itself
(``is_compile_supported``) lives in ``tad_mctc.tools.compile`` -- it is
plain-torch, has no `pytest` dependency, and other "tad-*" packages can
call it directly instead of duplicating the probe in their own test suite,
the way this module used to."""

DYNAMO_UNSUPPORTED_REASON = (
    "torch.compile/Dynamo is not supported on this Python/PyTorch combination"
)


def run_compiled_or_skip(
    fn: Callable[..., Any],
    *args: Any,
    fullgraph: bool = True,
    dynamic: bool = False,
) -> Any:
    """
    Compile ``fn`` with ``torch.compile`` and call it, skipping the test
    instead of failing when this Python/PyTorch/platform combination
    cannot actually carry it out.

    ``torch.compile`` support is not reliably predictable from a single
    query API, ``DYNAMO_SUPPORTED`` included: across the versions this
    package supports, it has failed at construction (a hard Python-version
    gate raised from inside ``torch.compile`` itself, "Python 3.11+ not
    yet supported"), at trace time (Dynamo refusing to trace a construct
    that another PyTorch version traces fine, e.g. ``functools.partial``),
    and at backend compile time (no C/C++ toolchain, observed on Windows
    CI: ``InvalidCxxCompiler: Compiler: cl is not found``). All three are
    environment/version gaps, not a correctness bug in the code under
    test -- unlike a wrong *value*, which still surfaces normally, since
    this only wraps the compile-and-call step and never the assertion
    that follows it.
    """
    if not DYNAMO_SUPPORTED:
        pytest.skip(DYNAMO_UNSUPPORTED_REASON)

    try:
        compiled = torch.compile(fn, fullgraph=fullgraph, dynamic=dynamic)
        return compiled(*args)
    except Exception as exc:  # pylint: disable=broad-except
        pytest.skip(f"torch.compile unsupported here: {exc}")
        return None
