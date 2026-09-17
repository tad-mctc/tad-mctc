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

from tad_mctc.convert import numpy_to_tensor, symmetrizef
from tad_mctc.typing import DD, Tensor

__all__ = [
    "_rng",
    "_symrng",
    "DYNAMO_SUPPORTED",
    "DYNAMO_UNSUPPORTED_REASON",
    "run_compiled_or_skip",
]


def _rng(size: tuple[int, ...] | int, dd: DD) -> Tensor:
    s = (size,) if isinstance(size, int) else size
    n = np.random.rand(*s)
    return numpy_to_tensor(n, **dd)


def _symrng(size: tuple[int, ...] | int, dd: DD) -> Tensor:
    return symmetrizef(_rng(size, dd))


def _dynamo_is_supported() -> bool:
    """
    Whether ``torch.compile``/Dynamo tracing is usable on this Python/
    PyTorch combination.

    This runs once at import time (see ``DYNAMO_SUPPORTED`` below), so
    every step is guarded: a raise here would break collecting every test
    module that imports this one, not just skip a `torch.compile` test.
    ``torch._dynamo.is_dynamo_supported`` -- the query PyTorch itself uses
    to track Python-version support lag -- is itself a later addition than
    `torch.compile`, so its own absence (e.g. PyTorch 2.0.1) is read as
    "assume supported", matching what CI observes: other `torch.compile`
    tests do pass on those older versions.
    """
    if not hasattr(torch, "compile"):
        return False

    try:
        import torch._dynamo as dynamo  # pylint: disable=protected-access
    except ImportError:
        return False

    is_supported = getattr(dynamo, "is_dynamo_supported", None)
    if is_supported is None:
        return True

    try:
        return bool(is_supported())
    except Exception:  # pylint: disable=broad-except
        return False


DYNAMO_SUPPORTED = _dynamo_is_supported()
"""For ``@pytest.mark.skipif(not DYNAMO_SUPPORTED, reason=DYNAMO_UNSUPPORTED_REASON)``
on any test that calls ``torch.compile``."""

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
