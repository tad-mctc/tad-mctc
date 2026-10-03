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
Test the shared ``is_compiling`` tracing-state helper and the
``is_compile_supported`` capability probe directly. Their callers
(``math/einsum.py``, ``storch/elemental.py``) exercise them only
indirectly, through their own compile tests.
"""

from __future__ import annotations

import pytest
import torch

from tad_mctc.tools import is_compile_supported, is_compiling

from ..utils import (
    DYNAMO_SUPPORTED,
    DYNAMO_UNSUPPORTED_REASON,
    run_compiled_or_skip,
)


def test_is_compiling_outside_compile() -> None:
    assert is_compiling() is False


def test_is_compile_supported_matches_test_suite_flag() -> None:
    # `test/utils.py`'s `DYNAMO_SUPPORTED` is just this function, called
    # once at import time; keep the two in sync so a future change to one
    # cannot silently drift from the other.
    assert is_compile_supported() is DYNAMO_SUPPORTED


@pytest.mark.skipif(not DYNAMO_SUPPORTED, reason=DYNAMO_UNSUPPORTED_REASON)
def test_is_compiling_inside_torch_compile_fullgraph() -> None:
    torch._dynamo.reset()

    def f(x: torch.Tensor) -> torch.Tensor:
        # `is_compiling()` itself is the thing under test, so its result is
        # smuggled out as a tensor rather than a Python `bool` return value
        # (returning a `bool` straight from a compiled function is not
        # representative of how the helper is actually used elsewhere in
        # this package, always as a guard around some tensor computation).
        flag = torch.tensor(1.0 if is_compiling() else 0.0)
        return x + flag

    x = torch.zeros(3)
    eager_result = f(x)
    compiled_result = run_compiled_or_skip(f, x)

    assert torch.equal(eager_result, torch.zeros(3))
    assert torch.equal(compiled_result, torch.ones(3))
