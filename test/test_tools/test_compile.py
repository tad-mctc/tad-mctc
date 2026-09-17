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
Test the shared ``is_compiling`` tracing-state helper (issue 21).

Neither of the two former private copies (``math/einsum.py``,
``storch/elemental.py``) had a dedicated test -- each was only exercised
indirectly through ``einsum``'s / ``sqrt``'s / ``pow``'s own compile tests.
Moving the code to a shared, public location is a reasonable point to add
direct coverage.
"""

from __future__ import annotations

import pytest
import torch

from tad_mctc.tools import is_compiling

from ..utils import (
    DYNAMO_SUPPORTED,
    DYNAMO_UNSUPPORTED_REASON,
    run_compiled_or_skip,
)


def test_is_compiling_outside_compile() -> None:
    assert is_compiling() is False


@pytest.mark.skipif(not DYNAMO_SUPPORTED, reason=DYNAMO_UNSUPPORTED_REASON)
def test_is_compiling_inside_torch_compile_fullgraph() -> None:
    torch._dynamo.reset()  # pylint: disable=protected-access

    def f(x: torch.Tensor) -> torch.Tensor:
        # `is_compiling()` itself is the thing under test, so its result is
        # smuggled out as a tensor rather than a Python `bool` return value
        # (returning a `bool` straight from a compiled function is not
        # representative of how the helper is actually used elsewhere in
        # this package, always as a guard around some tensor computation).
        flag = torch.tensor(1.0 if is_compiling() else 0.0)
        return x + flag

    compiled = torch.compile(f, fullgraph=True, dynamic=False)

    x = torch.zeros(3)
    eager_result = f(x)
    compiled_result = run_compiled_or_skip(compiled, x)

    assert torch.equal(eager_result, torch.zeros(3))
    assert torch.equal(compiled_result, torch.ones(3))
