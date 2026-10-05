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
Tools: Testing
--------------

`pytest` helpers shared by the test suites of the "tad-*" packages.

This module imports `pytest`, which is an optional dependency, so
:mod:`tad_mctc.tools` does not import it; import it explicitly. Register the
fixtures by importing them into a test suite's root ``conftest.py``::

    from tad_mctc.tools.testing import fixture_reset_dynamo  # noqa: F401
"""

from __future__ import annotations

from collections.abc import Generator

import pytest
import torch

from .compile import is_compile_supported

__all__ = [
    "COMPILE_UNSUPPORTED_REASON",
    "fixture_reset_dynamo",
    "requires_compile",
]


COMPILE_UNSUPPORTED_REASON = (
    "torch.compile/Dynamo is not supported on this Python/PyTorch combination"
)
"""Skip reason of :data:`requires_compile`."""

requires_compile = pytest.mark.skipif(
    not is_compile_supported(), reason=COMPILE_UNSUPPORTED_REASON
)
"""Skip marker for tests that call ``torch.compile``."""


@pytest.fixture(name="reset_dynamo")
def fixture_reset_dynamo() -> Generator[None, None, None]:
    """Reset ``torch.compile`` state before and after a test."""
    torch._dynamo.reset()  # pylint: disable=protected-access
    yield
    torch._dynamo.reset()  # pylint: disable=protected-access
