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
Tools: Compile
--------------

Introspection of the current ``torch.compile`` tracing state.
"""

from __future__ import annotations

import torch

__all__ = ["is_compile_supported", "is_compiling"]


def _probe_compile_supported() -> bool:
    """
    Whether ``torch.compile``/Dynamo tracing is usable on this Python/PyTorch
    combination -- a one-time capability probe, unlike :func:`is_compiling`'s
    per-call "are we being traced right now" query.
    """
    import torch._dynamo as dynamo  # pylint: disable=protected-access

    try:
        return bool(dynamo.is_dynamo_supported())
    except Exception:  # pragma: no cover  # pylint: disable=broad-except
        return False


_compile_supported = _probe_compile_supported()


def is_compile_supported() -> bool:
    """
    Whether ``torch.compile``/Dynamo tracing is usable in this environment.

    Unlike :func:`is_compiling`, this is not safe to call from code that
    is itself traced under ``torch.compile(fullgraph=True)`` -- it is meant
    for deciding, ahead of time, whether to attempt compiling at all.
    """
    return _compile_supported


def is_compiling() -> bool:
    """Whether we are currently being traced by ``torch.compile``."""
    return torch.compiler.is_compiling()
