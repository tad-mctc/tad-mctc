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

Introspection of the current ``torch.compile`` tracing state, and a
``torch.compile`` wrapper that checks a function traces as one graph.
"""

from __future__ import annotations

import shutil
import sys
from collections.abc import Callable
from typing import Any

import torch

__all__ = [
    "compile_fullgraph",
    "get_compile_backend",
    "has_cxx_compiler",
    "is_compile_supported",
    "is_compiling",
]


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


def has_cxx_compiler() -> bool:
    """
    Whether the C++ compiler that TorchInductor calls is on ``PATH``.

    On Windows, that is MSVC's ``cl``, which a plain CI runner does not
    expose (``InvalidCxxCompiler: Compiler: cl is not found``).
    """
    names = ["cl"] if sys.platform == "win32" else ["c++", "g++", "clang++"]
    return any(shutil.which(name) is not None for name in names)


def get_compile_backend() -> str:
    """
    The ``torch.compile`` backend usable in this environment.

    ``"inductor"`` if a C++ compiler is available (:func:`has_cxx_compiler`),
    otherwise ``"aot_eager"``, which still traces with Dynamo and runs
    AOTAutograd, just without generating C++ code.
    """
    return "inductor" if has_cxx_compiler() else "aot_eager"


def compile_fullgraph(
    fn: Callable[..., Any],
    *,
    dynamic: bool = False,
    backend: str | None = None,
) -> Callable[..., Any]:
    """
    ``torch.compile(fn)`` as one graph (``fullgraph=True``).

    Dynamo decides whether ``fn`` traces as one graph before any backend
    runs, so the check is the same on every backend.

    Parameters
    ----------
    fn : Callable[..., Any]
        Function to compile.
    dynamic : bool, optional
        Whether to compile with dynamic shapes. Defaults to ``False``.
    backend : str | None, optional
        The ``torch.compile`` backend. ``None`` (default) means
        :func:`get_compile_backend`.

    Returns
    -------
    Callable[..., Any]
        The compiled function.
    """
    if backend is None:
        backend = get_compile_backend()
    return torch.compile(fn, fullgraph=True, dynamic=dynamic, backend=backend)
