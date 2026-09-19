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

from typing import Callable

import torch

__all__ = ["is_compile_supported", "is_compiling"]


def _always_false() -> bool:  # pragma: no cover
    """``torch.compile`` does not exist, or exposes no way to ask."""
    return False


def _probe_compile_supported() -> bool:
    """
    Whether ``torch.compile``/Dynamo tracing is usable at all on this
    Python/PyTorch combination -- a one-time capability probe, unlike
    :func:`is_compiling`'s per-call "are we being traced right now" query.

    ``torch._dynamo.is_dynamo_supported`` -- the query PyTorch itself uses
    to track Python-version support lag -- is itself a later addition than
    ``torch.compile``, so its own absence (e.g. PyTorch 2.0.1) is read as
    "assume supported", matching what has been observed in practice on
    those older versions.
    """
    if not hasattr(torch, "compile"):
        return False  # pragma: no cover

    try:
        import torch._dynamo as dynamo  # pylint: disable=protected-access
    except ImportError:  # pragma: no cover
        return False

    is_supported = getattr(dynamo, "is_dynamo_supported", None)
    if is_supported is None:
        return True  # pragma: no cover

    try:
        return bool(is_supported())
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


def _resolve_is_compiling() -> Callable[[], bool]:
    """
    Resolve, once, which underlying "is compiling" query this PyTorch
    offers.

    This capability probe -- ``getattr``/``hasattr`` chains, an explicit
    ``import torch._dynamo`` -- has to run here, outside of
    :func:`is_compiling`'s own body. ``is_compiling()`` is called from
    inside code (``math/einsum.py``, ``storch/elemental.py``) that itself
    gets traced under ``torch.compile(fullgraph=True)``, and Dynamo cannot
    trace ``hasattr``/``getattr`` introspection on a module object
    (``Unsupported: hasattr: PythonModuleVariable()``); it can trace a
    plain call to a resolved function just fine. Resolving once at import
    time keeps :func:`is_compiling`'s traced body down to that one call.

    Neither the public ``torch.compiler.is_compiling`` nor its older,
    private predecessor ``torch._dynamo.is_compiling`` reliably exists (or
    resolves without raising) purely as a function of ``__tversion__``:
    ``torch.compiler`` can exist without yet having ``is_compiling`` on it
    (added later than the module itself), and ``torch._dynamo`` -- even on
    a version that ships it -- is only exposed as a ``torch`` attribute
    once something has imported it, which nothing upstream of this call is
    guaranteed to have done.
    """
    try:
        import torch._dynamo as _torch_dynamo  # noqa: F401  # pylint: disable=unused-import, protected-access
    except ImportError:  # pragma: no cover
        # Only unavailable on PyTorch < 2.0, not exercised by any single
        # CI job's torch version; the other probes below fall through.
        pass

    compiler = getattr(torch, "compiler", None)
    if compiler is not None and hasattr(compiler, "is_compiling"):
        return compiler.is_compiling

    dynamo = getattr(torch, "_dynamo", None)
    if dynamo is not None and hasattr(dynamo, "is_compiling"):
        return dynamo.is_compiling

    return _always_false  # pragma: no cover


_is_compiling_impl = _resolve_is_compiling()


def is_compiling() -> bool:
    """Whether we are currently being traced by ``torch.compile``."""
    return bool(_is_compiling_impl())
