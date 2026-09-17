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

__all__ = ["is_compiling"]


def is_compiling() -> bool:
    """
    Whether we are currently being traced by ``torch.compile``.

    Checked by capability rather than by a ``__tversion__`` cutoff: across
    the PyTorch versions this package supports, neither the public
    ``torch.compiler.is_compiling`` nor its older, private predecessor
    ``torch._dynamo.is_compiling`` reliably exists (or resolves without
    raising) purely as a function of version. ``torch.compiler`` can exist
    without yet having ``is_compiling`` (added later than the module
    itself), and ``torch._dynamo`` -- even on a version that ships it --
    is only exposed as a ``torch`` attribute once something has imported
    it, which nothing upstream of this call is guaranteed to have done.
    Every step below is therefore guarded, and any PyTorch version older
    than ``torch.compile`` itself correctly falls through to ``False``.
    """
    try:
        import torch._dynamo as _torch_dynamo  # noqa: F401  # pylint: disable=unused-import, protected-access
    except ImportError:
        pass

    compiler = getattr(torch, "compiler", None)
    if compiler is not None and hasattr(compiler, "is_compiling"):
        return bool(compiler.is_compiling())

    dynamo = getattr(torch, "_dynamo", None)
    if dynamo is not None and hasattr(dynamo, "is_compiling"):
        return bool(dynamo.is_compiling())

    return False
