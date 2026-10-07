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
Test the shared ``is_compiling`` tracing-state helper, the
``is_compile_supported`` capability probe (through the ``requires_compile``
marker built on it) and the ``compile_fullgraph`` wrapper directly. Their callers
(``math/einsum.py``, ``storch/elemental.py``) exercise them only
indirectly, through their own compile tests.
"""

from __future__ import annotations

import pytest
import torch

from tad_mctc.tools import compile as compile_module
from tad_mctc.tools import (
    compile_fullgraph,
    get_compile_backend,
    has_cxx_compiler,
    is_compile_supported,
    is_compiling,
)
from tad_mctc.tools.testing import COMPILE_UNSUPPORTED_REASON, requires_compile

from ..utils import run_compiled_or_skip


def test_is_compiling_outside_compile() -> None:
    assert is_compiling() is False


def test_requires_compile_skips_without_compile_support() -> None:
    assert requires_compile.args == (not is_compile_supported(),)
    assert requires_compile.kwargs["reason"] == COMPILE_UNSUPPORTED_REASON


@requires_compile
@pytest.mark.usefixtures("reset_dynamo")
def test_is_compiling_inside_torch_compile_fullgraph() -> None:
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


@pytest.mark.parametrize(
    "platform,found,expected",
    [
        ("linux", {"g++"}, True),
        ("linux", {"cl"}, False),
        ("win32", {"cl"}, True),
        ("win32", {"g++"}, False),
    ],
)
def test_has_cxx_compiler(
    monkeypatch: pytest.MonkeyPatch,
    platform: str,
    found: set[str],
    expected: bool,
) -> None:
    monkeypatch.setattr(compile_module.sys, "platform", platform)
    monkeypatch.setattr(
        compile_module.shutil,
        "which",
        lambda name: f"/usr/bin/{name}" if name in found else None,
    )
    assert has_cxx_compiler() is expected


@pytest.mark.parametrize(
    "compiler,expected", [(True, "inductor"), (False, "aot_eager")]
)
def test_get_compile_backend(
    monkeypatch: pytest.MonkeyPatch, compiler: bool, expected: str
) -> None:
    monkeypatch.setattr(compile_module, "has_cxx_compiler", lambda: compiler)
    assert get_compile_backend() == expected


@requires_compile
@pytest.mark.usefixtures("reset_dynamo")
def test_compile_fullgraph() -> None:
    def f(x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x) * 2.0

    x = torch.linspace(0.0, 1.0, 5)
    compiled = compile_fullgraph(f, backend="aot_eager")
    assert torch.equal(compiled(x), f(x))


@requires_compile
@pytest.mark.usefixtures("reset_dynamo")
def test_compile_fullgraph_default_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without `backend`, `get_compile_backend()` chooses it."""
    monkeypatch.setattr(
        compile_module, "get_compile_backend", lambda: "aot_eager"
    )

    def f(x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x) * 2.0

    x = torch.linspace(0.0, 1.0, 5)
    assert torch.equal(compile_fullgraph(f)(x), f(x))


@requires_compile
@pytest.mark.usefixtures("reset_dynamo")
def test_compile_fullgraph_graph_break() -> None:
    def f(x: torch.Tensor) -> torch.Tensor:
        torch._dynamo.graph_break()
        return x + 1.0

    compiled = compile_fullgraph(f, backend="aot_eager")
    with pytest.raises(Exception):
        compiled(torch.zeros(3))
