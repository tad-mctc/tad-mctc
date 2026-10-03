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
Test the checks for function-transformed tensors.
"""

from collections.abc import Callable

import pytest
import torch

from tad_mctc.autograd import checks

from ..utils import (
    DYNAMO_SUPPORTED,
    DYNAMO_UNSUPPORTED_REASON,
    compile_fullgraph,
)


def test_is_gradtracking_true(monkeypatch: pytest.MonkeyPatch) -> None:
    """Should return True when torch._C._functorch.is_gradtrackingtensor is True."""
    dummy = object()
    monkeypatch.setattr(
        torch._C._functorch,
        "is_gradtrackingtensor",
        lambda x: True,
    )
    assert checks.is_gradtracking(dummy) is True  # type: ignore[arg-type]


def test_is_gradtracking_false(monkeypatch: pytest.MonkeyPatch) -> None:
    """Should return False when torch._C._functorch.is_gradtrackingtensor is False."""
    dummy = object()
    monkeypatch.setattr(
        torch._C._functorch,
        "is_gradtrackingtensor",
        lambda x: False,
    )
    assert checks.is_gradtracking(dummy) is False  # type: ignore[arg-type]


###############################################################################


def test_plain_tensor_behavior() -> None:
    # A plain torch.Tensor should not be seen as grad-tracking or batched
    t = torch.tensor([1.0, 2.0, 3.0])
    assert checks.is_gradtracking(t) is False
    assert checks.is_vmapped(t) is False
    assert checks.is_functorch_tensor(t) is False


def test_gradtracking_tensor_via_grad() -> None:
    # grad(f) returns a grad-tracking tensor when applied
    def f(x: torch.Tensor) -> torch.Tensor:
        assert checks.is_gradtracking(x) is True
        assert checks.is_vmapped(x) is False
        assert checks.is_functorch_tensor(x) is True

        return x * x

    t = torch.tensor(4.0, requires_grad=True)
    _ = torch.func.jacrev(f)(t)  # pyright: ignore[reportPrivateImportUsage]


def test_batched_tensor_via_vmap() -> None:
    # vmap wraps a tensor into a batched tensor
    def f(x: torch.Tensor) -> torch.Tensor:
        assert checks.is_gradtracking(x) is False
        assert checks.is_vmapped(x) is True
        assert checks.is_functorch_tensor(x) is True

        return x * x

    t = torch.randn((2, 4), requires_grad=True)
    _ = torch.func.vmap(f)(t)  # pyright: ignore[reportPrivateImportUsage]


def test_grad_and_batched_tensor() -> None:
    # Combine grad + vmap to get a tensor that is both
    def f(x: torch.Tensor) -> torch.Tensor:
        assert checks.is_gradtracking(x) is True
        # found below the grad-tracking layer
        assert checks.is_vmapped(x) is True
        assert checks.is_functorch_tensor(x) is True

        return x**3

    t = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    grad_fn = torch.func.grad(f)  # pyright: ignore[reportPrivateImportUsage]
    _ = torch.func.vmap(grad_fn)(t)  # pyright: ignore[reportPrivateImportUsage]


@pytest.mark.parametrize(
    "transform, functorch, vmapped",
    [
        ("eager", False, False),
        ("jacrev", True, False),
        ("jacrev(jacrev)", True, False),
        ("vmap", True, True),
        ("vmap(jacrev)", True, True),
        ("vmap(jacrev(jacrev))", True, True),
    ],
)
def test_vmapped_vs_functorch_under_transforms(
    transform: str, functorch: bool, vmapped: bool
) -> None:
    """
    `is_functorch_tensor` is true under any transform, `is_vmapped` only if a
    `vmap` is active, even below grad-tracking layers. `jacrev` also wraps
    the arguments it does not differentiate (here: `n`).
    """
    seen: list[tuple[bool, bool]] = []

    def f(n: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        seen.append((checks.is_functorch_tensor(n), checks.is_vmapped(n)))
        return p.sum()

    numbers = torch.tensor([[1, 1], [6, 1]])
    positions = torch.rand(2, 2, 3, dtype=torch.double)
    jac = torch.func.jacrev  # pyright: ignore[reportPrivateImportUsage]
    vmap = torch.func.vmap  # pyright: ignore[reportPrivateImportUsage]

    if transform == "eager":
        f(numbers[0], positions[0])
    elif transform == "jacrev":
        jac(f, argnums=1)(numbers[0], positions[0])
    elif transform == "jacrev(jacrev)":
        jac(jac(f, argnums=1), argnums=1)(numbers[0], positions[0])
    elif transform == "vmap":
        vmap(f)(numbers, positions)
    elif transform == "vmap(jacrev)":
        vmap(jac(f, argnums=1))(numbers, positions)
    else:
        vmap(jac(jac(f, argnums=1), argnums=1))(numbers, positions)

    assert seen == [(functorch, vmapped)]


@pytest.mark.skipif(not DYNAMO_SUPPORTED, reason=DYNAMO_UNSUPPORTED_REASON)
@pytest.mark.parametrize(
    "check",
    [checks.is_gradtracking, checks.is_vmapped, checks.is_functorch_tensor],
)
def test_true_under_compile(check: Callable[[torch.Tensor], bool]) -> None:
    """
    The checks trace under `torch.compile(fullgraph=True)` (the functorch
    bindings they call cannot) and report the safe answer, `True`.

    Calls the compiled function directly instead of `run_compiled_or_skip`,
    so that a graph break fails the test instead of skipping it. Both
    branches compute something: a frame whose graph would be empty (e.g.
    just returning `x`) is run eagerly by older PyTorch even under
    `fullgraph=True`, which would hide the traced result.
    """
    torch._dynamo.reset()  # pylint: disable=protected-access

    def f(x: torch.Tensor) -> torch.Tensor:
        return x + (1.0 if check(x) else 0.0)

    x = torch.zeros(3)
    assert torch.equal(compile_fullgraph(f)(x), torch.ones(3))
    assert torch.equal(f(x), torch.zeros(3))


@pytest.mark.parametrize(
    "check",
    [
        checks.is_gradtracking,
        checks.is_vmapped,
        checks.is_functorch_tensor,
    ],
)
def test_checks_true_while_compiling(
    monkeypatch: pytest.MonkeyPatch, check: Callable[[torch.Tensor], bool]
) -> None:
    """While tracing, every check is `True` without touching functorch."""
    monkeypatch.setattr(checks, "is_compiling", lambda: True)
    assert check(torch.zeros(3)) is True
