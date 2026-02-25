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
Test hessian.
"""

import importlib
from unittest.mock import patch

import pytest
import torch

from tad_mctc._version import __tversion__
from tad_mctc.autograd import checks


def test_dummy():
    import tad_mctc._version

    torch_version = tad_mctc._version.__tversion__

    with patch("tad_mctc._version.__tversion__", new=(1, 9, 0)):
        # reload cached module to ensure that patched version is used
        importlib.reload(checks)

        tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        assert checks.is_batched(tensor) is False
        assert checks.is_gradtracking(tensor) is False
        assert checks.is_functorch_tensor(tensor) is False

    # reload for actual version
    importlib.reload(checks)
    assert torch_version == tad_mctc._version.__tversion__


@pytest.mark.skipif(__tversion__ < (2, 0, 0), reason="Requires torch>=2.0.0")
def test_is_gradtracking_true(monkeypatch):
    """Should return True when torch._C._functorch.is_gradtrackingtensor is True."""
    dummy = object()
    monkeypatch.setattr(
        torch._C._functorch,
        "is_gradtrackingtensor",
        lambda x: True,
    )
    assert checks.is_gradtracking(dummy) is True  # type: ignore[arg-type]


@pytest.mark.skipif(__tversion__ < (2, 0, 0), reason="Requires torch>=2.0.0")
def test_is_gradtracking_false(monkeypatch):
    """Should return False when torch._C._functorch.is_gradtrackingtensor is False."""
    dummy = object()
    monkeypatch.setattr(
        torch._C._functorch,
        "is_gradtrackingtensor",
        lambda x: False,
    )
    assert checks.is_gradtracking(dummy) is False  # type: ignore[arg-type]


@pytest.mark.skipif(__tversion__ < (2, 0, 0), reason="Requires torch>=2.0.0")
def test_is_batched_true(monkeypatch):
    """Should return True when torch._C._functorch.is_batchedtensor is True."""
    dummy = object()
    monkeypatch.setattr(
        torch._C._functorch,
        "is_batchedtensor",
        lambda x: True,
    )
    assert checks.is_batched(dummy) is True  # type: ignore[arg-type]


@pytest.mark.skipif(__tversion__ < (2, 0, 0), reason="Requires torch>=2.0.0")
def test_is_batched_false(monkeypatch):
    """Should return False when torch._C._functorch.is_batchedtensor is False."""
    dummy = object()
    monkeypatch.setattr(
        torch._C._functorch,
        "is_batchedtensor",
        lambda x: False,
    )
    assert checks.is_batched(dummy) is False  # type: ignore[arg-type]


@pytest.mark.skipif(__tversion__ < (2, 0, 0), reason="Requires torch>=2.0.0")
@pytest.mark.parametrize(
    "grad_val, batched_val, expected",
    [
        (True, False, True),  # only grad-tracking
        (False, True, True),  # only batched
        (True, True, True),  # both
        (False, False, False),  # neither
    ],
)
def test_is_functorch_tensor(monkeypatch, grad_val, batched_val, expected):
    """
    is_functorch_tensor should return True if either grad-tracking
    or batched (or both) is True, otherwise False.
    """
    monkeypatch.setattr(
        torch._C._functorch,
        "is_gradtrackingtensor",
        lambda x: grad_val,
    )
    monkeypatch.setattr(
        torch._C._functorch,
        "is_batchedtensor",
        lambda x: batched_val,
    )
    dummy = object()
    actual = checks.is_functorch_tensor(dummy)  # type: ignore[arg-type]
    assert actual is expected


###############################################################################


@pytest.mark.skipif(__tversion__ < (2, 0, 0), reason="Requires torch>=2.0.0")
def test_plain_tensor_behavior():
    # A plain torch.Tensor should not be seen as grad-tracking or batched
    t = torch.tensor([1.0, 2.0, 3.0])
    assert checks.is_gradtracking(t) is False
    assert checks.is_batched(t) is False
    assert checks.is_functorch_tensor(t) is False


@pytest.mark.skipif(__tversion__ < (2, 0, 0), reason="Requires torch>=2.0.0")
def test_gradtracking_tensor_via_grad():
    # grad(f) returns a grad-tracking tensor when applied
    def f(x: torch.Tensor) -> torch.Tensor:
        assert checks.is_gradtracking(x) is True
        assert checks.is_batched(x) is False
        assert checks.is_functorch_tensor(x) is True

        return x * x

    t = torch.tensor(4.0, requires_grad=True)
    _ = torch.func.jacrev(f)(t)


@pytest.mark.skipif(__tversion__ < (2, 0, 0), reason="Requires torch>=2.0.0")
def test_batched_tensor_via_vmap():
    # vmap wraps a tensor into a batched tensor
    def f(x: torch.Tensor) -> torch.Tensor:
        assert checks.is_gradtracking(x) is False
        assert checks.is_batched(x) is True
        assert checks.is_functorch_tensor(x) is True

        return x * x

    t = torch.randn((2, 4), requires_grad=True)
    _ = torch.func.vmap(f)(t)


@pytest.mark.skipif(__tversion__ < (2, 0, 0), reason="Requires torch>=2.0.0")
def test_grad_and_batched_tensor():
    # Combine grad + vmap to get a tensor that is both
    def f(x):
        assert checks.is_gradtracking(x) is True
        assert checks.is_batched(torch._C._functorch.get_unwrapped(x)) is True
        assert checks.is_functorch_tensor(x) is True

        return x**3

    t = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    grad_fn = torch.func.grad(f)
    _ = torch.func.vmap(grad_fn)(t)
