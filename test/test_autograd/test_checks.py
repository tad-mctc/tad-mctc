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
import pytest
import torch

from tad_mctc._version import __tversion__
from tad_mctc.autograd import is_batched, is_functorch_tensor, is_gradtracking


def test_is_gradtracking_true(monkeypatch):
    """Should return True when torch._C._functorch.is_gradtrackingtensor is True."""
    dummy = object()
    monkeypatch.setattr(
        torch._C._functorch,
        "is_gradtrackingtensor",
        lambda x: True,
    )
    assert is_gradtracking(dummy) is True  # type: ignore[arg-type]


def test_is_gradtracking_false(monkeypatch):
    """Should return False when torch._C._functorch.is_gradtrackingtensor is False."""
    dummy = object()
    monkeypatch.setattr(
        torch._C._functorch,
        "is_gradtrackingtensor",
        lambda x: False,
    )
    assert is_gradtracking(dummy) is False  # type: ignore[arg-type]


def test_is_batched_true(monkeypatch):
    """Should return True when torch._C._functorch.is_batchedtensor is True."""
    dummy = object()
    monkeypatch.setattr(
        torch._C._functorch,
        "is_batchedtensor",
        lambda x: True,
    )
    assert is_batched(dummy) is True  # type: ignore[arg-type]


def test_is_batched_false(monkeypatch):
    """Should return False when torch._C._functorch.is_batchedtensor is False."""
    dummy = object()
    monkeypatch.setattr(
        torch._C._functorch,
        "is_batchedtensor",
        lambda x: False,
    )
    assert is_batched(dummy) is False  # type: ignore[arg-type]


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
    assert is_functorch_tensor(dummy) is expected  # type: ignore[arg-type]


###############################################################################


@pytest.mark.skipif(__tversion__ < (2, 0, 0), reason="Requires torch>=2.0.0")
def test_plain_tensor_behavior():
    # A plain torch.Tensor should not be seen as grad-tracking or batched
    t = torch.tensor([1.0, 2.0, 3.0])
    assert is_gradtracking(t) is False
    assert is_batched(t) is False
    assert is_functorch_tensor(t) is False


@pytest.mark.skipif(__tversion__ < (2, 0, 0), reason="Requires torch>=2.0.0")
def test_gradtracking_tensor_via_grad():
    # grad(f) returns a grad-tracking tensor when applied
    def f(x: torch.Tensor) -> torch.Tensor:
        assert is_gradtracking(x) is True
        assert is_batched(x) is False
        assert is_functorch_tensor(x) is True

        return x * x

    t = torch.tensor(4.0, requires_grad=True)
    _ = torch.func.jacrev(f)(t)


@pytest.mark.skipif(__tversion__ < (2, 0, 0), reason="Requires torch>=2.0.0")
def test_batched_tensor_via_vmap():
    # vmap wraps a tensor into a batched tensor
    def f(x: torch.Tensor) -> torch.Tensor:
        assert is_gradtracking(x) is False
        assert is_batched(x) is True
        assert is_functorch_tensor(x) is True

        return x * x

    t = torch.randn((2, 4), requires_grad=True)
    _ = torch.func.vmap(f)(t)


@pytest.mark.skipif(__tversion__ < (2, 0, 0), reason="Requires torch>=2.0.0")
def test_grad_and_batched_tensor():
    # Combine grad + vmap to get a tensor that is both
    def f(x):
        assert is_gradtracking(x) is True
        assert is_batched(torch._C._functorch.get_unwrapped(x)) is True
        assert is_functorch_tensor(x) is True

        return x**3

    t = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    grad_fn = torch.func.grad(f)
    _ = torch.func.vmap(grad_fn)(t)
