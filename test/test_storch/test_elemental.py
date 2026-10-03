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
Test elemental safeops.
"""

from __future__ import annotations

import pytest
import torch

from tad_mctc import storch
from tad_mctc.typing import DD

from ..conftest import DEVICE
from ..utils import (
    DYNAMO_SUPPORTED,
    DYNAMO_UNSUPPORTED_REASON,
    compile_fullgraph,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_sqrt_fail(dtype: torch.dtype) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    with pytest.raises(TypeError):
        storch.safe_sqrt(torch.tensor([1, 2, 3], **dd), eps=str(0))  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        storch.safe_sqrt(torch.tensor([-1, 2, 3], **dd), eps=-2)


@pytest.mark.skipif(not DYNAMO_SUPPORTED, reason=DYNAMO_UNSUPPORTED_REASON)
def test_sqrt_torch_compile_fullgraph_default_eps() -> None:
    """
    Representative case: with the default ``eps`` (built internally via
    ``get_eps``), ``torch.compile(fullgraph=True)`` must be able to trace
    ``storch.safe_sqrt`` without graph breaks. A domain check such as
    ``if eps < 0.0:`` reads a tensor's value at trace time, which Dynamo
    rejects as data-dependent control flow under ``fullgraph=True``, so it
    is skipped while compiling.
    """
    torch._dynamo.reset()

    x = torch.tensor([-1.0, 2.0, 3.0], dtype=torch.float64, device=DEVICE)

    def f(x: torch.Tensor) -> torch.Tensor:
        return storch.safe_sqrt(x)

    compiled = compile_fullgraph(f)

    eager_value = f(x)
    compiled_value = compiled(x)

    assert pytest.approx(eager_value.cpu()) == compiled_value.cpu()


@pytest.mark.skipif(not DYNAMO_SUPPORTED, reason=DYNAMO_UNSUPPORTED_REASON)
def test_sqrt_torch_compile_fullgraph_negative_eps() -> None:
    """
    A negative ``eps`` reaches the domain check's ``raise`` path. Eager mode
    must reject it with a ``ValueError``, but
    ``torch.compile(fullgraph=True)`` must still be able to trace the call:
    the domain check is skipped while compiling, the same dispatch shape
    ``math.einsum`` uses.
    """
    torch._dynamo.reset()

    x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, device=DEVICE)
    negative_eps = torch.tensor(-2.0, dtype=torch.float64, device=DEVICE)

    # eager mode: domain check still raises
    with pytest.raises(ValueError):
        storch.safe_sqrt(x, eps=negative_eps)

    def f(x: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        return storch.safe_sqrt(x, eps=eps)

    compiled = compile_fullgraph(f)

    compiled_value = compiled(x, negative_eps)
    assert (torch.isnan(compiled_value) == False).all()


@pytest.mark.skipif(not DYNAMO_SUPPORTED, reason=DYNAMO_UNSUPPORTED_REASON)
def test_sqrt_torch_compile_fullgraph_python_negative_eps_raises() -> None:
    """
    Unlike a ``Tensor`` ``eps``, a negative Python ``eps`` is a constant to
    Dynamo and must still be rejected under
    ``torch.compile(fullgraph=True)``, see
    ``test_pow_torch_compile_fullgraph_python_eps_zero_raises``.
    """
    torch._dynamo.reset()

    x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, device=DEVICE)

    def f(x: torch.Tensor) -> torch.Tensor:
        return storch.safe_sqrt(x, eps=-2.0)

    compiled = compile_fullgraph(f)

    from torch._dynamo.exc import Unsupported

    with pytest.raises((ValueError, Unsupported)):
        compiled(x)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_sqrt(dtype: torch.dtype) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    x = torch.tensor([-1, 2, 3], **dd)

    out = storch.safe_sqrt(x)
    assert (torch.isnan(out) == False).all()

    out = storch.safe_sqrt(x, eps=0.1)
    assert (torch.isnan(out) == False).all()

    out = storch.safe_sqrt(x, eps=0)
    assert (torch.isnan(out) == False).all()

    out = storch.safe_sqrt(x, eps=torch.tensor(torch.finfo(dtype).eps))
    assert (torch.isnan(out) == False).all()


###############################################################################


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_divide_fail(dtype: torch.dtype) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}
    x = torch.tensor([1, 2, 3], **dd)
    y = torch.tensor([1, 2, 3], **dd)

    with pytest.raises(TypeError):
        storch.safe_divide(x, y, eps="0")  # type: ignore[arg-type]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_divide(dtype: torch.dtype) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    x = torch.tensor([-1, 2, 3], **dd)
    y = torch.tensor([0, 0, 3], **dd)

    out = storch.safe_divide(x, y)
    assert (torch.isnan(out) == False).all()

    out = storch.safe_divide(x, y, eps=0.1)
    assert (torch.isnan(out) == False).all()

    out = storch.safe_divide(x, y, eps=0)
    assert (torch.isnan(out) == False).all()

    out = storch.safe_divide(x, y, eps=torch.tensor(torch.finfo(dtype).eps))
    assert (torch.isnan(out) == False).all()


###############################################################################


def test_pow_fail() -> None:
    dd: DD = {"device": DEVICE, "dtype": torch.float32}
    x = torch.tensor([1, 2, 3], **dd)

    with pytest.raises(TypeError):
        storch.safe_pow(x, 2, eps="0")  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        storch.safe_pow(x, 2, eps=0)

    with pytest.raises(ValueError):
        storch.safe_pow(x, 2, eps=torch.tensor(0.0))

    with pytest.raises(ValueError):
        storch.safe_pow(x, "2")  # type: ignore[arg-type]


@pytest.mark.skipif(not DYNAMO_SUPPORTED, reason=DYNAMO_UNSUPPORTED_REASON)
def test_pow_torch_compile_fullgraph_eps_zero() -> None:
    """
    ``eps == 0`` is the validation branch, the same shape as
    ``safe_sqrt``'s domain check: guarded behind
    ``is_compiling()`` so eager mode still raises a ``ValueError`` and
    ``torch.compile(fullgraph=True)`` can trace the call instead of
    rejecting the tensor-valued ``if (eps == 0).any():`` as data-dependent
    control flow.
    """
    torch._dynamo.reset()

    x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, device=DEVICE)
    zero_eps = torch.tensor(0.0, dtype=torch.float64, device=DEVICE)

    # eager mode: validation still raises
    with pytest.raises(ValueError):
        storch.safe_pow(x, 2, eps=zero_eps)

    def f(x: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        return storch.safe_pow(x, 2, eps=eps)

    compiled = compile_fullgraph(f)

    compiled_value = compiled(x, zero_eps)
    assert (torch.isnan(compiled_value) == False).all()


@pytest.mark.skipif(not DYNAMO_SUPPORTED, reason=DYNAMO_UNSUPPORTED_REASON)
def test_pow_torch_compile_fullgraph_python_eps_zero_raises() -> None:
    """
    A Python ``eps`` is a constant to Dynamo, so its check is no
    data-dependent control flow and must still reject ``eps=0`` under
    ``torch.compile(fullgraph=True)``. Dynamo reports a ``raise`` in a
    fullgraph trace as ``Unsupported``, whose message carries the
    ``ValueError`` text only on newer torch versions.
    """
    torch._dynamo.reset()

    x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, device=DEVICE)

    def f(x: torch.Tensor) -> torch.Tensor:
        return storch.safe_pow(x, 2, eps=0.0)

    compiled = compile_fullgraph(f)

    from torch._dynamo.exc import Unsupported

    with pytest.raises((ValueError, Unsupported)):
        compiled(x)


@pytest.mark.skipif(not DYNAMO_SUPPORTED, reason=DYNAMO_UNSUPPORTED_REASON)
def test_pow_torch_compile_fullgraph_scalar_exponent() -> None:
    """
    A Python scalar exponent never reaches the ``Tensor``-exponent branch,
    but it does pass the ``eps == 0`` validation. The scalar-exponent path
    must fullgraph-compile and match eager.
    """
    torch._dynamo.reset()

    x = torch.tensor([-1.0, 0.0, 2.0], dtype=torch.float64, device=DEVICE)

    def f(x: torch.Tensor) -> torch.Tensor:
        return storch.safe_pow(x, 2)

    compiled = compile_fullgraph(f)

    eager_value = f(x)
    compiled_value = compiled(x)

    assert torch.equal(eager_value, compiled_value)


@pytest.mark.skipif(not DYNAMO_SUPPORTED, reason=DYNAMO_UNSUPPORTED_REASON)
def test_pow_torch_compile_fullgraph_tensor_exponent() -> None:
    """
    The ``Tensor``-exponent branch: ``(exponent > 0).all() & (x >= 0).all()``
    chooses between two full sub-computations (the untouched fast path and
    the eps-clamped slow path) that genuinely disagree at ``x == 0``. The
    choice is branchless (select the *base* tensor via ``torch.where`` on
    that scalar predicate, then call ``torch.pow`` exactly once), so it
    compiles under ``fullgraph=True`` while reproducing the eager result
    exactly, including the ``x == 0`` boundary.
    """
    torch._dynamo.reset()

    x = torch.tensor([0.0, 1.0, -2.0], dtype=torch.float64, device=DEVICE)
    exponent = torch.tensor([2.0, 3.0, 0.5], dtype=torch.float64, device=DEVICE)

    def f(x: torch.Tensor, exponent: torch.Tensor) -> torch.Tensor:
        return storch.safe_pow(x, exponent)

    compiled = compile_fullgraph(f)

    eager_value = f(x, exponent)
    compiled_value = compiled(x, exponent)

    assert torch.equal(eager_value, compiled_value)


@pytest.mark.skipif(not DYNAMO_SUPPORTED, reason=DYNAMO_UNSUPPORTED_REASON)
def test_pow_torch_compile_fullgraph_tensor_exponent_fast_path() -> None:
    """
    Same as above but exercising the fast-path branch specifically (all
    exponents positive, all ``x`` non-negative, including an exact zero),
    where the result must stay exactly ``torch.pow``'s.
    """
    torch._dynamo.reset()

    x = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64, device=DEVICE)
    exponent = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64, device=DEVICE)

    def f(x: torch.Tensor, exponent: torch.Tensor) -> torch.Tensor:
        return storch.safe_pow(x, exponent)

    compiled = compile_fullgraph(f)

    eager_value = f(x, exponent)
    compiled_value = compiled(x, exponent)

    assert torch.equal(eager_value, compiled_value)


def test_pow_tensor_exponent_zero_x_matches_fast_path() -> None:
    """
    With an all-positive ``Tensor`` exponent
    and ``x`` containing an exact zero, the fast path
    (``torch.pow(x, exponent)`` untouched) must be taken, not the
    eps-substituting slow path -- the slow path would turn ``0 ** positive``
    into a small nonzero value (``eps ** exponent``) instead of leaving it at
    exactly ``0``. It is checked against ``torch.pow`` directly (the oracle
    for the fast-path region) rather than against a hardcoded expected
    value.
    """
    x = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64, device=DEVICE)
    exponent = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64, device=DEVICE)

    out = storch.safe_pow(x, exponent)

    assert torch.equal(out, torch.pow(x, exponent))
    assert (out[x == 0] == 0).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("xlist", [[-1, 0, 2], [1, 2, 3]])
def test_pow(dtype: torch.dtype, xlist: list[int]) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}
    x = torch.tensor(xlist, **dd)

    assert (torch.pow(x, 2) == storch.safe_pow(x, 2)).all()

    # positive integer exponents
    out = storch.safe_pow(x, 2, eps=torch.tensor(torch.finfo(dtype).eps))
    assert (torch.isnan(out) == False).all()

    # positive integer exponents with float
    out = storch.safe_pow(x, 2.0)
    assert (torch.isnan(out) == False).all()

    # positive integer scalar tensor exponents
    out = storch.safe_pow(x, torch.tensor(1, **dd))
    assert (torch.isnan(out) == False).all()

    # positive integer tensor exponents
    out = storch.safe_pow(x, torch.tensor([1, 2, 3], **dd))
    assert (torch.isnan(out) == False).all()

    # negative integer exponents
    out = storch.safe_pow(x, -2)
    assert (torch.isnan(out) == False).all()

    # negative integer exponents with float
    out = storch.safe_pow(x, -2.0)
    assert (torch.isnan(out) == False).all()

    # negative integer scalar tensor exponents
    out = storch.safe_pow(x, torch.tensor(-1, **dd))
    assert (torch.isnan(out) == False).all()

    # negative integer tensor exponents
    out = storch.safe_pow(x, torch.tensor([-1, -2, -3], **dd))
    assert (torch.isnan(out) == False).all()

    # positive fractional exponents
    out = storch.safe_pow(x, 0.5)
    assert (torch.isnan(out) == False).all()

    # positive fractional scalar tensor exponents
    out = storch.safe_pow(x, torch.tensor(0.5, **dd))
    assert (torch.isnan(out) == False).all()

    # positive fractional tensor exponents
    out = storch.safe_pow(x, torch.tensor([0.5, 1, 2], **dd))
    assert (torch.isnan(out) == False).all()

    # negative fractional exponents
    out = storch.safe_pow(x, -0.5)
    assert (torch.isnan(out) == False).all()

    # negative fractional scalar tensor exponents
    out = storch.safe_pow(x, torch.tensor(-0.5, **dd))
    assert (torch.isnan(out) == False).all()

    # negative fractional tensor exponents
    out = storch.safe_pow(x, torch.tensor([-0.5, -1, -2], **dd))
    assert (torch.isnan(out) == False).all()

    # zero exponents
    out = storch.safe_pow(x, 0, eps=1.0e-10)
    assert (torch.isnan(out) == False).all()

    out = storch.safe_pow(x, torch.tensor(0, **dd))
    assert (torch.isnan(out) == False).all()

    out = storch.safe_pow(x, torch.tensor([0, 0, 0], **dd), eps=999)
    assert (torch.isnan(out) == False).all()


###############################################################################


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_reciprocal_fail(dtype: torch.dtype) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}
    x = torch.tensor([1, 2, 3], **dd)

    with pytest.raises(TypeError):
        storch.safe_reciprocal(x, eps="0")  # type: ignore[arg-type]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_reciprocal(dtype: torch.dtype) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    x = torch.tensor([-1, 2, 3], **dd)

    out = storch.safe_reciprocal(x)
    assert (torch.isnan(out) == False).all()

    out = storch.safe_reciprocal(x, eps=0.1)
    assert (torch.isnan(out) == False).all()

    out = storch.safe_reciprocal(x, eps=0)
    assert (torch.isnan(out) == False).all()

    out = storch.safe_reciprocal(x, eps=torch.tensor(torch.finfo(dtype).eps))
    assert (torch.isnan(out) == False).all()


# `torch.pow` with a scalar and with a tensor exponent can differ in the last
# bit (different kernels, e.g. on macOS), so parity checks use a tolerance.
RTOL = {torch.float32: 1e-6, torch.float64: 1e-14}


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("exponent", [-3, -2, -1, 0, 1, 2, 3])
def test_pow_tensor_whole_exponent_matches_int_exponent(
    dtype: torch.dtype, exponent: int
) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}
    x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0], **dd)

    expected = storch.safe_pow(x, exponent)
    out = storch.safe_pow(x, torch.tensor(float(exponent), **dd))

    assert torch.allclose(out, expected, rtol=RTOL[dtype], atol=0)
    assert not torch.isnan(out).any()


def test_pow_tensor_exponent_negative_base_whole() -> None:
    x = torch.tensor([-2.0], dtype=torch.float64)

    out = storch.safe_pow(x, torch.tensor(-1.0, dtype=torch.float64))
    assert torch.allclose(out, torch.tensor([-0.5], dtype=torch.float64))


@pytest.mark.parametrize("exponent", [-3, -2, -1, 0, 1, 2, 3])
@pytest.mark.parametrize("xval", [-2.0, 0.0, 2.0])
def test_pow_tensor_whole_exponent_gradient_matches_int_exponent(
    exponent: int, xval: float
) -> None:
    dd: DD = {"device": DEVICE, "dtype": torch.float64}

    x1 = torch.tensor([xval], requires_grad=True, **dd)
    (g_scalar,) = torch.autograd.grad(storch.safe_pow(x1, exponent).sum(), x1)

    x2 = torch.tensor([xval], requires_grad=True, **dd)
    p = torch.tensor(float(exponent), **dd)
    (g_tensor,) = torch.autograd.grad(storch.safe_pow(x2, p).sum(), x2)

    assert torch.allclose(g_tensor, g_scalar, rtol=1e-14, atol=0)
    assert not torch.isnan(g_tensor).any()


def test_pow_tensor_exponent_one_has_unit_gradient_at_zero() -> None:
    dd: DD = {"device": DEVICE, "dtype": torch.float64}
    x = torch.tensor([0.0], requires_grad=True, **dd)

    (grad,) = torch.autograd.grad(
        storch.safe_pow(x, torch.tensor(1.0, **dd)).sum(), x
    )
    assert torch.equal(grad, torch.ones_like(grad))


def test_pow_tensor_exponent_gradcheck_negative_base() -> None:
    dd: DD = {"device": DEVICE, "dtype": torch.float64}
    x = torch.tensor([-2.0, -1.5], requires_grad=True, **dd)
    p = torch.tensor([3.0, 2.0], **dd)

    assert torch.autograd.gradcheck(storch.safe_pow, (x, p))

    # Finite differences cannot check the gradient with respect to the
    # exponent: a negative base is only defined for whole exponents, which
    # a perturbed exponent is not. Compare with the analytic value
    # d/dp (sign * |x|**p) = sign * |x|**p * log|x| instead.
    p = torch.tensor([3.0, 2.0], requires_grad=True, **dd)
    x = torch.tensor([-2.0, -1.5], **dd)
    (grad,) = torch.autograd.grad(storch.safe_pow(x, p).sum(), p)

    sign = torch.tensor([-1.0, 1.0], **dd)
    expected = sign * x.abs() ** p.detach() * x.abs().log()
    assert torch.allclose(grad, expected)


@pytest.mark.parametrize("xlist", [[-2.0], [-0.5, 0.0, 0.5], [2.0, 3.0]])
def test_pow_mixed_exponent_tensor(xlist: list[float]) -> None:
    dd: DD = {"device": DEVICE, "dtype": torch.float64}
    x = torch.tensor(xlist, **dd).unsqueeze(-1)
    exponents = [-3.0, -0.5, 0.0, 1.0, 1.5, 2.0]
    p = torch.tensor(exponents, **dd)

    out = storch.safe_pow(x, p)
    assert out.shape == (len(xlist), len(exponents))
    assert not torch.isnan(out).any()

    # each column equals the result for that exponent alone
    for i, e in enumerate(exponents):
        single = storch.safe_pow(x[:, 0], torch.tensor(e, **dd))
        assert torch.allclose(out[:, i], single, rtol=1e-14, atol=0)


@pytest.mark.parametrize("bad", [0, 0.0, -1e-6, float("nan")])
def test_pow_rejects_nonpositive_or_nan_eps_number(bad: float) -> None:
    x = torch.tensor([1.0, 2.0])
    with pytest.raises(ValueError, match="larger than 0"):
        storch.safe_pow(x, 2, eps=bad)


@pytest.mark.parametrize("bad", [0.0, -1e-6, float("nan")])
def test_pow_rejects_nonpositive_or_nan_eps_tensor(bad: float) -> None:
    x = torch.tensor([1.0, 2.0])
    with pytest.raises(ValueError, match="larger than 0"):
        storch.safe_pow(x, 0.5, eps=torch.tensor([1e-6, bad]))


@pytest.mark.parametrize("exponent", [-0.5, 0.5, 1.5])
@pytest.mark.parametrize("xval", [-2.0, 0.0, 2.0])
def test_pow_scalar_and_tensor_fractional_exponent_agree(
    exponent: float, xval: float
) -> None:
    dd: DD = {"device": DEVICE, "dtype": torch.float64}

    x1 = torch.tensor([xval], requires_grad=True, **dd)
    out_scalar = storch.safe_pow(x1, exponent)
    (g_scalar,) = torch.autograd.grad(out_scalar.sum(), x1)

    x2 = torch.tensor([xval], requires_grad=True, **dd)
    out_tensor = storch.safe_pow(x2, torch.tensor(exponent, **dd))
    (g_tensor,) = torch.autograd.grad(out_tensor.sum(), x2)

    # `torch.pow` with a scalar and with a tensor exponent can differ in the
    # last bit (different kernels), so compare with a tight tolerance
    assert torch.allclose(out_tensor, out_scalar, rtol=1e-14, atol=0)
    assert torch.allclose(g_tensor, g_scalar, rtol=1e-14, atol=0)
    assert not torch.isnan(g_tensor).any()


def test_pow_zero_base_positive_fractional_exponent_is_zero() -> None:
    dd: DD = {"device": DEVICE, "dtype": torch.float64}
    x = torch.zeros(1, **dd)

    assert storch.safe_pow(x, 0.5).item() == 0.0
    assert storch.safe_pow(x, torch.tensor(0.5, **dd)).item() == 0.0


def test_pow_default_eps_is_not_validated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default `eps` (and a number) must not read tensor values, which
    would synchronize with the device on every call."""
    calls: list[int] = []
    original = torch.Tensor.all

    def spy(self: torch.Tensor, *args, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "all", spy)

    x = torch.tensor([1.0, 2.0])
    storch.safe_pow(x, 0.5)
    storch.safe_pow(x, 0.5, eps=1e-6)
    assert calls == []

    storch.safe_pow(x, 0.5, eps=torch.tensor(1e-6))
    assert calls == [1]
