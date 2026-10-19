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
Test `attach_gradient`: values with a known, differentiable gradient.
"""

from __future__ import annotations

import itertools

import pytest
import torch
from torch.autograd import forward_ad
from torch.func import jacfwd, jacrev, vmap

from tad_mctc._version import __tversion__
from tad_mctc.autograd import attach_gradient
from tad_mctc.tools.testing import requires_compile
from tad_mctc.typing import Tensor

DD = {"dtype": torch.double}


def _lse(x: Tensor) -> Tensor:
    return torch.logsumexp(x, dim=-1)


def _attached(x: Tensor) -> Tensor:
    # the value is computed without autograd: no graph through `_lse`
    return attach_gradient(_lse, x, torch.softmax(x, dim=-1))


def _softplus(x: Tensor) -> Tensor:
    return torch.nn.functional.softplus(x)


def _attached_elementwise(x: Tensor) -> Tensor:
    return attach_gradient(_softplus, x, torch.sigmoid(x))


def _nest(f, modes: str):
    for mode in modes:
        f = (jacrev if mode == "r" else jacfwd)(f)
    return f


def _point() -> Tensor:
    gen = torch.Generator().manual_seed(0)
    return torch.randn(4, generator=gen, **DD)


def test_value() -> None:
    x = torch.randn(3, 4, **DD)
    assert torch.equal(_attached(x), _lse(x))


@pytest.mark.parametrize("order", [1, 2, 3])
def test_all_nestings(order: int) -> None:
    x = _point()
    for modes in itertools.product("rf", repeat=order):
        ref = _nest(_lse, modes)(x)
        got = _nest(_attached, modes)(x)
        assert pytest.approx(ref.numpy(), abs=1e-12) == got.numpy()


@pytest.mark.parametrize("modes", ["r", "rr", "fr", "rf", "ff", "fff", "rfr"])
def test_vmap(modes: str) -> None:
    x = torch.randn(3, 4, **DD)
    batched = vmap(_nest(_attached, modes))(x)
    looped = torch.stack([_nest(_lse, modes)(xi) for xi in x])
    assert pytest.approx(looped.numpy(), abs=1e-12) == batched.numpy()


def test_elementwise() -> None:
    x = _point()
    for modes in ("r", "rf", "fff"):
        ref = _nest(_softplus, modes)(x)
        got = _nest(_attached_elementwise, modes)(x)
        assert pytest.approx(ref.numpy(), abs=1e-12) == got.numpy()


def test_gradcheck() -> None:
    x = torch.randn(2, 4, **DD, requires_grad=True)
    assert torch.autograd.gradcheck(_attached, (x,))
    assert torch.autograd.gradgradcheck(_attached, (x,))
    assert torch.autograd.gradcheck(_attached, (x,), check_forward_ad=True)


def test_plain_forward_ad() -> None:
    x, t = _point(), torch.ones(4, **DD)
    with forward_ad.dual_level():
        out = _attached(forward_ad.make_dual(x, t))
        tangent = forward_ad.unpack_dual(out).tangent
    assert pytest.approx((torch.softmax(x, -1) * t).sum().item()) == tangent


def test_function_is_not_differentiated() -> None:
    def broken(x: Tensor) -> Tensor:
        return _lse(x.detach())

    x = _point()
    got = jacrev(lambda y: attach_gradient(broken, y, torch.softmax(y, -1)))
    assert pytest.approx(torch.softmax(x, -1).numpy()) == got(x).numpy()


def test_shape_mismatch() -> None:
    x = _point()
    with pytest.raises(ValueError, match="shape"):
        attach_gradient(_lse, x, x[:2])


@requires_compile
def test_compile_backward() -> None:
    x = _point().requires_grad_(True)
    compiled = torch.compile(_attached, fullgraph=True, backend="aot_eager")
    (grad,) = torch.autograd.grad(compiled(x), x)
    assert pytest.approx(torch.softmax(x, -1).detach().numpy()) == grad


@requires_compile
@pytest.mark.skipif(
    __tversion__ < (2, 11, 0), reason="jacrev under compile needs >= 2.11"
)
def test_compile_jacrev() -> None:
    x = _point()
    compiled = torch.compile(
        jacrev(_attached), fullgraph=True, backend="aot_eager"
    )
    assert pytest.approx(torch.softmax(x, -1).numpy()) == compiled(x).numpy()


@requires_compile
@pytest.mark.skipif(
    __tversion__ < (2, 11, 0), reason="raises before 2.11 (documented)"
)
@pytest.mark.xfail(
    strict=True,
    reason="PyTorch 2.11-2.14: nested derivatives of a Function inside "
    "compiled code miss its backward's dependence on saved intermediates",
)
def test_compile_hessian_is_wrong() -> None:
    x = _point()
    compiled = torch.compile(
        jacrev(jacrev(_attached)), fullgraph=True, backend="aot_eager"
    )
    ref = jacrev(jacrev(_lse))(x)
    assert pytest.approx(ref.numpy(), abs=1e-12) == compiled(x).numpy()


def test_vmap_unbatched_gradient() -> None:
    x = torch.randn(3, 4, **DD)
    g = torch.softmax(x[0], dim=-1)
    out = vmap(lambda x: attach_gradient(_lse, x, g))(x)
    assert torch.equal(out, _lse(x))
