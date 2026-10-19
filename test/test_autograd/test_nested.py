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
Test `nested_jvp`: forward-mode rules visible to enclosing forward-mode
transforms.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch.func import jacfwd, jacrev, jvp, vmap

from tad_mctc.autograd import nested_jvp
from tad_mctc.typing import Tensor


def _cube(rule_wrapper: Any, vmap_rule: bool) -> Any:
    """`x**3 * c` with a hand-written jvp, wrapped or not."""

    class Cube(torch.autograd.Function):
        generate_vmap_rule = vmap_rule

        @staticmethod
        def forward(x: Tensor, c: Tensor) -> Tensor:
            return x**3 * c

        @staticmethod
        def setup_context(ctx: Any, inputs: Any, output: Any) -> None:
            x, c = inputs
            ctx.save_for_forward(x, c)
            ctx.save_for_backward(x, c)
            ctx.scale = 3.0

        @staticmethod
        def backward(ctx: Any, grad: Tensor) -> tuple[Tensor, Tensor]:
            x, c = ctx.saved_tensors
            return grad * ctx.scale * x**2 * c, grad * x**3

        @staticmethod
        @rule_wrapper
        def jvp(ctx: Any, x_dot: Tensor | None, c_dot: Tensor | None) -> Tensor:
            x, c = ctx.saved_tensors
            out = torch.zeros_like(x)
            if x_dot is not None:
                out = out + ctx.scale * x**2 * c * x_dot
            if c_dot is not None:
                out = out + x**3 * c_dot
            return out

    if not vmap_rule:

        def batched(info: Any, in_dims: Any, x: Tensor, c: Tensor) -> Any:
            def front(value: Tensor, dim: int | None) -> Tensor:
                if dim is None:
                    return value.expand(info.batch_size, *value.shape)
                return value.movedim(dim, 0)

            return Cube.apply(front(x, in_dims[0]), front(c, in_dims[1])), 0

        Cube.vmap = staticmethod(batched)  # type: ignore[method-assign]

    return Cube.apply


def _identity(rule: Any) -> Any:
    return rule


x0 = torch.tensor(1.5, dtype=torch.double)
c0 = torch.tensor(0.7, dtype=torch.double)


def test_plain_rule_is_hidden() -> None:
    """The PyTorch behavior that `nested_jvp` fixes."""
    cube = _cube(_identity, vmap_rule=True)
    second = jacfwd(jacfwd(lambda x: cube(x, c0)))(x0)
    assert second.item() == 0.0  # instead of 6 x c


@pytest.mark.parametrize("vmap_rule", [True, False])
def test_second_order(vmap_rule: bool) -> None:
    cube = _cube(nested_jvp, vmap_rule)
    expected = 6 * x0 * c0

    def f(x: Tensor) -> Tensor:
        return cube(x, c0)

    assert pytest.approx(expected.item()) == jacfwd(jacfwd(f))(x0).item()
    assert pytest.approx(expected.item()) == jacrev(jacfwd(f))(x0).item()
    assert pytest.approx(expected.item()) == jacfwd(jacrev(f))(x0).item()

    nested = jvp(
        lambda x: jvp(f, (x,), (torch.ones_like(x),))[1],
        (x0,),
        (torch.ones_like(x0),),
    )[1]
    assert pytest.approx(expected.item()) == nested.item()

    # mixed with respect to the second argument
    mixed = jacfwd(jacfwd(cube, argnums=0), argnums=1)(x0, c0)
    assert pytest.approx(3 * x0.item() ** 2) == mixed.item()


def test_third_order_jvp() -> None:
    """Three `jvp` levels without `vmap` in between work."""
    cube = _cube(nested_jvp, vmap_rule=True)
    one = torch.ones_like(x0)

    def d(f: Any) -> Any:
        return lambda x: jvp(f, (x,), (one,))[1]

    third = d(d(d(lambda x: cube(x, c0))))(x0)
    assert pytest.approx(6 * c0.item()) == third.item()


def test_third_order_custom_vmap_rule() -> None:
    """With a custom `vmap` rule, `jacfwd` nests to any order."""
    cube = _cube(nested_jvp, vmap_rule=False)
    third = jacfwd(jacfwd(jacfwd(lambda x: cube(x, c0))))(x0)
    assert pytest.approx(6 * c0.item()) == third.item()


def test_third_order_through_generated_vmap_rule() -> None:
    cube = _cube(nested_jvp, vmap_rule=True)

    def f(x: Tensor) -> Tensor:
        return cube(x, c0)

    with pytest.raises(NotImplementedError, match="generated `vmap` rule"):
        jacfwd(jacfwd(jacfwd(f)))(x0)

    # reverse mode for the outer derivatives works
    third = jacfwd(jacrev(jacrev(f)))(x0)
    assert pytest.approx(6 * c0.item()) == third.item()


def test_vmap() -> None:
    cube = _cube(nested_jvp, vmap_rule=True)
    xs = torch.tensor([0.5, 1.0, 2.0], dtype=torch.double)

    out = vmap(jacfwd(jacfwd(lambda x: cube(x, c0))))(xs)
    assert pytest.approx((6 * xs * c0).tolist()) == out.tolist()


def test_forward_ad() -> None:
    """`torch.autograd.forward_ad` (no `torch.func` level) is unchanged."""
    import torch.autograd.forward_ad as fwad

    cube = _cube(nested_jvp, vmap_rule=True)
    with fwad.dual_level():
        dual = fwad.make_dual(x0, torch.ones_like(x0))
        tangent = fwad.unpack_dual(cube(dual, c0)).tangent

    assert pytest.approx(3 * x0.item() ** 2 * c0.item()) == tangent.item()
