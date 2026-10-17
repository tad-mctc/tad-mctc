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
Test `NodeFunction`: custom autograd functions with `Node` arguments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
import torch
from torch.func import grad, jacfwd, jacrev, vmap
from torch.utils import _pytree as pytree

from tad_mctc.tools.testing import requires_compile
from tad_mctc.tree import Node, NodeFunction, child, context
from tad_mctc.tree import function as function_module

from ..utils import COMPILE_BACKEND, compile_fullgraph

DTYPE = torch.float64


class Pair(Node):
    x: torch.Tensor = child()
    y: torch.Tensor = child()
    scale: float = context(default=1.0)


class Outer(Node):
    pair: Pair = child()
    extra: dict[str, torch.Tensor] = child(default_factory=dict)


def _pair(x: torch.Tensor, y: torch.Tensor | None = None) -> Pair:
    return Pair(x=x, y=torch.full_like(x, 3.0) if y is None else y, scale=2.0)


class Product(NodeFunction):
    """``scale * x * y * z`` of a Pair and a tensor, with all derivatives"""

    @staticmethod
    def forward(pair: Pair, z: torch.Tensor) -> torch.Tensor:
        return pair.scale * pair.x * pair.y * z

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:
        NodeFunction.save(ctx, *inputs)

    @staticmethod
    def backward(ctx: Any, g: torch.Tensor) -> tuple[Any, ...]:
        pair, z = NodeFunction.saved(ctx)
        s = pair.scale
        needs = ctx.needs_input_grad
        grads = {}
        if needs[0].x:
            grads["x"] = g * s * pair.y * z
        if needs[0].y:
            grads["y"] = g * s * pair.x * z
        dz = g * s * pair.x * pair.y if needs[1] else None
        return grads, dz

    @staticmethod
    def jvp(ctx: Any, pair_t: Pair, z_t: torch.Tensor | None) -> torch.Tensor:
        pair, z = NodeFunction.saved(ctx)
        s = pair.scale
        out = torch.zeros_like(z)
        if pair_t.x is not None:
            out = out + s * pair_t.x * pair.y * z
        if pair_t.y is not None:
            out = out + s * pair.x * pair_t.y * z
        if z_t is not None:
            out = out + s * pair.x * pair.y * z_t
        return out


class GeneratedProduct(Product):
    generate_vmap_rule = True


class CustomVmapProduct(Product):
    """Batching rule written by hand, with the batch dimensions structured"""

    @staticmethod
    def vmap(info: Any, in_dims: tuple[Any, ...], pair: Pair, z: torch.Tensor):  # type: ignore[no-untyped-def]
        pair_dims, z_dim = in_dims

        def front(t: torch.Tensor, dim: int | None) -> torch.Tensor:
            return t.unsqueeze(0) if dim is None else t.movedim(dim, 0)

        moved = Pair(
            x=front(pair.x, pair_dims.x),
            y=front(pair.y, pair_dims.y),
            scale=pair.scale,
        )
        return CustomVmapProduct.apply(moved, front(z, z_dim)), 0


class NoJvpProduct(NodeFunction):
    @staticmethod
    def forward(pair: Pair) -> torch.Tensor:
        return pair.x * pair.y

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:
        NodeFunction.save(ctx, inputs[0])

    @staticmethod
    def backward(ctx: Any, g: torch.Tensor) -> Any:
        (pair,) = NodeFunction.saved(ctx)
        return {"x": g * pair.y, "y": g * pair.x}


@dataclass(frozen=True)
class Slot:
    """A pytree key of its own kind, as other libraries may register"""

    name: str

    def get(self, parent: Any) -> Any:
        return parent.values[self.name]


class Slots:
    """A container registered with `Slot` keys"""

    def __init__(self, **values: Any) -> None:
        self.values = values


pytree.register_pytree_node(
    Slots,
    lambda s: (list(s.values.values()), list(s.values)),
    lambda children, names: Slots(**dict(zip(names, children))),
    flatten_with_keys_fn=lambda s: (
        [(Slot(name), value) for name, value in s.values.items()],
        list(s.values),
    ),
)


def _values() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(0)
    return tuple(  # type: ignore[return-value]
        torch.rand(4, dtype=DTYPE, generator=gen) for _ in range(3)
    )


def _reference(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):  # type: ignore[no-untyped-def]
    return 2.0 * x * y * z


def test_gradients_reach_node_leaves() -> None:
    """Leaves inside a Node get their gradients (plain Function: zeros)."""
    x, y, z = (t.clone().requires_grad_() for t in _values())
    Product.apply(_pair(x, y), z).sum().backward()

    xr, yr, zr = (t.detach().clone().requires_grad_() for t in (x, y, z))
    _reference(xr, yr, zr).sum().backward()
    for got, ref in ((x, xr), (y, yr), (z, zr)):
        assert got.grad is not None and ref.grad is not None
        assert torch.allclose(got.grad, ref.grad)


def test_needs_input_grad_is_structured() -> None:
    """Only the leaves that require gradients get one."""
    x, y, z = _values()
    x = x.clone().requires_grad_()
    out = Product.apply(_pair(x, y), z)
    (dx,) = torch.autograd.grad(out.sum(), x)
    assert torch.allclose(dx, 2.0 * y * z)


@pytest.mark.parametrize("transform", [jacrev, jacfwd], ids=["rev", "fwd"])
def test_jacobians(transform: Any) -> None:
    """Jacobians w.r.t. a Node (its leaves) and a tensor match autograd."""
    x, y, z = _values()

    def func(pair: Pair, z: torch.Tensor) -> torch.Tensor:
        return GeneratedProduct.apply(pair, z)

    jac_pair, jac_z = transform(func, argnums=(0, 1))(_pair(x, y), z)
    assert torch.allclose(jac_pair.x, torch.diag(2.0 * y * z))
    assert torch.allclose(jac_pair.y, torch.diag(2.0 * x * z))
    assert torch.allclose(jac_z, torch.diag(2.0 * x * y))


def test_jvp_tangents_are_structured() -> None:
    """A tangent for only one leaf of a Node arrives as such."""
    x, y, z = _values()
    tx = torch.ones_like(x)

    def of_x(x: torch.Tensor) -> torch.Tensor:
        return Product.apply(_pair(x, y), z)

    tangent = torch.func.jvp(of_x, (x,), (tx,))[1]
    assert torch.allclose(tangent, 2.0 * y * z)


@pytest.mark.parametrize("cls", [GeneratedProduct, CustomVmapProduct])
def test_vmap(cls: type[NodeFunction]) -> None:
    """Generated and custom batching rules over a Node and a tensor."""
    x, y, z = _values()
    xs = torch.stack([x, 2 * x])

    out = vmap(lambda x: cls.apply(_pair(x, y), z))(xs)
    assert torch.allclose(out[0], _reference(x, y, z))
    assert torch.allclose(out[1], _reference(2 * x, y, z))


def test_second_order() -> None:
    """Hessians through the Function's own derivatives."""
    x, y, z = _values()

    def loss(x: torch.Tensor) -> torch.Tensor:
        return (GeneratedProduct.apply(_pair(x, y), z) ** 2).sum()

    expected = torch.diag(2 * (2.0 * y * z) ** 2)
    assert torch.allclose(jacfwd(grad(loss))(x), expected)


def test_nested_nodes_and_containers() -> None:
    """Gradients of nested Nodes and dicts are nested dicts."""

    class Total(NodeFunction):
        @staticmethod
        def forward(outer: Outer) -> torch.Tensor:
            return (outer.pair.x * outer.extra["w"]).sum()

        @staticmethod
        def setup_context(
            ctx: Any, inputs: tuple[Any, ...], output: Any
        ) -> None:
            NodeFunction.save(ctx, inputs[0])

        @staticmethod
        def backward(ctx: Any, g: torch.Tensor) -> Any:
            (outer,) = NodeFunction.saved(ctx)
            return {
                "pair": {"x": g * outer.extra["w"]},
                "extra": {"w": g * outer.pair.x},
            }

    x, y, w = (t.clone().requires_grad_() for t in _values())
    outer = Outer(pair=Pair(x=x, y=y), extra={"w": w})
    Total.apply(outer).backward()
    assert x.grad is not None and w.grad is not None and y.grad is None
    assert torch.allclose(x.grad, w.detach())
    assert torch.allclose(w.grad, x.detach())


def test_containers_with_their_own_keys() -> None:
    """Gradients of a container with its own key type keep its structure."""

    class Weighted(NodeFunction):
        @staticmethod
        def forward(slots: Slots) -> torch.Tensor:
            return (slots.values["a"] * slots.values["b"]).sum()

        @staticmethod
        def setup_context(
            ctx: Any, inputs: tuple[Any, ...], output: Any
        ) -> None:
            NodeFunction.save(ctx, inputs[0])

        @staticmethod
        def backward(ctx: Any, g: torch.Tensor) -> Any:
            (slots,) = NodeFunction.saved(ctx)
            return Slots(a=g * slots.values["b"], b=None)

    a, b, _ = _values()
    a = a.clone().requires_grad_()
    Weighted.apply(Slots(a=a, b=b)).backward()
    assert a.grad is not None and torch.allclose(a.grad, b)


def test_forward_skips_a_leading_context() -> None:
    """Dynamo may pass the context first (PyTorch 2.14, without gradients)."""
    x, y, z = _values()
    leaves, spec = pytree.tree_flatten((_pair(x, y), z))
    function = Product._function  # pylint: disable=protected-access
    out = function.forward(None, spec, *leaves)
    assert torch.allclose(out, _reference(x, y, z))


def test_custom_jvp_breaks_the_graph_only_with_gradients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    The check breaks the graph only while compiling and recording gradients
    of a leaf. Dynamo traces this code without running it, so it is called
    with compilation faked here.
    """
    messages: list[str] = []
    monkeypatch.setattr(function_module, "is_compiling", lambda: True)
    monkeypatch.setattr(
        torch._dynamo,  # pylint: disable=protected-access
        "graph_break",
        lambda msg="": messages.append(msg),
    )
    x, y, z = _values()

    function_module._check_compiled_jvp(Product, [x, y, z])
    with torch.no_grad():
        function_module._check_compiled_jvp(Product, [x.requires_grad_(), y])
    assert not messages

    function_module._check_compiled_jvp(Product, [x, y, z])
    assert (
        len(messages) == 1 and "Product defines a custom `jvp`" in messages[0]
    )


def test_other_arguments_pass_through() -> None:
    """Values that are no tensors reach `forward` unchanged."""

    class Shift(NodeFunction):
        @staticmethod
        def forward(t: torch.Tensor, offset: int, label: str) -> torch.Tensor:
            assert label == "shift"
            return t + offset

        @staticmethod
        def backward(ctx: Any, g: torch.Tensor) -> Any:
            return g, None, None

    t = torch.zeros(2, requires_grad=True)
    Shift.apply(t, 3, "shift").sum().backward()
    assert t.grad is not None and torch.equal(t.grad, torch.ones(2))


def test_save_restores_structures() -> None:
    """`save` and `saved` restore Nodes, containers and other values."""

    class Check(NodeFunction):
        @staticmethod
        def forward(pair: Pair, items: list[Any]) -> torch.Tensor:
            return pair.x.clone()

        @staticmethod
        def setup_context(
            ctx: Any, inputs: tuple[Any, ...], output: Any
        ) -> None:
            NodeFunction.save(ctx, *inputs, None)

        @staticmethod
        def backward(ctx: Any, g: torch.Tensor) -> Any:
            pair, items, nothing = NodeFunction.saved(ctx)
            assert isinstance(pair, Pair) and pair.scale == 2.0
            assert items[0] == "label" and torch.equal(items[1], pair.y)
            assert nothing is None
            return {"x": g}, None

    x, y, _ = _values()
    x = x.clone().requires_grad_()
    Check.apply(_pair(x, y), ["label", y]).sum().backward()
    assert x.grad is not None


def test_wrong_gradient_structure() -> None:
    """A gradient for something that is no tensor argument is refused."""

    class Wrong(NoJvpProduct):
        @staticmethod
        def backward(ctx: Any, g: torch.Tensor) -> Any:
            return {"z": g}

    x, y, _ = _values()
    out = Wrong.apply(_pair(x.clone().requires_grad_(), y))
    with pytest.raises(TypeError, match="no tensor of the arguments"):
        out.sum().backward()


def test_forward_must_return_tensors() -> None:
    """A Node output is refused with an explanation."""

    class ReturnsNode(NodeFunction):
        @staticmethod
        def forward(pair: Pair) -> Any:
            return pair

    x, y, _ = _values()
    with pytest.raises(TypeError, match="tensor or a tuple of tensors"):
        ReturnsNode.apply(_pair(x, y))


def test_methods_left_out_keep_torch_defaults() -> None:
    """Without `jvp` or `vmap`, forward mode and batching fail as for a
    plain Function."""
    x, y, _ = _values()
    with pytest.raises(Exception, match="jvp"):
        torch.func.jvp(lambda x: NoJvpProduct.apply(_pair(x, y)), (x,), (x,))
    with pytest.raises(Exception, match="vmap"):
        vmap(lambda x: NoJvpProduct.apply(_pair(x, y)))(x[None])


@requires_compile
def test_compile_without_gradients() -> None:
    """Fullgraph compile of a NodeFunction (also with a custom `jvp`)."""
    x, y, z = _values()
    compiled = compile_fullgraph(lambda p, z: Product.apply(p, z))
    assert torch.allclose(compiled(_pair(x, y), z), _reference(x, y, z))


@requires_compile
def test_compile_forward_mode_and_vmap() -> None:
    """Compiled forward mode and generated batching rules."""
    x, y, z = _values()

    def jac(x: torch.Tensor) -> torch.Tensor:
        return jacfwd(lambda x: GeneratedProduct.apply(_pair(x, y), z))(x)

    assert torch.allclose(compile_fullgraph(jac)(x), torch.diag(2.0 * y * z))

    def product(x: torch.Tensor) -> torch.Tensor:
        return GeneratedProduct.apply(_pair(x, y), z)

    batched = compile_fullgraph(vmap(product))(torch.stack([x, 2 * x]))
    assert torch.allclose(batched[1], _reference(2 * x, y, z))


@requires_compile
def test_compile_with_backward() -> None:
    """Compiled reverse mode of a NodeFunction without a custom `jvp`."""
    x, y, _ = _values()
    x = x.clone().requires_grad_()
    compiled = compile_fullgraph(lambda p: NoJvpProduct.apply(p).sum())
    compiled(_pair(x, y)).backward()
    assert x.grad is not None and torch.allclose(x.grad, y)


@requires_compile
@pytest.mark.skipif(
    not function_module._explains_compiled_jvp(),
    reason="this PyTorch version traces custom `jvp` methods, or its "
    "graph breaks take no explanation",
)
def test_compile_with_backward_and_jvp_explains() -> None:
    """The PyTorch limitation is reported with an explanation."""
    x, y, z = _values()
    x = x.clone().requires_grad_()
    compiled = compile_fullgraph(lambda p, z: Product.apply(p, z).sum())
    with pytest.raises(Exception, match="defines a custom `jvp`"):
        compiled(_pair(x, y), z)


@requires_compile
def test_compile_with_backward_and_jvp_without_fullgraph() -> None:
    """Without `fullgraph`, the function runs eagerly at the graph break."""
    x, y, z = _values()
    x = x.clone().requires_grad_()
    (expected,) = torch.autograd.grad(Product.apply(_pair(x, y), z).sum(), x)

    compiled = torch.compile(
        lambda p, z: Product.apply(p, z).sum(), backend=COMPILE_BACKEND
    )
    compiled(_pair(x, y), z).backward()
    assert x.grad is not None and torch.allclose(x.grad, expected)
