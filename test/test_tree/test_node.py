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
Test the `Node` base class: layout rules, construction, value checks, pytree
behaviour, conversion, `replace`, copying and identity semantics.
"""

from __future__ import annotations

import copy
import dataclasses
import pickle

import pytest
import torch
from torch.utils import _pytree as pytree

from tad_mctc.tree import Node, NodeLayoutError, child

from .samples import Base, Holder, Override, Plain, Sub, _double


def _sub(dtype: torch.dtype = torch.float64, **kwargs) -> Sub:  # type: ignore[no-untyped-def]
    return Sub(
        numbers=torch.tensor([1, 1, 8]),
        positions=torch.rand(3, 3, dtype=dtype),
        **kwargs,
    )


def test_field_order() -> None:
    assert Sub._child_names == ("numbers", "positions", "charge", "uhf", "rcov")
    assert Sub._context_names == ("label", "cutoff")
    assert Plain._child_names == Sub._child_names
    assert Plain._context_names == Sub._context_names
    assert isinstance(
        Plain(numbers=torch.tensor([1]), positions=torch.zeros(1, 3)), Plain
    )


def test_construction() -> None:
    with pytest.raises(TypeError):
        Base(torch.tensor([1]), torch.zeros(1, 3))  # type: ignore[call-arg]

    obj = _sub()
    with pytest.raises(dataclasses.FrozenInstanceError):
        obj.positions = torch.zeros(3, 3)  # type: ignore[misc]

    with pytest.raises(RuntimeError):
        Sub(numbers=torch.tensor([1]), positions=torch.zeros(1, 2))


def test_layout_error_undeclared_field() -> None:
    with pytest.raises(NodeLayoutError):

        class _A(Node):
            x: int = 1


def test_layout_error_annotation_without_field() -> None:
    with pytest.raises(NodeLayoutError):

        class _A(Base):
            y: int


def test_layout_error_init() -> None:
    with pytest.raises(NodeLayoutError):

        class _A(Base):
            def __init__(self) -> None:
                pass


def test_layout_error_post_init() -> None:
    with pytest.raises(NodeLayoutError):

        class _A(Base):
            def __post_init__(self) -> None:
                pass


def test_layout_error_reserved_name() -> None:
    with pytest.raises(NodeLayoutError):

        class _A(Node):
            dtype: int = child()  # type: ignore[assignment]


def test_layout_error_dunder_name() -> None:
    # a literal `__x` in a class body would be name-mangled
    namespace = {"__annotations__": {"__x": "int"}, "__x": child()}
    with pytest.raises(NodeLayoutError):
        type("A", (Node,), namespace)


def test_private_field_allowed() -> None:
    class A(Node):
        _private: torch.Tensor = child()

    assert A._child_names == ("_private",)
    assert A(_private=torch.zeros(1))._private.shape == (1,)


def test_value_error_tensor_in_context() -> None:
    with pytest.raises(TypeError):
        Base(
            numbers=torch.tensor([1]),
            positions=torch.zeros(1, 3),
            label=torch.zeros(1),  # type: ignore[arg-type]
        )


def test_value_error_float_list_in_child() -> None:
    with pytest.raises(TypeError):
        _sub(rcov=[1.0, 2.0])


def test_value_error_unhashable_static() -> None:
    with pytest.raises(TypeError):
        _sub(rcov={1, 2})


def test_value_error_mixed_float_dtypes() -> None:
    with pytest.raises(TypeError):
        _sub(charge=torch.zeros(3, dtype=torch.float32))


def test_value_error_mixed_devices() -> None:
    with pytest.raises(RuntimeError):
        _sub(charge=torch.zeros(3, dtype=torch.float64, device="meta"))


def test_keep_dtype_accepted() -> None:
    obj = _sub(uhf=torch.zeros(1, dtype=torch.float32))
    assert obj.uhf is not None and obj.uhf.dtype == torch.float32


def test_static_child_value() -> None:
    obj = _sub(rcov=_double)
    assert len(pytree.tree_leaves(obj)) == 2

    leaves, spec = pytree.tree_flatten(obj)
    assert pytree.tree_unflatten(leaves, spec).rcov is _double


def test_round_trip_and_structure() -> None:
    obj = _sub(charge=torch.zeros(1, dtype=torch.float64))
    leaves, spec = pytree.tree_flatten(obj)
    new = pytree.tree_unflatten(leaves, spec)
    assert type(new) is Sub
    for name in Sub._child_names:
        assert getattr(new, name) is getattr(obj, name)
    assert new.label == obj.label and new.cutoff == obj.cutoff

    assert pytree.tree_structure(_sub()) == pytree.tree_structure(_sub())
    assert pytree.tree_structure(_sub()) != pytree.tree_structure(obj)

    paths, _ = pytree.tree_flatten_with_path(obj)
    assert [pytree.keystr(p) for p, _ in paths] == [
        ".numbers",
        ".positions",
        ".charge",
    ]


def test_conversion_identity() -> None:
    obj = _sub()
    assert obj.to() is obj
    assert obj.to(dtype=torch.float64) is obj


def test_conversion_dtype() -> None:
    uhf = torch.zeros(1, dtype=torch.float64)
    obj = _sub(uhf=uhf)
    new = obj.to(dtype=torch.float32)
    assert new.positions.dtype == torch.float32
    assert new.numbers.dtype == torch.int64
    assert new.uhf is uhf

    other = obj.type(torch.float32)
    assert other.positions.dtype == new.positions.dtype
    assert other.numbers.dtype == new.numbers.dtype
    assert other.uhf is uhf


def test_conversion_device() -> None:
    new = _sub().to(device="meta")
    assert new.numbers.device.type == "meta"
    assert new.positions.device.type == "meta"


def test_dtype_device_dd() -> None:
    obj = _sub()
    assert obj.dtype == torch.float64
    assert obj.device == torch.device("cpu")
    assert obj.dd == {"device": torch.device("cpu"), "dtype": torch.float64}

    class Ints(Node):
        numbers: torch.Tensor = child()

    with pytest.raises(AttributeError):
        Ints(numbers=torch.tensor([1])).dtype


def test_nesting() -> None:
    holder = Holder(
        system=_sub(), extra={"a": torch.zeros(2, dtype=torch.float64)}
    )
    new = holder.to(dtype=torch.float32)
    assert new.system.positions.dtype == torch.float32  # type: ignore[attr-defined]
    assert new.extra["a"].dtype == torch.float32

    paths, _ = pytree.tree_flatten_with_path(holder)
    keys = [pytree.keystr(p) for p, _ in paths]
    assert ".system.positions" in keys
    assert ".extra['a']" in keys


def test_replace() -> None:
    obj = _sub()
    new = obj.replace(cutoff=40.0)
    assert new is not obj and new.cutoff == 40.0
    assert new.positions is obj.positions

    with pytest.raises(RuntimeError):
        obj.replace(positions=torch.zeros(3, 2, dtype=torch.float64))


def test_convert_child_override() -> None:
    obj = Override(
        numbers=torch.tensor([1]),
        positions=torch.zeros(1, 3, dtype=torch.float64),
        charge=torch.zeros(1, dtype=torch.float64),
    )
    new = obj.to(dtype=torch.float32)
    assert new.charge == "overridden"
    assert new.positions.dtype == torch.float32


def test_pickle_and_deepcopy() -> None:
    obj = _sub(charge=torch.ones(1, dtype=torch.float64))
    for new in (pickle.loads(pickle.dumps(obj)), copy.deepcopy(obj)):
        assert type(new) is Sub
        for name in ("numbers", "positions", "charge"):
            assert torch.equal(getattr(new, name), getattr(obj, name))
        assert new.label == obj.label and new.cutoff == obj.cutoff


def test_repr() -> None:
    text = repr(_sub())
    assert "Tensor(shape=(3, 3)" in text
    assert "tensor(" not in text


def test_identity_semantics() -> None:
    obj = _sub()
    assert obj.__eq__(obj)
    other = obj.replace()
    assert not (obj == other)
    assert isinstance(hash(obj), int)


def _placeholder_objects() -> list[Sub]:
    return [
        _sub(),
        _sub(charge=torch.zeros(3, dtype=torch.float64)),
        _sub(rcov=_double),
    ]


@pytest.mark.parametrize("obj", _placeholder_objects())
def test_placeholder_round_trip(obj: Sub) -> None:
    # what `vmap` does internally
    spec = pytree.tree_structure(obj)
    probe = pytree.tree_unflatten([0] * spec.num_leaves, spec)
    assert pytree.tree_structure(probe) == spec
    assert pytree._broadcast_to_and_flatten(0, spec) == [0] * spec.num_leaves


@pytest.mark.parametrize("obj", _placeholder_objects())
def test_structure_preserved(obj: Sub) -> None:
    spec = pytree.tree_structure(obj)
    assert pytree.tree_structure(copy.deepcopy(obj)) == spec
    assert pytree.tree_structure(pickle.loads(pickle.dumps(obj))) == spec
    assert pytree.tree_structure(obj.to(dtype=torch.float32)) == spec


def test_replace_recomputes_structure() -> None:
    obj = _sub(rcov=_double)
    assert len(pytree.tree_leaves(obj)) == 2

    new = obj.replace(rcov=torch.zeros(3, dtype=torch.float64))
    assert len(pytree.tree_leaves(new)) == 3
