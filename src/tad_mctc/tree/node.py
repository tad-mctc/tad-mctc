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
Tree: Node
==========

Base class for frozen tensor containers that work under ``torch.func.vmap``,
``jacrev``, ``jacfwd`` and ``torch.compile``.

Subclasses are frozen dataclasses. ``Node.__init_subclass__`` applies
``dataclasses.dataclass(frozen=True, eq=False, kw_only=True, repr=False)`` to
every subclass, so subclasses never use the decorator themselves.

Every field is declared with :func:`child` or :func:`context`. A child field
holds a tensor, a :class:`Node`, a list, tuple or dict containing only tensors
and nodes (nested containers are allowed), ``None``, or a static hashable
value such as a function or a float. A context field holds a static hashable
value; never a tensor or a node, also not inside a container.

The objects are registered as pytrees. The leaves are the child fields holding
a tensor, a node or a container. The tree structure (pytree context) consists
of the names of those fields, the child fields holding ``None`` or a static
value (with their values), and all context field values. Which child
fields are leaves is decided once, at construction, from the values, and stored
on the object. Flattening reads the stored decision and never re-derives it
from the current values; unflattening restores it from the tree structure and
``.to()`` copies it. This is required because ``vmap`` builds a probe object by
unflattening a tree structure with placeholder leaves (plain ints) and
flattens that object again; the result must have the same structure.
Consequently, two objects only have the same tree structure if the same child
fields are set to tensors and their static values compare equal. ``vmap`` and
``stack`` over several objects need the same tree structure.

Construction (the generated ``__init__``, also used by ``replace``) runs in
this order: ``_normalize()``, which returns a dict of field values to fill in;
the value checks; ``_validate()``. Unflattening (used by ``vmap``, ``jacrev``,
``torch.compile`` and ``.to()``) runs none of these.

The value checks verify that context fields hold no tensor or node and are
hashable, that containers in child fields hold only tensors and nodes, that
static child values are hashable, that all floating-point tensors in child
fields share one dtype (except fields declared with ``keep_dtype=True``), and
that the tensors of the object and its child nodes are all on one device.

``_validate()`` may read only metadata: types, dtypes, devices, shapes and
Python values. It must never read tensor values (no ``.item()``, ``.any()``,
``bool(tensor)``), so that construction works inside ``vmap`` and compiled
code.

``.to(device=None, dtype=None)`` converts field by field. Floating-point
tensors take ``dtype`` unless declared with ``keep_dtype=True``; all tensors
take ``device``. Integer and boolean tensors keep their dtype, child nodes
convert recursively and static values are unchanged. If nothing changes,
``.to`` returns ``self``. ``.type(dtype)`` is ``.to(dtype=dtype)``. Subclasses
may override ``_convert_child`` for one field and call ``super()`` for the
others.

``dtype`` is the dtype of the object's first floating-point tensor, otherwise
that of its first child node; an ``AttributeError`` is raised if there is
none. ``device`` works the same way with any tensor. ``dd`` returns both as a
:class:`~tad_mctc.typing.DD` dict.

The value checks are skipped inside compiled code (``torch.compile``);
``_validate`` always runs.

Equality and hashing are by identity (``eq=False``).

Subclasses must not define ``__init__``, ``__post_init__`` or ``__setattr__``,
must not use the field names ``to``, ``type``, ``replace``, ``dtype``,
``device``, ``dd`` or ``_node_leaf_fields``, must not use field names starting with two underscores
and must not have annotated class attributes that are not fields (except
``ClassVar``). Violations raise :class:`NodeLayoutError` when the class is
defined.
"""

from __future__ import annotations

import dataclasses
import functools
import inspect
import typing
from collections.abc import Callable
from typing import Any, ClassVar

import torch
from torch import Tensor
from torch.utils import _pytree as pytree
from typing_extensions import dataclass_transform

from ..tools import is_compiling
from ..typing import DD
from ..typing.compat import Self

__all__ = ["Node", "NodeLayoutError", "child", "context"]

_KIND = "tad_mctc.tree.kind"
_KEEP_DTYPE = "tad_mctc.tree.keep_dtype"
_CHILD = "child"
_CONTEXT = "context"
_RESERVED = frozenset({"to", "type", "replace", "dtype", "device", "dd"})
_LAYOUT = "_node_leaf_fields"


class NodeLayoutError(TypeError):
    """A `Node` subclass declares its fields incorrectly."""


def child(
    *,
    default: Any = dataclasses.MISSING,
    default_factory: Callable[[], Any] | Any = dataclasses.MISSING,
    keep_dtype: bool = False,
) -> Any:
    """
    Declare a child field: a tensor, `Node`, container of them, `None`, or a
    static value.

    Parameters
    ----------
    default : Any, optional
        Default value of the field.
    default_factory : Callable[[], Any], optional
        Factory for the default value.
    keep_dtype : bool, optional
        Exclude the field from dtype checks and dtype conversion. Defaults to
        ``False``.

    Returns
    -------
    Any
        The dataclass field.
    """
    return dataclasses.field(
        default=default,
        default_factory=default_factory,
        metadata={_KIND: _CHILD, _KEEP_DTYPE: keep_dtype},
    )


def context(
    *,
    default: Any = dataclasses.MISSING,
    default_factory: Callable[[], Any] | Any = dataclasses.MISSING,
) -> Any:
    """
    Declare a context field: a static, hashable value; never a tensor.

    Parameters
    ----------
    default : Any, optional
        Default value of the field.
    default_factory : Callable[[], Any], optional
        Factory for the default value.

    Returns
    -------
    Any
        The dataclass field.
    """
    return dataclasses.field(
        default=default,
        default_factory=default_factory,
        metadata={_KIND: _CONTEXT},
    )


def _is_container(value: Any) -> bool:
    return isinstance(value, (list, tuple, dict))


def _container_items(value: Any) -> list[Any]:
    if isinstance(value, dict):
        return list(value.values())
    return list(value)


def _container_only_arrays(value: Any) -> bool:
    for item in _container_items(value):
        if isinstance(item, (Tensor, Node)):
            continue

        if _is_container(item) and _container_only_arrays(item):
            continue

        return False

    return True


def _holds_array(value: Any) -> bool:
    if isinstance(value, (Tensor, Node)):
        return True

    if isinstance(value, (list, tuple, set, frozenset)):
        return any(_holds_array(v) for v in value)

    if isinstance(value, dict):
        return any(_holds_array(v) for v in value.values())

    return False


def _is_array_value(value: Any) -> bool:
    """True if the value becomes pytree leaves (tensor, node, container)."""
    return isinstance(value, (Tensor, Node)) or _is_container(value)


def _is_classvar(annotation: Any) -> bool:
    if isinstance(annotation, str):
        return annotation.split("[", 1)[0].strip() in (
            "ClassVar",
            "typing.ClassVar",
        )

    return typing.get_origin(annotation) is ClassVar or annotation is ClassVar


@dataclass_transform(
    frozen_default=True,
    kw_only_default=True,
    eq_default=False,
    field_specifiers=(child, context),
)
class Node:
    """Base class for frozen tensor containers (see module docstring)."""

    __dataclass_fields__: ClassVar[dict[str, dataclasses.Field[Any]]]
    _child_names: ClassVar[tuple[str, ...]] = ()
    _context_names: ClassVar[tuple[str, ...]] = ()
    _keep_dtype_names: ClassVar[frozenset[str]] = frozenset()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        for forbidden in ("__init__", "__post_init__", "__setattr__"):
            if forbidden in cls.__dict__:
                raise NodeLayoutError(
                    f"{cls.__qualname__} defines `{forbidden}`. Node subclasses"
                    " must not; use `_normalize`, `_validate` or a classmethod "
                    "constructor instead."
                )
        dataclasses.dataclass(frozen=True, eq=False, kw_only=True, repr=False)(
            cls
        )
        _set_layout(cls)
        _wrap_init(cls)
        _register(cls)

    # -- hooks ------------------------------------------------------------

    def _normalize(self) -> dict[str, Any]:
        """Return field values to fill in during construction."""
        return {}

    def _validate(self) -> None:
        """Check metadata (types, dtypes, devices, shapes); never values."""

    def _convert_child(
        self,
        name: str,
        value: Any,
        device: torch.device | str | None,
        dtype: torch.dtype | None,
    ) -> Any:
        """Convert one child field for `to()`. Subclasses may override for
        a specific field and call `super()` for all others."""
        keep = name in self._keep_dtype_names
        return _convert_value(value, device, None if keep else dtype)

    # -- conversion and access --------------------------------------------

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Self:
        """
        Convert the child fields to a device and/or dtype.

        Parameters
        ----------
        device : torch.device | str | None, optional
            Target device for all tensors.
        dtype : torch.dtype | None, optional
            Target dtype for floating-point tensors (except fields declared
            with ``keep_dtype=True``).

        Returns
        -------
        Self
            The converted object, or ``self`` if nothing changes.
        """
        changes: dict[str, Any] = {}
        for name in self._child_names:
            old = getattr(self, name)
            new = self._convert_child(name, old, device, dtype)
            if new is not old:
                changes[name] = new

        if not changes:
            return self

        return _copy_with(self, changes)

    def type(self, dtype: torch.dtype) -> Self:
        """
        Convert floating-point tensors to a dtype; same as ``to(dtype=dtype)``.

        Parameters
        ----------
        dtype : torch.dtype
            Target dtype.

        Returns
        -------
        Self
            The converted object.
        """
        return self.to(dtype=dtype)

    def replace(self, **changes: Any) -> Self:
        """
        Create a new object with some fields replaced (runs construction
        checks).

        Parameters
        ----------
        **changes : Any
            Field values to replace.

        Returns
        -------
        Self
            The new object.
        """
        # Not `dataclasses.replace`, which `torch.compile` cannot trace.
        current = {
            name: getattr(self, name)
            for name in self._child_names + self._context_names
        }
        return type(self)(**{**current, **changes})

    @property
    def dtype(self) -> torch.dtype:
        """
        Dtype of the first floating-point tensor (own tensors first, then
        child nodes).

        Raises
        ------
        AttributeError
            If the object holds no floating-point tensor.
        """
        for t in _own_tensors(self):
            if t.is_floating_point():
                return t.dtype

        for node in _child_nodes(self):
            try:
                return node.dtype
            except AttributeError:
                continue

        raise AttributeError(
            f"{type(self).__name__} holds no floating-point tensor, so it has "
            "no dtype."
        )

    @property
    def device(self) -> torch.device:
        """
        Device of the first tensor (own tensors first, then child nodes).

        Raises
        ------
        AttributeError
            If the object holds no tensor.
        """
        for t in _own_tensors(self):
            return t.device

        for node in _child_nodes(self):
            try:
                return node.device
            except AttributeError:
                continue

        raise AttributeError(
            f"{type(self).__name__} holds no tensor, so it has no device."
        )

    @property
    def dd(self) -> DD:
        """Device and dtype as a dict, ready to be passed as ``**dd``."""
        return {"device": self.device, "dtype": self.dtype}

    def __repr__(self) -> str:
        parts = []
        for f in dataclasses.fields(self):
            parts.append(f"{f.name}={_short_repr(getattr(self, f.name))}")

        return f"{type(self).__name__}({', '.join(parts)})"


# -- helpers --------------------------------------------------------------


def _short_repr(value: Any) -> str:
    if isinstance(value, Tensor):
        return (
            f"Tensor(shape={tuple(value.shape)}, dtype={value.dtype}, "
            f"device={value.device})"
        )

    return repr(value)


def _collect(value: Any, kind: type, out: list[Any]) -> None:
    # Module-level recursion instead of a closure: Dynamo (torch 2.4) cannot
    # trace nested functions with free variables.
    if isinstance(value, kind):
        out.append(value)
    elif _is_container(value):
        for item in _container_items(value):
            _collect(item, kind, out)


def _own_tensors(node: Node) -> list[Tensor]:
    out: list[Tensor] = []
    for name in node._child_names:
        _collect(getattr(node, name), Tensor, out)

    return out


def _child_nodes(node: Node) -> list[Node]:
    out: list[Node] = []
    for name in node._child_names:
        _collect(getattr(node, name), Node, out)

    return out


def _convert_value(
    value: Any, device: torch.device | str | None, dtype: torch.dtype | None
) -> Any:
    if isinstance(value, Tensor):
        target = (
            dtype if (dtype is not None and value.is_floating_point()) else None
        )
        return value.to(device=device, dtype=target)

    if isinstance(value, Node):
        return value.to(device=device, dtype=dtype)

    if _is_container(value):
        items = _container_items(value)
        new_items = [_convert_value(v, device, dtype) for v in items]
        if all(n is o for n, o in zip(new_items, items)):
            return value
        if isinstance(value, dict):
            return dict(zip(value.keys(), new_items))
        if isinstance(value, tuple) and hasattr(value, "_fields"):
            return type(value)(*new_items)

        return type(value)(new_items)

    return value


def _copy_with(node: Node, changes: dict[str, Any]) -> Any:
    new = object.__new__(type(node))
    for f in dataclasses.fields(node):
        value = changes.get(f.name, getattr(node, f.name))
        object.__setattr__(new, f.name, value)

    object.__setattr__(new, _LAYOUT, getattr(node, _LAYOUT))
    return new


def _set_layout(cls: type[Node]) -> None:
    fields = dataclasses.fields(cls)
    children, contexts, keep = [], [], set()
    for f in fields:
        if f.name in _RESERVED or f.name == _LAYOUT:
            raise NodeLayoutError(
                f"{cls.__qualname__}.{f.name}: '{f.name}' is reserved by Node."
            )
        if f.name.startswith("__"):
            raise NodeLayoutError(
                f"{cls.__qualname__}.{f.name}: field names must not start "
                "with two underscores."
            )

        kind = f.metadata.get(_KIND)
        if kind == _CHILD:
            children.append(f.name)
            if f.metadata.get(_KEEP_DTYPE, False):
                keep.add(f.name)
        elif kind == _CONTEXT:
            contexts.append(f.name)
        else:
            raise NodeLayoutError(
                f"{cls.__qualname__}.{f.name}: declare the field with "
                "`child()` or `context()`."
            )

    field_names = {f.name for f in fields}
    annotated: set[str] = set()
    for klass in cls.__mro__:
        if klass in (object, Node) or not issubclass(klass, Node):
            continue
        for name, annotation in inspect.get_annotations(klass).items():
            if not _is_classvar(annotation):
                annotated.add(name)

    missing = sorted(annotated - field_names)
    if missing:
        raise NodeLayoutError(
            f"{cls.__qualname__}: annotated names {missing} are not fields."
        )

    cls._child_names = tuple(children)
    cls._context_names = tuple(contexts)
    cls._keep_dtype_names = frozenset(keep)


def _wrap_init(cls: type[Node]) -> None:
    generated = cls.__init__

    @functools.wraps(generated)
    def __init__(self: Node, *args: Any, **kwargs: Any) -> None:
        generated(self, *args, **kwargs)
        if type(self) is cls:
            _finish_init(self)

    cls.__init__ = __init__  # type: ignore[method-assign, assignment]


def _finish_init(node: Node) -> None:
    updates = node._normalize()
    names = {f.name for f in dataclasses.fields(node)}
    for name, value in updates.items():
        if name not in names:
            raise NodeLayoutError(
                f"{type(node).__qualname__}._normalize returned unknown "
                f"field '{name}'."
            )

        object.__setattr__(node, name, value)

    if not is_compiling():
        _check_values(node)

    object.__setattr__(node, _LAYOUT, _leaf_fields_from_values(node))
    node._validate()


def _check_values(node: Node) -> None:
    cls_name = type(node).__qualname__
    for name in node._context_names:
        value = getattr(node, name)
        if _holds_array(value):
            raise TypeError(
                f"{cls_name}.{name} is a context field but holds a tensor or "
                "Node; declare it with `child()`."
            )

        _require_hashable(cls_name, name, value)

    for name in node._child_names:
        value = getattr(node, name)
        if _is_container(value):
            if not _container_only_arrays(value):
                raise TypeError(
                    f"{cls_name}.{name}: a container in a child field may "
                    "only hold tensors, Nodes and containers of them."
                )
        elif not isinstance(value, (Tensor, Node)) and value is not None:
            _require_hashable(cls_name, name, value)

    tensors: list[Tensor] = []
    for name in node._child_names:
        if name not in node._keep_dtype_names:
            _collect(getattr(node, name), Tensor, tensors)

    floating = {t.dtype for t in tensors if t.is_floating_point()}
    if len(floating) > 1:
        raise TypeError(
            f"{cls_name}: floating-point tensors have different dtypes "
            f"{sorted(str(d) for d in floating)}."
        )

    devices = {t.device for t in _own_tensors(node)}
    for sub in _child_nodes(node):
        try:
            devices.add(sub.device)
        except AttributeError:
            pass  # object without a device: nothing to compare

    if len(devices) > 1:
        raise RuntimeError(
            f"{cls_name}: tensors are on different devices "
            f"{sorted(str(d) for d in devices)}."
        )


def _require_hashable(cls_name: str, name: str, value: Any) -> None:
    try:
        hash(value)
    except TypeError as exc:
        raise TypeError(
            f"{cls_name}.{name} must be hashable (it becomes part of the "
            f"pytree context), but {type(value).__name__} is not."
        ) from exc


# -- pytree registration --------------------------------------------------


def _register(cls: type[Node]) -> None:
    def flatten_with_keys(node: Node) -> tuple[list[tuple[Any, Any]], Any]:
        values, context_ = _split(node)
        names = context_[0]
        keyed = [(pytree.GetAttrKey(n), v) for n, v in zip(names, values)]
        return keyed, context_

    def unflatten(values: Any, context_: Any) -> Node:
        array_names, static_items, context_values = context_
        node = object.__new__(cls)
        for name, value in zip(array_names, values):
            object.__setattr__(node, name, value)
        for name, value in static_items:
            object.__setattr__(node, name, value)
        for name, value in zip(cls._context_names, context_values):
            object.__setattr__(node, name, value)
        object.__setattr__(node, _LAYOUT, array_names)
        return node

    pytree.register_pytree_node(
        cls,
        _split,
        unflatten,
        serialized_type_name=f"{cls.__module__}.{cls.__qualname__}",
        flatten_with_keys_fn=flatten_with_keys,
    )


def _leaf_fields_from_values(node: Node) -> tuple[str, ...]:
    """Child fields whose values become pytree leaves. Called once, when a
    node is constructed; afterwards the result is stored on the node."""
    return tuple(
        name
        for name in node._child_names
        if _is_array_value(getattr(node, name))
    )


def _split(node: Node) -> tuple[list[Any], Any]:
    # The leaf/static split is read from the node, never re-derived from
    # the current values: `vmap` unflattens a treespec with placeholder
    # leaves (plain ints) and flattens the result again, which must give
    # back the same structure.
    array_names: tuple[str, ...] = getattr(node, _LAYOUT)
    values = [getattr(node, name) for name in array_names]
    static_items = tuple(
        (name, getattr(node, name))
        for name in node._child_names
        if name not in array_names
    )
    context_values = tuple(getattr(node, n) for n in node._context_names)
    return values, (array_names, static_items, context_values)
