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
Tree: NodeFunction
==================

Custom autograd functions whose arguments may contain :class:`Node` objects.

``torch.autograd.Function.apply`` only tracks tensors that are passed to it
directly. A tensor inside a :class:`Node` (or a list or dict) is invisible to
it, so its gradient is silently lost. A :class:`NodeFunction` is written like
a ``torch.autograd.Function``, with the static methods ``forward``,
``setup_context``, ``backward`` and optionally ``jvp`` and ``vmap``, but
``apply`` hands torch the tensor leaves of the arguments, and the methods see
the arguments structured as they were passed:

- ``forward(*args)`` and ``setup_context(ctx, inputs, output)`` get the
  arguments of ``apply``. The output must be a tensor or a tuple of tensors.
- ``backward(ctx, *grad_outputs)`` returns one gradient per argument:
  ``None``, a tensor for a tensor, or for a Node (or a list, tuple or dict)
  a dict of the gradients of its fields, nested like the fields. Fields that
  are left out have no gradient. ``ctx.needs_input_grad`` holds the flags in
  the same structure, e.g. ``ctx.needs_input_grad[0].positions``.
- ``jvp(ctx, *tangents)`` gets the tangents structured like the arguments,
  with ``None`` for leaves without a tangent, and returns the tangents of the
  outputs.
- ``vmap(info, in_dims, *args)`` gets ``in_dims`` structured like the
  arguments. Alternatively, set ``generate_vmap_rule = True``.

:meth:`NodeFunction.save` and :meth:`NodeFunction.saved` keep Nodes (and other
pytrees) from ``setup_context`` for ``backward`` and ``jvp``.

Example
-------
>>> import torch
>>> from tad_mctc.tree import Node, NodeFunction, child
>>> class Pair(Node):
...     x: torch.Tensor = child()
...     y: torch.Tensor = child()
>>> class Product(NodeFunction):
...     @staticmethod
...     def forward(pair):
...         return pair.x * pair.y
...
...     @staticmethod
...     def setup_context(ctx, inputs, output):
...         NodeFunction.save(ctx, inputs[0])
...
...     @staticmethod
...     def backward(ctx, grad):
...         (pair,) = NodeFunction.saved(ctx)
...         return {"x": grad * pair.y, "y": grad * pair.x}
>>> x = torch.tensor(2.0, requires_grad=True)
>>> Product.apply(Pair(x=x, y=torch.tensor(3.0))).backward()
>>> x.grad
tensor(3.)

``torch.compile`` (as of PyTorch 2.14) cannot trace an autograd function with
a custom ``jvp`` while gradients are recorded and breaks the graph there: it
runs the function eagerly, and ``fullgraph=True`` fails. From PyTorch 2.7 on,
a :class:`NodeFunction` with ``jvp`` adds an explanation to this graph break.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

import torch
from torch import Tensor
from torch.utils import _pytree as pytree

from .._version import __tversion__
from ..tools import is_compiling

__all__ = ["NodeFunction"]


# Dynamo cannot trace a `torch.autograd.Function` with a custom `jvp` while
# gradients are recorded ("Unsupported custom jvp", as of PyTorch 2.14), so
# compiling a function and differentiating it in reverse mode fails. Compiled
# forward mode and code without gradients work. Assumed to be fixed in a later
# release; raise the bound once a release is checked. A plain Python
# constant, read when `apply` runs, so it never becomes part of a graph.
_COMPILE_FAILS_WITH_CUSTOM_JVP = __tversion__ < "2.15.0"

# `torch._dynamo.graph_break` takes the explanation from PyTorch 2.7 on.
# Before, Dynamo's own graph break ("NYI - User defind jvp") is kept.
_GRAPH_BREAK_TAKES_MESSAGE = __tversion__ >= "2.7.0"

_TENSOR = object()
"""Placeholder for a saved tensor (see `NodeFunction.save`)."""


class NodeFunction:
    """
    Base class for custom autograd functions with :class:`Node` arguments
    (see module docstring).

    Subclasses define the static methods of a ``torch.autograd.Function``;
    the ones they leave out keep torch's defaults.
    """

    generate_vmap_rule: ClassVar[bool] = False
    """Let torch generate the ``vmap`` rule (as ``torch.autograd.Function``)."""

    _function: ClassVar[type[torch.autograd.Function]]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls._function = _flat_function(cls)

    @classmethod
    def apply(cls, *args: Any) -> Any:
        """
        Evaluate the function, recording it for differentiation.

        Parameters
        ----------
        *args : Any
            Arguments of ``forward``: tensors, Nodes, containers of them and
            any other values.

        Returns
        -------
        Any
            The output of ``forward``.
        """
        leaves, spec = pytree.tree_flatten(args)
        if _explains_compiled_jvp() and _defines(cls, "jvp"):
            _check_compiled_jvp(cls, leaves)
        return cls._function.apply(spec, *leaves)

    # -- the methods of torch.autograd.Function ---------------------------

    @staticmethod
    def forward(*args: Any) -> Any:
        """Evaluate the function (see ``torch.autograd.Function``)."""
        raise NotImplementedError("A NodeFunction must define `forward`.")

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:
        """Save what the derivatives need (see ``torch.autograd.Function``)."""

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        """Gradients of the arguments (see module docstring)."""
        raise NotImplementedError

    @staticmethod
    def jvp(ctx: Any, *tangents: Any) -> Any:
        """Tangents of the outputs (see module docstring)."""
        raise NotImplementedError

    @staticmethod
    def vmap(info: Any, in_dims: tuple[Any, ...], *args: Any) -> Any:
        """Batching rule (see module docstring)."""
        raise NotImplementedError

    # -- saving structured inputs -----------------------------------------

    @staticmethod
    def save(ctx: Any, *objects: Any) -> None:
        """
        Save tensors, Nodes, containers of them and other values in
        ``setup_context``, for ``backward`` and ``jvp``.

        Parameters
        ----------
        ctx : Any
            The context of ``setup_context``.
        *objects : Any
            The objects to save.
        """
        leaves, spec = pytree.tree_flatten(objects)
        tensors = [leaf for leaf in leaves if isinstance(leaf, Tensor)]
        ctx.save_for_backward(*tensors)
        ctx.save_for_forward(*tensors)
        others = [_TENSOR if isinstance(x, Tensor) else x for x in leaves]
        ctx.node_function_saved = (others, spec)

    @staticmethod
    def saved(ctx: Any) -> tuple[Any, ...]:
        """
        The objects of :meth:`save`, in ``backward`` or ``jvp``.

        Parameters
        ----------
        ctx : Any
            The context of ``backward`` or ``jvp``.

        Returns
        -------
        tuple[Any, ...]
            The saved objects.
        """
        others, spec = ctx.node_function_saved
        tensors = iter(ctx.saved_tensors)
        leaves = [next(tensors) if x is _TENSOR else x for x in others]
        return pytree.tree_unflatten(leaves, spec)


def _defines(cls: type[NodeFunction], name: str) -> bool:
    """Whether a subclass overrides the method `name`"""
    return getattr(cls, name) is not getattr(NodeFunction, name)


def _flat_function(cls: type[NodeFunction]) -> type[torch.autograd.Function]:
    """
    The ``torch.autograd.Function`` of a NodeFunction: its inputs are the
    pytree spec of the arguments and their leaves.
    """

    def forward(*args: Any) -> Any:
        # Without gradients, Dynamo passes the context first to a `forward`
        # without fixed parameters (as of PyTorch 2.14)
        if not isinstance(args[0], pytree.TreeSpec):
            args = args[1:]
        spec, *leaves = args
        output = cls.forward(*pytree.tree_unflatten(leaves, spec))
        if not isinstance(output, Tensor) and not (
            isinstance(output, tuple)
            and all(isinstance(out, Tensor) for out in output)
        ):
            raise TypeError(
                f"{cls.__qualname__}.forward must return a tensor or a tuple "
                f"of tensors, not {type(output).__name__}."
            )
        return output

    def setup_context(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:
        spec, *leaves = inputs
        ctx.node_function_spec = spec
        cls.setup_context(ctx, pytree.tree_unflatten(leaves, spec), output)

    def backward(ctx: Any, *grad_outputs: Any) -> tuple[Any, ...]:
        spec = ctx.node_function_spec
        flags = list(ctx.needs_input_grad[1:])
        ctx.needs_input_grad = pytree.tree_unflatten(flags, spec)
        grads = cls.backward(ctx, *grad_outputs)
        if not isinstance(grads, tuple):
            grads = (grads,)
        return (None, *_leaf_gradients(cls, grads, spec))

    def jvp(ctx: Any, *tangents: Any) -> Any:
        spec = ctx.node_function_spec
        return cls.jvp(ctx, *pytree.tree_unflatten(list(tangents[1:]), spec))

    def vmap(info: Any, in_dims: tuple[Any, ...], *args: Any) -> Any:
        spec, *leaves = args
        dims = pytree.tree_unflatten(list(in_dims[1:]), spec)
        return cls.vmap(info, dims, *pytree.tree_unflatten(leaves, spec))

    namespace: dict[str, Any] = {
        "__module__": cls.__module__,
        "__qualname__": f"{cls.__qualname__}.Function",
        "forward": staticmethod(forward),
        "setup_context": staticmethod(setup_context),
        "generate_vmap_rule": cls.generate_vmap_rule,
    }
    # only what the subclass defines, so that torch keeps its defaults
    optional: dict[str, Callable[..., Any]] = {
        "backward": backward,
        "jvp": jvp,
        "vmap": vmap,
    }
    for name, method in optional.items():
        if _defines(cls, name):
            namespace[name] = staticmethod(method)

    return type(cls.__name__, (torch.autograd.Function,), namespace)


def _leaf_gradients(
    cls: type[NodeFunction], grads: tuple[Any, ...], spec: pytree.TreeSpec
) -> list[Any]:
    """
    The gradients of `backward` (one per argument, structured as described
    in the module docstring) as one gradient per leaf of the arguments.
    """
    # position of each leaf in the arguments, keyed by its path
    numbered = pytree.tree_unflatten(list(range(spec.num_leaves)), spec)
    leaf_of = {
        _path(path): leaf
        for path, leaf in pytree.tree_flatten_with_path(numbered)[0]
    }

    out: list[Any] = [None] * spec.num_leaves
    for path, grad in pytree.tree_flatten_with_path(grads)[0]:
        if grad is None:
            continue
        key = _path(path)
        if key not in leaf_of:
            raise TypeError(
                f"{cls.__qualname__}.backward returned a gradient at "
                f"{key}, which is no tensor of the arguments. Return one "
                "gradient per argument: None, a tensor, or a dict of the "
                "gradients of a Node's fields."
            )
        out[leaf_of[key]] = grad
    return out


def _path(path: tuple[Any, ...]) -> tuple[Any, ...]:
    """A key path with Node fields and dict keys alike"""
    parts: list[Any] = []
    for key in path:
        if isinstance(key, pytree.GetAttrKey):
            parts.append(key.name)
        elif isinstance(key, pytree.MappingKey):
            parts.append(key.key)
        elif isinstance(key, pytree.SequenceKey):
            parts.append(key.idx)
        else:
            parts.append(key)
    return tuple(parts)


def _explains_compiled_jvp() -> bool:
    """Whether `apply` explains the graph break of a custom `jvp`"""
    return _COMPILE_FAILS_WITH_CUSTOM_JVP and _GRAPH_BREAK_TAKES_MESSAGE


def _check_compiled_jvp(cls: type[NodeFunction], leaves: list[Any]) -> None:
    """
    Break the graph with an explanation if `torch.compile` cannot trace `cls`
    here (see constant). A graph break, not an error: without ``fullgraph``,
    the function then runs eagerly, as it would without this check.
    """
    if not (is_compiling() and torch.is_grad_enabled()):
        return

    if any(isinstance(x, Tensor) and x.requires_grad for x in leaves):
        torch._dynamo.graph_break(  # pylint: disable=protected-access
            msg=(
                f"{cls.__qualname__} defines a custom `jvp`, and torch.compile "
                "cannot trace autograd functions with a custom `jvp` while "
                f"gradients are recorded (PyTorch {torch.__version__}). "
                "Compile only code without gradients or with forward mode "
                "(torch.func.jvp, jacfwd), and differentiate in reverse mode "
                "outside of torch.compile."
            )
        )
