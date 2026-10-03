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
Data: Tables
============

Resolution of per-element tables, indexed by atomic number, that are given
either as a :class:`Tensor` or as a :class:`~tad_mctc.typing.TableFunction`
such as :func:`tad_mctc.data.radii.COV_D3`.
"""

from __future__ import annotations

import weakref

import torch

from ..autograd.unwrap import unwrap_gradtracking
from ..tools import is_compiling
from ..typing import TableFunction, Tensor

__all__ = ["resolve_table"]


# Cache of the tensors built by a `TableFunction`, per device and dtype.
#
# The keys are weak references to the table functions: an entry lives only
# as long as its function. A function created anew for every call (a
# lambda, a `functools.partial`) therefore cannot grow the cache without
# bound, and its entry is dropped as soon as the function is freed. The
# module-level tables such as `radii.COV_D3` live for the whole process,
# and so do their entries.
#
# Weak keys also rule out a stale hit: unlike a bare `id()`, a weak
# reference is never mistaken for another object that later reuses the
# same address.
_TABLE_CACHE: weakref.WeakKeyDictionary[
    TableFunction, dict[tuple[torch.device, torch.dtype], Tensor]
] = weakref.WeakKeyDictionary()


def resolve_table(table: Tensor | TableFunction, like: Tensor) -> Tensor:
    """
    Resolve a per-element table on the device and dtype of `like`.

    A :class:`Tensor` is just moved with :meth:`Tensor.to` (a no-op if it is
    already on the right device and dtype). A
    :class:`~tad_mctc.typing.TableFunction` is called once per device and
    dtype, and the table it builds is cached for as long as the function
    itself exists.

    The cache is bypassed while ``torch.compile`` traces, where the table is
    simply built as part of the graph, and for functions that cannot be
    weakly referenced, which are called anew every time.

    Parameters
    ----------
    table : Tensor | TableFunction
        Either an already-built table, or a function such as
        :func:`tad_mctc.data.radii.COV_D3` that builds one for a given
        device and dtype.
    like : Tensor
        Tensor whose device and dtype the table is resolved to.

    Returns
    -------
    Tensor
        The table, one entry per atomic number. Callers must treat it as
        read-only: it is shared, via the cache, with every other call
        site that resolves the same table on the same device and dtype.
    """
    if isinstance(table, Tensor):
        return table.to(device=like.device, dtype=like.dtype)

    # A cold cache filled while tracing would make the table an output of
    # the graph. Inside a `torch.func` transform (e.g.
    # `torch.compile(jacrev(...))`), that output is a functorch wrapper,
    # which fails to compile with "Cannot access storage of TensorWrapper".
    # Without the cache, the table is just a constant of the graph.
    if is_compiling():
        return table(device=like.device, dtype=like.dtype)

    try:
        per_table = _TABLE_CACHE.setdefault(table, {})
    except TypeError:  # not weakly referenceable
        return table(device=like.device, dtype=like.dtype)

    key = (like.device, like.dtype)
    cached = per_table.get(key)
    if cached is None:
        # A table built inside a `jacrev` is wrapped at that level, and would
        # escape it from the cache (see `unwrap_gradtracking`).
        cached = unwrap_gradtracking(
            table(device=like.device, dtype=like.dtype)
        )
        per_table[key] = cached
    return cached
