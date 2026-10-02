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
Test the resolution and caching of per-element tables.
"""

from __future__ import annotations

import functools
import gc

import torch

from tad_mctc.data import radii
from tad_mctc.data.table import _TABLE_CACHE, resolve_table


def _table(
    *, device: torch.device | None = None, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    return torch.arange(10, device=device, dtype=dtype)


def test_resolve_table_reuses_cached_tensor_for_same_table_device_dtype() -> (
    None
):
    """Two calls with the same `TableFunction` and the same `like` device/
    dtype must not re-invoke the table and must return the very same
    tensor object."""
    calls: list[tuple[torch.device | None, torch.dtype]] = []

    def spy_table(
        *,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        calls.append((device, dtype))
        return torch.arange(10, device=device, dtype=dtype)

    like = torch.zeros(3, dtype=torch.double)

    first = resolve_table(spy_table, like)
    second = resolve_table(spy_table, like)

    assert len(calls) == 1
    assert first is second


def test_resolve_table_rebuilds_for_different_dtype() -> None:
    """A cache keyed only by table identity (ignoring device/dtype) would
    wrongly hand back a float32 table when a float64 one is asked for."""
    calls: list[tuple[torch.device | None, torch.dtype]] = []

    def spy_table(
        *,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        calls.append((device, dtype))
        return torch.arange(10, device=device, dtype=dtype)

    like_f32 = torch.zeros(3, dtype=torch.float32)
    like_f64 = torch.zeros(3, dtype=torch.float64)

    first = resolve_table(spy_table, like_f32)
    second = resolve_table(spy_table, like_f64)

    assert len(calls) == 2
    assert first.dtype == torch.float32
    assert second.dtype == torch.float64


def test_resolve_table_tensor_is_moved_not_cached() -> None:
    """A tensor is only moved to the device and dtype of `like`."""
    table = torch.arange(10)
    out = resolve_table(table, torch.zeros(3, dtype=torch.double))

    assert out.dtype == torch.double
    assert torch.equal(out, table.double())


def test_cache_entry_lives_as_long_as_function() -> None:
    """A module-level table stays cached; a local one is dropped with it."""
    like = torch.zeros(3, dtype=torch.double)

    def local_table(
        *, device: torch.device | None = None, dtype: torch.dtype
    ) -> torch.Tensor:
        return torch.arange(10, device=device, dtype=dtype)

    _TABLE_CACHE.clear()
    resolve_table(radii.COV_D3, like)
    resolve_table(local_table, like)
    assert len(_TABLE_CACHE) == 2

    del local_table
    gc.collect()

    assert list(_TABLE_CACHE.keys()) == [radii.COV_D3]


def test_cache_does_not_grow_with_new_functions() -> None:
    """A new lambda or `functools.partial` per call must not grow the cache:
    its entry goes away once the function is freed."""
    like = torch.zeros(3, dtype=torch.double)
    _TABLE_CACHE.clear()

    for i in range(50):
        out = resolve_table(
            lambda *, device=None, dtype, i=i: torch.full(
                (10,), float(i), device=device, dtype=dtype
            ),
            like,
        )
        assert out[0] == i  # never a stale hit from an earlier function

        resolve_table(functools.partial(_table), like)

    gc.collect()
    assert len(_TABLE_CACHE) == 0


def test_not_weakly_referenceable_is_not_cached() -> None:
    """A callable without weak reference support is called every time."""

    class Table:  # pylint: disable=too-few-public-methods
        __slots__ = ("calls",)

        def __init__(self) -> None:
            self.calls = 0

        def __call__(
            self, *, device: torch.device | None = None, dtype: torch.dtype
        ) -> torch.Tensor:
            self.calls += 1
            return torch.arange(10, device=device, dtype=dtype)

    table = Table()
    like = torch.zeros(3, dtype=torch.double)
    resolve_table(table, like)
    resolve_table(table, like)

    assert table.calls == 2
