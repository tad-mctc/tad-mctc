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
Tests of :class:`tad_mctc.ncoord.common.CNModel` that hold for every
evaluation path: the presets, call-time validation, the CN cap's numerics,
the per-element tables and their cache, and derivatives with respect to
model parameters. The plain molecular `__call__` serves as the vehicle.

Tests specific to one path live in the quadrant modules
(`test_dense_molecular.py`, `test_dense_periodic.py`,
`test_sparse_molecular.py`, `test_sparse_periodic.py`, and `test_sparse.py`
for both sparse geometries). Per-preset correctness against the mctc-lib
Fortran references lives in `test_reference.py` and `test_grad/`.
"""

from __future__ import annotations

import pytest
import torch
from torch.func import vmap

from tad_mctc._version import __tversion__
from tad_mctc.autograd import jacrev_matches_finite_diff
from tad_mctc.data import radii
from tad_mctc.data.structures import get_structure
from tad_mctc.io.structure import Structure
from tad_mctc.ncoord import defaults
from tad_mctc.ncoord.common import CNModel, cut_coordination_number
from tad_mctc.ncoord.count import erf_count
from tad_mctc.ncoord.d3 import cn_d3
from tad_mctc.ncoord.d4 import cn_d4, d4_en_weight
from tad_mctc.ncoord.eeq import cn_eeq, cn_eeq_en
from tad_mctc.ncoord.eeqbc import cn_eeqbc, cn_eeqbc_en
from tad_mctc.ncoord.gfn2 import cn_gfn2

from ..utils import (
    DYNAMO_SUPPORTED,
    DYNAMO_UNSUPPORTED_REASON,
    compile_fullgraph,
    jacrev,
    run_compiled_or_skip,
)

# ---------------------------------------------------------------------------
# Presets: fields match the spec exactly


def test_preset_fields_match_spec() -> None:
    assert cn_d3.cutoff == defaults.CUTOFF_D3
    assert cn_d4.cutoff == defaults.CUTOFF_D4
    assert cn_d4.pair_weight is d4_en_weight
    assert cn_eeq.cutoff == defaults.CUTOFF_EEQ
    assert cn_eeq.cn_max == defaults.CUTOFF_EEQ_MAX
    assert cn_eeq_en.cutoff == defaults.CUTOFF_EEQ
    assert cn_eeq_en.cn_max is None
    assert cn_eeqbc.cutoff == defaults.CUTOFF_EEQBC
    assert cn_eeqbc.cn_max is None
    assert cn_eeqbc_en.pair_weight is not None
    assert cn_gfn2.cutoff == defaults.CUTOFF_GFN2


def test_non_scalar_cn_max_raises() -> None:
    """A per-atom cap is rejected when the model is built, not on each
    call, and also for a variant made with `replace`."""
    with pytest.raises(ValueError, match="cn_max"):
        CNModel(count=erf_count, cutoff=5.0, cn_max=torch.tensor([1.0, 2.0]))

    with pytest.raises(ValueError, match="cn_max"):
        cn_eeq.replace(cn_max=torch.tensor([1.0, 2.0]))


# ---------------------------------------------------------------------------
# CN cap numerics


def test_cn_cap_tensor_matches_float() -> None:
    """A 0-d tensor `cn_max` must give the same cap as the equal Python
    float, on either device/dtype combination."""
    cn = torch.tensor([1.0, 5.0, 12.0], dtype=torch.double)

    capped_float = cut_coordination_number(cn, 8.0)
    capped_tensor = cut_coordination_number(cn, torch.tensor(8.0))

    assert torch.allclose(capped_float, capped_tensor)


def test_cn_cap_large_float_is_noop() -> None:
    """The `> 50` shortcut only applies to a plain Python number and
    disables the cap exactly."""
    cn = torch.tensor([1.0, 5.0, 100.0], dtype=torch.double)
    assert torch.equal(cut_coordination_number(cn, 100.0), cn)


def test_cn_cap_checks_is_not_none_not_truthiness() -> None:
    """A `cn_max` of exactly `0` is a valid (if degenerate) cap, not a
    falsy sentinel meaning "no cap" -- `CNModel` must test `is not None`."""
    model_zero_cap = CNModel(count=erf_count, cutoff=25.0, cn_max=0.0)
    numbers = torch.tensor([6, 1, 1, 1, 1])
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, -1.0, -1.0],
        ],
        dtype=torch.double,
    )
    structure = Structure(numbers=numbers, positions=positions)

    cn = model_zero_cap(structure)
    # `cut_coordination_number(cn, 0.0)` is finite and strictly below the
    # uncapped value for any positive `cn`, unlike an accidentally skipped
    # cap (`if self.cn_max:` would treat `0.0` as "no cap" and return `cn`
    # unmodified).
    uncapped = model_zero_cap.replace(cn_max=None)(structure)
    assert torch.isfinite(cn).all()
    assert torch.all(cn < uncapped)


# ---------------------------------------------------------------------------


def test_dispatch_without_pair_weight_skips_en() -> None:
    """`en` is only read when `pair_weight` is set (see `CNModel`'s own
    docstring), so evaluating a model with `pair_weight=None` must not
    resolve `en` -- not even build the tensor."""
    en_calls: list[tuple[torch.device | None, torch.dtype]] = []

    def en_table(
        *,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        en_calls.append((device, dtype))
        return torch.arange(10, device=device, dtype=dtype)

    numbers = torch.tensor([1, 1])
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.double
    )
    structure = Structure(numbers=numbers, positions=positions)
    model = CNModel(count=erf_count, cutoff=5.0, en=en_table)

    model(structure)

    assert en_calls == []


def test_dispatch_with_pair_weight_resolves_en_like_rcov() -> None:
    """With `pair_weight` set, `en` must be resolved on `positions`'
    device/dtype, same as `rcov`."""
    en_calls: list[tuple[torch.device | None, torch.dtype]] = []

    def en_table(
        *,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        en_calls.append((device, dtype))
        return torch.arange(10, device=device, dtype=dtype)

    numbers = torch.tensor([1, 1])
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.double
    )
    structure = Structure(numbers=numbers, positions=positions)
    model = CNModel(
        count=erf_count,
        cutoff=5.0,
        en=en_table,
        pair_weight=lambda en_i, en_j: en_i - en_j,
    )

    model(structure)

    assert en_calls == [(positions.device, positions.dtype)]


def test_table_cache_is_shared_across_replace() -> None:
    """The cache lives at function level, not on the `CNModel` instance, so
    `.replace()` copies (the documented way to get a different variant)
    share it rather than each rebuilding their own table."""
    calls: list[tuple[torch.device | None, torch.dtype]] = []

    def spy_table(
        *,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        calls.append((device, dtype))
        return radii.COV_D3(device=device, dtype=dtype)

    model = CNModel(count=erf_count, cutoff=25.0, rcov=spy_table)
    other = model.replace(cutoff=40.0)

    numbers = torch.tensor([6, 1, 1, 1, 1])
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, -1.0, -1.0],
        ],
        dtype=torch.double,
    )
    structure = Structure(numbers=numbers, positions=positions)

    first = model(structure)
    second = other(structure)

    assert len(calls) == 1
    assert torch.allclose(first, second)


def _cached_tables() -> list[torch.Tensor]:
    """All tensors in the table cache, across tables, devices and dtypes."""
    from tad_mctc.data.table import _TABLE_CACHE

    return [
        t for per_table in _TABLE_CACHE.values() for t in per_table.values()
    ]


def test_table_cache_survives_cold_fill_under_vmap() -> None:
    """The first resolution of a `TableFunction` (a cold cache) happening
    *while* ``torch.func.vmap`` is tracing must cache a plain `Tensor`, not a
    functorch batched-tensor wrapper -- `radii.COV_D3()` builds a table with
    no data dependency on the vmapped argument, so it is never wrapped, but
    this pins that behaviour down as a regression test."""
    from tad_mctc.data.table import _TABLE_CACHE

    sample = get_structure("mb16_43", "01")
    numbers = sample.numbers
    positions = sample.positions.double()
    structure = Structure(numbers=numbers, positions=positions)

    _TABLE_CACHE.clear()

    def f(scale: torch.Tensor) -> torch.Tensor:
        return cn_d3(structure) * scale

    batch = torch.tensor([1.0, 1.01, 1.02], dtype=torch.double)
    batched = vmap(f)(batch)
    looped = torch.stack([f(s) for s in batch])
    assert torch.allclose(batched, looped)

    assert len(_TABLE_CACHE) > 0
    for cached in _cached_tables():
        assert type(cached) is torch.Tensor

    # The cached tensor must remain usable in a plain eager call afterward.
    eager = cn_d3(structure)
    assert torch.allclose(eager, f(torch.tensor(1.0, dtype=torch.double)))


def test_table_cache_survives_cold_fill_under_jacrev() -> None:
    """A cold cache filled while ``torch.func.jacrev`` is active must cache
    the plain table, not the grad-tracking wrapper created at that level
    (`type()` does not tell them apart). Otherwise, a later Hessian, i.e. a
    different transform stack, fails with "INTERNAL ASSERT FAILED ...
    escaped?"."""
    from tad_mctc.data.table import _TABLE_CACHE

    sample = get_structure("mb16_43", "01")
    numbers = sample.numbers
    positions = sample.positions.double()

    def energy(p: torch.Tensor) -> torch.Tensor:
        return cn_d3(Structure(numbers=numbers, positions=p)).sum()

    hess = jacrev(jacrev(energy))

    _TABLE_CACHE.clear()
    first = hess(positions)

    assert len(_TABLE_CACHE) > 0
    for cached in _cached_tables():
        assert not getattr(torch._C, "_functorch").is_functorch_wrapped_tensor(
            cached
        )

    # A second Hessian and a plain call reuse the cached table.
    assert torch.allclose(hess(positions), first)
    assert torch.allclose(jacrev(energy)(positions), jacrev(energy)(positions))
    energy(positions)


@pytest.mark.skipif(not DYNAMO_SUPPORTED, reason=DYNAMO_UNSUPPORTED_REASON)
def test_table_cache_not_filled_under_compile() -> None:
    """A cold cache is left cold while `torch.compile(fullgraph=True)` is
    tracing (see `test_table_cache_cold_under_compiled_jacrev` for why), and
    the next eager call fills it with a plain `Tensor`."""
    from tad_mctc.data.table import _TABLE_CACHE

    sample = get_structure("mb16_43", "01")
    numbers = sample.numbers
    positions = sample.positions.double()

    _TABLE_CACHE.clear()
    torch._dynamo.reset()  # pylint: disable=protected-access

    def f(p: torch.Tensor) -> torch.Tensor:
        return cn_d3(Structure(numbers=numbers, positions=p))

    compiled_value = run_compiled_or_skip(f, positions)
    assert len(_TABLE_CACHE) == 0

    eager_value = f(positions)
    assert torch.allclose(compiled_value, eager_value)

    assert len(_TABLE_CACHE) > 0
    for cached in _cached_tables():
        assert type(cached) is torch.Tensor


@pytest.mark.skipif(not DYNAMO_SUPPORTED, reason=DYNAMO_UNSUPPORTED_REASON)
@pytest.mark.skipif(
    __tversion__ < (2, 5, 0),
    reason="`torch.compile` of `torch.func` transforms needs PyTorch 2.5.0.",
)
@pytest.mark.parametrize("warm", [False, True])
def test_table_cache_cold_under_compiled_jacrev(warm: bool) -> None:
    """`torch.compile(jacrev(...))` with a `TableFunction`. Filling a cold
    cache while tracing would make the table a graph output, which inside
    the transform is a functorch wrapper and fails to compile ("Cannot
    access storage of TensorWrapper"). Called directly rather than through
    `run_compiled_or_skip`, which would turn that failure into a skip."""
    from tad_mctc.data.table import _TABLE_CACHE

    sample = get_structure("mb16_43", "01")
    numbers = sample.numbers
    positions = sample.positions.double()

    def energy(p: torch.Tensor) -> torch.Tensor:
        return cn_d3(Structure(numbers=numbers, positions=p)).sum()

    grad = jacrev(energy)
    batch = torch.stack([positions, positions * 1.01])

    for fn, arg in ((grad, positions), (vmap(grad), batch)):
        _TABLE_CACHE.clear()
        torch._dynamo.reset()  # pylint: disable=protected-access
        if warm:
            energy(positions)

        compiled = compile_fullgraph(fn)(arg)
        assert torch.allclose(compiled, fn(arg))


# ---------------------------------------------------------------------------
# Tables: a tensor and its matching table function give the same result


def test_tensor_table_matches_table_function() -> None:
    sample = get_structure("mb16_43", "01")
    numbers = sample.numbers
    positions = sample.positions.double()
    structure = Structure(numbers=numbers, positions=positions)

    as_function = CNModel(count=erf_count, cutoff=25.0, rcov=radii.COV_D3)
    as_tensor = CNModel(
        count=erf_count, cutoff=25.0, rcov=radii.COV_D3(dtype=torch.double)
    )

    assert torch.allclose(as_function(structure), as_tensor(structure))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_model_follows_input_dtype(dtype: torch.dtype) -> None:
    model = CNModel(count=erf_count, cutoff=25.0)
    numbers = torch.tensor([6, 1, 1, 1, 1])
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, -1.0, -1.0],
        ],
        dtype=dtype,
    )
    structure = Structure(numbers=numbers, positions=positions)
    assert model(structure).dtype == dtype


# ---------------------------------------------------------------------------
# `cn_max` as a tensor: matches the float preset, and survives jacrev/vmap/compile


def test_cn_eeq_tensor_cn_max_matches_float() -> None:
    sample = get_structure("mb16_43", "01")
    numbers = sample.numbers
    positions = sample.positions.double()
    structure = Structure(numbers=numbers, positions=positions)

    float_result = cn_eeq(structure)
    tensor_result = cn_eeq.replace(cn_max=torch.tensor(cn_eeq.cn_max))(
        structure
    )
    assert torch.allclose(float_result, tensor_result)


def test_cn_eeq_jacrev_wrt_cn_max_matches_finite_differences() -> None:
    sample = get_structure("mb16_43", "01")
    numbers = sample.numbers
    positions = sample.positions.double()
    structure = Structure(numbers=numbers, positions=positions)

    def f(cn_max: torch.Tensor) -> torch.Tensor:
        return cn_eeq.replace(cn_max=cn_max)(structure)

    cn_max = torch.tensor(8.0, dtype=torch.double)
    assert jacrev_matches_finite_diff(f, cn_max)


def test_cn_eeq_vmap_over_cn_max() -> None:
    """`vmap` over a batch of `cn_max` values, obtained via `.replace()`
    inside the vmapped function, matches a plain Python loop -- for the
    dense (all-pairs) path."""
    sample = get_structure("mb16_43", "01")
    numbers = sample.numbers
    positions = sample.positions.double()
    structure = Structure(numbers=numbers, positions=positions)

    cn_max_batch = torch.tensor([6.0, 8.0, 10.0], dtype=torch.double)

    def f(cn_max: torch.Tensor) -> torch.Tensor:
        return cn_eeq.replace(cn_max=cn_max)(structure)

    batched = vmap(f)(cn_max_batch)
    looped = torch.stack([f(c) for c in cn_max_batch])
    assert torch.allclose(batched, looped)


# ---------------------------------------------------------------------------
# `cn_d4`: jacrev with respect to the EN-weight parameters


def test_cn_d4_jacrev_wrt_k5_matches_finite_differences() -> None:
    sample = get_structure("mb16_43", "01")
    numbers = sample.numbers
    positions = sample.positions.double()
    structure = Structure(numbers=numbers, positions=positions)

    def f(k5: torch.Tensor) -> torch.Tensor:
        from functools import partial

        weight = partial(d4_en_weight, k5=k5)
        return cn_d4.replace(pair_weight=weight)(structure).sum()

    k5 = torch.tensor(defaults.D4_K5, dtype=torch.double)
    assert jacrev_matches_finite_diff(f, k5)


# ---------------------------------------------------------------------------
# `cut_coordination_number`: exact formula and fullgraph compile


def test_cut_coordination_number_matches_log1p_formula() -> None:
    cn = torch.tensor([1.0, 5.0, 12.0], dtype=torch.double)
    cn_max = 8.0

    got = cut_coordination_number(cn, cn_max)
    want = torch.log1p(
        torch.exp(torch.tensor(cn_max, dtype=torch.double))
    ) - torch.log1p(torch.exp(torch.tensor(cn_max, dtype=torch.double) - cn))
    assert torch.allclose(got, want, atol=1e-12, rtol=0)


@pytest.mark.skipif(not DYNAMO_SUPPORTED, reason=DYNAMO_UNSUPPORTED_REASON)
def test_cut_coordination_number_compiles_fullgraph_with_tensor_cn_max() -> (
    None
):
    cn = torch.tensor([1.0, 5.0, 12.0], dtype=torch.double)
    cn_max = torch.tensor(8.0, dtype=torch.double)

    torch._dynamo.reset()  # pylint: disable=protected-access
    compiled_value = compile_fullgraph(cut_coordination_number)(cn, cn_max)
    assert torch.allclose(compiled_value, cut_coordination_number(cn, cn_max))
