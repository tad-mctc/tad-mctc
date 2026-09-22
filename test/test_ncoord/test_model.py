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
Tests specific to :class:`tad_mctc.ncoord.common.CNModel` itself: call-time
validation, the CN cap's numerics, the padding-atom element-1 substitution,
and derivatives with respect to the lattice for the dense periodic path.

Per-preset correctness against the mctc-lib Fortran references lives in
`test_reference.py`.
"""

from __future__ import annotations

import pytest
import torch

from tad_mctc.autograd import (
    jacrev,
    jacrev_matches_finite_diff,
    vmap,
    vmap_matches_loop,
)
from tad_mctc.data import radii
from tad_mctc.data.structures import get_structure
from tad_mctc.io.structure import Structure
from tad_mctc.ncoord import defaults
from tad_mctc.ncoord.common import (
    CNModel,
    _resolve_table,
    cut_coordination_number,
)
from tad_mctc.ncoord.count import erf_count
from tad_mctc.ncoord.d3 import cn_d3
from tad_mctc.ncoord.d4 import cn_d4, d4_en_weight
from tad_mctc.ncoord.eeq import cn_eeq, cn_eeq_en
from tad_mctc.ncoord.eeqbc import cn_eeqbc, cn_eeqbc_en
from tad_mctc.ncoord.gfn2 import cn_gfn2
from tad_mctc.neighbor.images import build_periodic_shifts

from ..conftest import DEVICE
from ..utils import (
    DYNAMO_SUPPORTED,
    DYNAMO_UNSUPPORTED_REASON,
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


def _condensed_positions(nat: int, dtype: torch.dtype) -> torch.Tensor:
    generator = torch.Generator(device=DEVICE).manual_seed(0)
    density = 0.01
    box_edge = (nat / density) ** (1.0 / 3.0)
    return (
        torch.rand(nat, 3, generator=generator).to(dtype=dtype, device=DEVICE)
        * box_edge
    )


# ---------------------------------------------------------------------------
# Validation


def test_with_precomputed_shifts_without_lattice_raises() -> None:
    """A `Structure` with no `lattice` has nothing periodic to evaluate,
    even when the caller already has a precomputed shift table -- the
    translation math still needs `structure.lattice`."""
    model = CNModel(count=erf_count, cutoff=5.0)
    numbers = torch.tensor([1, 1])
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    structure = Structure(numbers=numbers, positions=positions)

    dummy_lattice = 10.0 * torch.eye(3, dtype=torch.double)
    shifts = build_periodic_shifts(
        dummy_lattice, torch.ones(3, dtype=torch.bool), cutoff=model.cutoff
    )

    with pytest.raises(ValueError):
        model.with_precomputed_shifts(structure, shifts=shifts)


def test_call_batched_numbers_matches_single_system_loop() -> None:
    """A leading batch dimension on `structure.numbers`/`structure.
    positions` routes to the batched dense-periodic path (issue 04) and
    matches calling `__call__` once per system in a Python loop."""
    model = CNModel(count=erf_count, cutoff=5.0)
    numbers = torch.tensor([[1, 1], [1, 1]])
    positions = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.1]],
        ],
        dtype=torch.double,
    )
    lattice = torch.stack([10.0 * torch.eye(3, dtype=torch.double)] * 2)
    structure = Structure(numbers=numbers, positions=positions, lattice=lattice)

    batched = model(structure)
    looped = torch.stack(
        [
            model(Structure(numbers=n, positions=p, lattice=lat))
            for n, p, lat in zip(numbers, positions, lattice)
        ]
    )

    assert torch.allclose(batched, looped, atol=1e-11, rtol=0)


@pytest.mark.parametrize(
    "model",
    [
        CNModel(count=erf_count, cutoff=9.0),
        # exercises the batched `pair_weight` branch
        cn_d4.replace(cutoff=9.0),
        # exercises batched `cut_coordination_number`
        cn_eeq.replace(cutoff=9.0),
    ],
    ids=["plain", "pair_weight", "cn_max"],
)
def test_call_batched_heterogeneous_atoms_and_lattices(
    model: CNModel,
) -> None:
    """A batch mixing **different atom counts** (padded) *and* **different
    lattice sizes** (needing different ring counts) still gives correct
    per-system results, cross-checked against the single-system `__call__`
    for each system individually. The smaller/denser
    cell needs strictly more image rings than the larger/sparser one at
    this cutoff, so the shared table is sized to the more demanding
    system; the less demanding system's extra shift entries are masked
    out by the ordinary cutoff check rather than causing any error.
    Parametrized over a `pair_weight` preset (`cn_d4`) and a `cn_max`
    preset (`cn_eeq`) too, not just a bare `CNModel`: both branches are
    otherwise only exercised by the single-system path."""
    from tad_mctc.batch import pack

    numbers_small = torch.tensor([14, 14])
    positions_small = torch.tensor(
        [[0.0, 0.0, 0.0], [1.5, 1.5, 1.5]], dtype=torch.double
    )
    lattice_small = 6.0 * torch.eye(3, dtype=torch.double)  # dense: more rings

    numbers_large = torch.tensor([14, 14, 14])
    positions_large = torch.tensor(
        [[0.0, 0.0, 0.0], [3.0, 3.0, 3.0], [6.0, 0.0, 0.0]],
        dtype=torch.double,
    )
    lattice_large = 20.0 * torch.eye(3, dtype=torch.double)  # sparse: fewer

    numbers = pack([numbers_small, numbers_large])
    positions = pack([positions_small, positions_large])
    lattice = torch.stack([lattice_small, lattice_large])

    structure = Structure(numbers=numbers, positions=positions, lattice=lattice)
    batched = model(structure)

    single_small = model(
        Structure(
            numbers=numbers_small,
            positions=positions_small,
            lattice=lattice_small,
        )
    )
    single_large = model(
        Structure(
            numbers=numbers_large,
            positions=positions_large,
            lattice=lattice_large,
        )
    )

    assert torch.allclose(
        batched[0, : numbers_small.shape[0]], single_small, atol=1e-11, rtol=0
    )
    assert torch.allclose(
        batched[1, : numbers_large.shape[0]], single_large, atol=1e-11, rtol=0
    )
    # Padding atoms contribute nothing.
    assert torch.allclose(
        batched[0, numbers_small.shape[0] :],
        torch.zeros_like(batched[0, numbers_small.shape[0] :]),
    )


def test_call_batched_jacrev_wrt_positions_matches_finite_differences() -> None:
    """`jacrev` with respect to `positions` on the batched dense-periodic
    path matches a finite-difference Jacobian for a batch that actually
    has a padding atom, so the atom-count/shift-count padding and masking
    do not silently break autodiff at batch scale (the failure mode
    `_cn_dense_per`'s own "mask before the square root" comment guards
    against for the single-system path). The padded
    slot in the smaller system lands at `[0, 0, 0]` -- `pack`'s zero
    padding -- which coincides exactly with that system's own atom 0, the
    same zero-distance situation the single-system comment warns about."""
    from tad_mctc.batch import pack

    model = CNModel(count=erf_count, cutoff=9.0)

    numbers_small = torch.tensor([14, 14])
    positions_small = torch.tensor(
        [[0.0, 0.0, 0.0], [1.5, 1.5, 1.5]], dtype=torch.double
    )
    lattice_small = 6.0 * torch.eye(3, dtype=torch.double)

    numbers_large = torch.tensor([14, 14, 14])
    positions_large = torch.tensor(
        [[0.0, 0.0, 0.0], [3.0, 3.0, 3.0], [6.0, 0.0, 0.0]],
        dtype=torch.double,
    )
    lattice_large = 20.0 * torch.eye(3, dtype=torch.double)

    numbers = pack([numbers_small, numbers_large])
    positions = pack([positions_small, positions_large])
    lattice = torch.stack([lattice_small, lattice_large])

    def f(p: torch.Tensor) -> torch.Tensor:
        structure = Structure(numbers=numbers, positions=p, lattice=lattice)
        return model(structure)

    assert jacrev_matches_finite_diff(f, positions)


def test_non_scalar_cn_max_raises() -> None:
    model = CNModel(
        count=erf_count, cutoff=5.0, cn_max=torch.tensor([1.0, 2.0])
    )
    numbers = torch.tensor([1, 1])
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    structure = Structure(numbers=numbers, positions=positions)

    with pytest.raises(ValueError):
        model(structure)


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
# Padding atoms: element 1, not 0


def test_padding_rcov_jacobian_is_finite_with_element_one() -> None:
    """A batched, padded dense evaluation differentiated with respect to
    the `rcov` table must give a finite Jacobian: padding atoms look up
    element 1, whose radius is non-zero, so `r0**norm_exp` never sees a
    zero base (see `CNModel`'s spec, padding-atom substitution)."""
    from tad_mctc.batch import pack
    from tad_mctc.data import radii

    sih4 = get_structure("mb16_43", "SiH4")
    mb = get_structure("mb16_43", "01")
    numbers = pack([sih4.numbers, mb.numbers])
    positions = pack([sih4.positions.double(), mb.positions.double()])
    structure = Structure(numbers=numbers, positions=positions)

    def cn_sum(table: torch.Tensor) -> torch.Tensor:
        model = CNModel(count=erf_count, cutoff=25.0, rcov=table)
        return model(structure).sum()

    table = radii.COV_D3(dtype=torch.double)
    jacobian = jacrev(cn_sum)(table)
    assert torch.isfinite(jacobian).all()


# ---------------------------------------------------------------------------
# Lattice as a call-time argument: jacrev and vmap


def test_dense_periodic_periodic_mask_leaves_slab_axis_unwrapped() -> None:
    """The internal wrap must fold only the axes the structure's own
    `periodic` marks, exactly like `wrap_to_central_cell` itself (see also
    `test_precomputed_shifts.py`'s `test_periodic_cn_dense_matches_for_
    unwrapped_positions`). For a slab (periodic in x/y, vacuum along z), an
    atom moved by one whole lattice vector along z is a physically
    different, more isolated system -- wrapping z anyway, as for a
    structure whose mask marks all three axes periodic, would silently
    fold it back and hide that."""
    lattice = torch.diag(torch.tensor([6.0, 6.0, 40.0], dtype=torch.double))
    periodic_slab = torch.tensor([True, True, False])
    periodic_all = torch.tensor([True, True, True])

    numbers = torch.tensor([14, 14])
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.5, 1.5, 2.0]], dtype=torch.double
    )
    displaced = positions.clone()
    displaced[1, 2] += lattice[2, 2]

    def slab(pos: torch.Tensor, periodic: torch.Tensor) -> Structure:
        return Structure(
            numbers=numbers, positions=pos, lattice=lattice, periodic=periodic
        )

    # The structure's mask, not the table's, decides which axes are
    # periodic, so one table built for all three axes serves both masks.
    shifts = build_periodic_shifts(lattice, periodic_all, cutoff=cn_d3.cutoff)

    baseline = cn_d3.with_precomputed_shifts(
        slab(positions, periodic_slab), shifts=shifts
    )
    correct_slab = cn_d3.with_precomputed_shifts(
        slab(displaced, periodic_slab), shifts=shifts
    )
    wrong_all_periodic = cn_d3.with_precomputed_shifts(
        slab(displaced, periodic_all), shifts=shifts
    )

    # Moving atom 1 a full 40 Bohr along the vacuum axis is a real
    # physical change once z is correctly left unwrapped.
    assert not torch.allclose(baseline, correct_slab)
    # Treating z as periodic anyway silently folds the displacement away.
    assert torch.allclose(baseline, wrong_all_periodic, atol=1e-11, rtol=0)


@pytest.mark.skipif(not DYNAMO_SUPPORTED, reason=DYNAMO_UNSUPPORTED_REASON)
def test_dense_periodic_compiles_fullgraph() -> None:
    """The dense periodic path traces under `torch.compile(fullgraph=True)`
    in both `positions` and `lattice`: no data-dependent shape or control
    flow reaches consumption, since `shifts` is built ahead of the call."""
    sample = get_structure("other", "periodic_cubic")
    assert sample.lattice is not None and sample.periodic is not None
    numbers = sample.numbers
    positions = sample.positions.double()
    lattice = sample.lattice.double()
    periodic = sample.periodic

    shifts = build_periodic_shifts(lattice, periodic, cutoff=cn_d3.cutoff)

    def f(p: torch.Tensor, lat: torch.Tensor) -> torch.Tensor:
        structure = Structure(numbers=numbers, positions=p, lattice=lat)
        return cn_d3.with_precomputed_shifts(structure, shifts=shifts)

    torch._dynamo.reset()  # pylint: disable=protected-access
    compiled_value = run_compiled_or_skip(f, positions, lattice)

    assert torch.allclose(compiled_value, f(positions, lattice))


def test_dense_periodic_jacrev_wrt_lattice_matches_finite_differences() -> None:
    """`jacrev` with respect to `lattice` matches a finite-difference
    Jacobian for the dense periodic path (`with_precomputed_shifts`)."""
    sample = get_structure("other", "periodic_cubic")
    assert sample.lattice is not None and sample.periodic is not None
    numbers = sample.numbers
    positions = sample.positions.double()
    lattice = sample.lattice.double()
    periodic = sample.periodic

    model = CNModel(count=erf_count, cutoff=25.0)
    shifts = build_periodic_shifts(lattice, periodic, cutoff=model.cutoff)

    def f(lat: torch.Tensor) -> torch.Tensor:
        structure = Structure(numbers=numbers, positions=positions, lattice=lat)
        return model.with_precomputed_shifts(structure, shifts=shifts)

    assert jacrev_matches_finite_diff(f, lattice)


def test_vmap_over_lattices_dense_periodic_with_one_shared_shifts() -> None:
    """`vmap` over a batch of single-system `Structure`s with a varying
    `lattice` and one shared `shifts` table, for the dense periodic
    path, matches a plain Python loop -- `vmap` over several
    single-system `Structure`s, not issue 04's leading-batch-dimension
    path (kept distinct, per this feature's spec). The table is built at
    the smallest lattice in the batch, which needs the most image rings
    for a given cutoff, so it safely covers the larger, scaled-up
    lattices too."""
    sample = get_structure("other", "periodic_cubic")
    assert sample.lattice is not None and sample.periodic is not None
    numbers = sample.numbers
    positions = sample.positions.double()
    lattice = sample.lattice.double()
    periodic = sample.periodic

    model = CNModel(count=erf_count, cutoff=25.0)

    batch = torch.stack([lattice * scale for scale in (1.0, 1.01, 1.02)])
    shifts = build_periodic_shifts(
        batch[0], periodic, cutoff=model.cutoff
    )  # scale >= 1.0 only shrinks the required ring count

    def f(lat: torch.Tensor) -> torch.Tensor:
        structure = Structure(numbers=numbers, positions=positions, lattice=lat)
        return model.with_precomputed_shifts(structure, shifts=shifts)

    assert vmap_matches_loop(f, batch)


# ---------------------------------------------------------------------------
# Issue 05: `_resolve_table` caches a `TableFunction`'s resolved tensor at
# function level, keyed by table identity + device/dtype, instead of
# rebuilding it on every call.


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

    first = _resolve_table(spy_table, like)
    second = _resolve_table(spy_table, like)

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

    first = _resolve_table(spy_table, like_f32)
    second = _resolve_table(spy_table, like_f64)

    assert len(calls) == 2
    assert first.dtype == torch.float32
    assert second.dtype == torch.float64


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
    from tad_mctc.data import radii

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


def test_table_cache_survives_cold_fill_under_vmap() -> None:
    """The first resolution of a `TableFunction` (a cold cache) happening
    *while* `tad_mctc.autograd.vmap` (``torch.func.vmap`` on PyTorch>=2.0)
    is tracing must cache a plain `Tensor`, not a functorch batched-tensor
    wrapper -- `radii.COV_D3()` builds a table with no data dependency on
    the vmapped argument, so it is never wrapped, but this pins that
    behaviour down as a regression test."""
    from tad_mctc.ncoord.common import _TABLE_CACHE

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
    for cached in _TABLE_CACHE.values():
        assert type(cached) is torch.Tensor

    # The cached tensor must remain usable in a plain eager call afterward.
    eager = cn_d3(structure)
    assert torch.allclose(eager, f(torch.tensor(1.0, dtype=torch.double)))


@pytest.mark.skipif(not DYNAMO_SUPPORTED, reason=DYNAMO_UNSUPPORTED_REASON)
def test_table_cache_survives_cold_fill_under_compile() -> None:
    """Same guarantee as the vmap test above, for a cold cache filled
    while `torch.compile(fullgraph=True)` is tracing."""
    from tad_mctc.ncoord.common import _TABLE_CACHE

    sample = get_structure("mb16_43", "01")
    numbers = sample.numbers
    positions = sample.positions.double()

    _TABLE_CACHE.clear()
    torch._dynamo.reset()  # pylint: disable=protected-access

    def f(p: torch.Tensor) -> torch.Tensor:
        return cn_d3(Structure(numbers=numbers, positions=p))

    compiled_value = run_compiled_or_skip(f, positions)

    assert len(_TABLE_CACHE) > 0
    for cached in _TABLE_CACHE.values():
        assert type(cached) is torch.Tensor

    eager_value = f(positions)
    assert torch.allclose(compiled_value, eager_value)


# ---------------------------------------------------------------------------
# Tables: a tensor and its matching table function give the same result


def test_tensor_table_matches_table_function() -> None:
    from tad_mctc.data import radii

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
    compiled_value = run_compiled_or_skip(cut_coordination_number, cn, cn_max)
    assert torch.allclose(compiled_value, cut_coordination_number(cn, cn_max))
