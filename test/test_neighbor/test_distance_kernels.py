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
Test the distance-kernel registry (`tad_mctc.neighbor._distance_kernels`):
`select_kernel`'s own priority/override logic in isolation, with fake
kernels and no GPU, plus that `build_neighborlist`'s `distance_kernel`
override reaches it correctly.
"""

from __future__ import annotations

import pytest
import torch

from tad_mctc.neighbor._distance_kernels import (
    DistanceKernel,
    pair_distance_squared,
    select_kernel,
    split_lattice,
)
from tad_mctc.neighbor.list import build_neighborlist
from tad_mctc.typing import Tensor

from ..utils import hydrogens

_CPU = torch.device("cpu")
_FLOAT64 = torch.float64


def _unused_compute(positions_a: Tensor, positions_b: Tensor) -> Tensor:
    """Placeholder `compute`: these tests exercise `applicable`/`name`
    selection only and never call `compute` itself."""
    raise NotImplementedError


_always = DistanceKernel(
    "always", lambda device, dtype, tile_width: True, _unused_compute
)
_never = DistanceKernel(
    "never", lambda device, dtype, tile_width: False, _unused_compute
)


def test_select_kernel_picks_first_applicable_in_priority_order() -> None:
    """A kernel earlier in the list wins over a later, also-applicable
    one -- list order is the priority order, not just a filter."""
    first = DistanceKernel(
        "first", lambda device, dtype, tile_width: True, _unused_compute
    )
    second = DistanceKernel(
        "second", lambda device, dtype, tile_width: True, _unused_compute
    )

    chosen = select_kernel(_CPU, _FLOAT64, 32, kernels=[first, second])

    assert chosen.name == "first"


def test_select_kernel_skips_inapplicable_kernels() -> None:
    """A kernel earlier in the list but not applicable is skipped, not
    just deprioritised."""
    chosen = select_kernel(_CPU, _FLOAT64, 32, kernels=[_never, _always])

    assert chosen.name == "always"


def test_select_kernel_raises_if_none_applicable() -> None:
    """No silent fallback: if every candidate reports itself
    inapplicable, `select_kernel` must say so, not return something
    anyway."""
    with pytest.raises(RuntimeError, match="no distance kernel is applicable"):
        select_kernel(_CPU, _FLOAT64, 32, kernels=[_never])


def test_select_kernel_force_overrides_priority_order() -> None:
    """`force` picks the named kernel even when an earlier, also-
    applicable one would otherwise win."""
    first = DistanceKernel(
        "first", lambda device, dtype, tile_width: True, _unused_compute
    )
    second = DistanceKernel(
        "second", lambda device, dtype, tile_width: True, _unused_compute
    )

    chosen = select_kernel(
        _CPU, _FLOAT64, 32, force="second", kernels=[first, second]
    )

    assert chosen.name == "second"


def test_select_kernel_force_unknown_name_raises() -> None:
    """Forcing a name that isn't in the candidate list must raise, not
    fall through to automatic selection."""
    with pytest.raises(ValueError, match="not a known kernel"):
        select_kernel(
            _CPU, _FLOAT64, 32, force="nonexistent", kernels=[_always]
        )


def test_select_kernel_force_inapplicable_kernel_raises() -> None:
    """Forcing a kernel that exists but reports itself inapplicable for
    this device/dtype/tile_width must raise, not silently substitute a
    different kernel."""
    with pytest.raises(ValueError, match="is not applicable"):
        select_kernel(
            _CPU, _FLOAT64, 32, force="never", kernels=[_never, _always]
        )


def test_default_registry_picks_baddbmm_on_cpu() -> None:
    """The real, production registry: `baddbmm` is CPU's automatic
    choice, regardless of dtype or tile width (see the module docstring
    for why no tile-width floor was added for the one measured
    exception)."""
    for dtype in (torch.float32, torch.float64):
        for tile_width in (2, 8, 32):
            assert select_kernel(_CPU, dtype, tile_width).name == "baddbmm"


def test_default_registry_force_triton_on_cpu_raises() -> None:
    """Forcing `"triton"` on a CPU device must raise -- it is never
    applicable off CUDA, independent of whether the optional `triton`
    dependency happens to be installed."""
    with pytest.raises(ValueError, match="is not applicable"):
        select_kernel(_CPU, _FLOAT64, 32, force="triton")


def test_build_neighborlist_distance_kernel_forces_broadcast() -> None:
    """`build_neighborlist(..., distance_kernel="broadcast")` must find
    the same pairs as the automatic (here: `baddbmm`-selecting) choice
    on CPU -- forcing a different, always-correct kernel changes speed,
    never the result."""
    torch.manual_seed(0)
    # Explicit `device="cpu"`, not the ambient default: under `--cuda`,
    # `conftest.py` sets a CUDA default device, which would silently
    # change which kernel "automatic" resolves to.
    positions = torch.rand(80, 3, dtype=torch.float64, device="cpu") * 15.0
    cutoff = 6.0

    automatic = build_neighborlist(hydrogens(positions), cutoff=cutoff, tile=8)
    forced = build_neighborlist(
        hydrogens(positions), cutoff=cutoff, tile=8, distance_kernel="broadcast"
    )

    def pair_set(nbl: object) -> set[tuple[int, int]]:
        idx_i = nbl.idx_i[nbl.mask]  # type: ignore[attr-defined]
        idx_j = nbl.idx_j[nbl.mask]  # type: ignore[attr-defined]
        return set(zip(idx_i.tolist(), idx_j.tolist()))

    assert pair_set(automatic) == pair_set(forced)


def test_build_neighborlist_distance_kernel_unknown_name_raises() -> None:
    """An unknown `distance_kernel` name must raise through the public
    `build_neighborlist` entry point too, not just `select_kernel`
    directly."""
    positions = torch.rand(10, 3, dtype=torch.float64, device="cpu")
    with pytest.raises(ValueError, match="not a known kernel"):
        build_neighborlist(
            hydrogens(positions), cutoff=3.0, distance_kernel="nonexistent"
        )


def test_build_neighborlist_distance_kernel_triton_off_cuda_raises() -> None:
    """Forcing `"triton"` through the public API on a non-CUDA device
    must raise clearly, not silently run a different kernel -- this is
    exactly what makes an A/B timing comparison trustworthy. Explicit
    `device="cpu"`, not the ambient default, since `--cuda` would
    otherwise make these positions CUDA tensors and defeat the test."""
    positions = torch.rand(10, 3, dtype=torch.float64, device="cpu")
    with pytest.raises(ValueError, match="is not applicable"):
        build_neighborlist(
            hydrogens(positions), cutoff=3.0, distance_kernel="triton"
        )


def test_split_lattice() -> None:
    cell = torch.eye(3)

    assert split_lattice(None) == (None, None)

    for single in (cell, cell.unsqueeze(0)):
        shared, per_system = split_lattice(single)
        assert per_system is None
        assert shared is not None and shared.shape == (3, 3)

    shared, per_system = split_lattice(cell.expand(4, 3, 3))
    assert shared is None
    assert per_system is not None and per_system.shape == (4, 3, 3)


def test_pair_distance_squared_is_public() -> None:
    import tad_mctc.neighbor as neighbor

    assert neighbor.pair_distance_squared is pair_distance_squared
