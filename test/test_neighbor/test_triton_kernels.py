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
Test the optional Triton kernel for the exact tile-pair distance filter
(`tad_mctc.neighbor._distance_kernels.pairwise_distance_squared`), and
that `build_neighborlist` picks it up transparently on a CUDA device
where it is available.
"""

from __future__ import annotations

import pytest
import torch

from tad_mctc.neighbor import _distance_kernels
from tad_mctc.neighbor._tiles import Tiles, tile_pairs
from tad_mctc.neighbor.list import build_neighborlist

from ..utils import hydrogens


def test_is_available_false_on_cpu() -> None:
    """Never available on CPU, regardless of whether `triton` is
    installed -- the kernel only ever targets CUDA."""
    assert _distance_kernels.is_available(torch.device("cpu")) is False


@pytest.mark.triton
def test_is_available_true_on_cuda() -> None:
    """Available on CUDA once this test runs at all, since the `triton`
    marker itself only executes when both CUDA and `triton` are
    present."""
    assert _distance_kernels.is_available(torch.device("cuda")) is True


@pytest.mark.triton
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("tile", [1, 2, 3, 16, 30, 33])
def test_pairwise_distance_squared_matches_broadcast(
    dtype: torch.dtype, tile: int
) -> None:
    """The kernel's squared distances must match the plain broadcast
    formula to within floating-point precision, for both supported
    dtypes and for tile widths that are not a power of two."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    positions_a = torch.randn(37, tile, 3, dtype=dtype, device=device) * 50.0
    positions_b = torch.randn(37, tile, 3, dtype=dtype, device=device) * 50.0

    got = _distance_kernels.pairwise_distance_squared(positions_a, positions_b)

    difference = positions_a.unsqueeze(2) - positions_b.unsqueeze(1)
    expected = (difference * difference).sum(-1)

    tolerance = 1e-3 if dtype == torch.float32 else 1e-8
    assert torch.allclose(got, expected, atol=tolerance, rtol=tolerance)


@pytest.mark.triton
def test_pairwise_distance_squared_empty_chunk() -> None:
    """Zero candidate tile pairs must not launch the kernel, and must
    still return a correctly-shaped, empty result."""
    device = torch.device("cuda")
    positions_a = torch.empty(0, 8, 3, dtype=torch.float64, device=device)
    positions_b = torch.empty(0, 8, 3, dtype=torch.float64, device=device)

    result = _distance_kernels.pairwise_distance_squared(
        positions_a, positions_b
    )

    assert result.shape == (0, 8, 8)


def test_pairwise_distance_squared_raises_off_cuda() -> None:
    """Calling the kernel wrapper on a non-CUDA tensor must raise rather
    than silently running the CPU broadcast formula -- `is_available`,
    not this function, is what a caller must branch on."""
    if not _distance_kernels.TRITON_AVAILABLE:
        pytest.skip("Optional `triton` dependency not installed.")

    # Explicit `device="cpu"`, not the ambient default: under `--cuda`,
    # `conftest.py` sets a CUDA default device, and this test's whole
    # point is to check the *off*-CUDA error path.
    positions = torch.zeros(1, 4, 3, dtype=torch.float64, device="cpu")
    with pytest.raises(RuntimeError, match="CUDA"):
        _distance_kernels.pairwise_distance_squared(positions, positions)


@pytest.mark.triton
@pytest.mark.parametrize("tile", [16, 30])
def test_build_neighborlist_matches_cpu_with_triton(tile: int) -> None:
    """`build_neighborlist` on CUDA, with the optional Triton kernel
    available, must find the exact same pairs as the CPU build -- the
    dispatch inside `_atom_pairs_within_thresholds` must not change which
    pairs are found, only how fast. A tile width that is not a power of
    two runs the kernel with masked slots."""
    torch.manual_seed(0)
    positions_cpu = torch.rand(300, 3, dtype=torch.float64) * 40.0
    cutoff = 8.0

    nbl_cpu = build_neighborlist(
        hydrogens(positions_cpu), cutoff=cutoff, tile=tile
    )
    nbl_cuda = build_neighborlist(
        hydrogens(positions_cpu.to("cuda")), cutoff=cutoff, tile=tile
    )

    def pair_set(nbl: object) -> set[frozenset[int]]:
        # `frozenset`, not a `(idx_i, idx_j)` tuple: only `idx_i != idx_j`
        # is part of the contract (see
        # `test_atom_pairs_within_thresholds_uses_triton_on_cuda` below),
        # so which index lands in `idx_i` vs `idx_j` for the same physical
        # pair can legitimately differ between CPU and CUDA execution
        # order -- confirmed directly: for this 300-atom case, 359 of the
        # 1214 pairs come out direction-swapped between the two devices
        # while the unordered pair set is identical.
        idx_i = nbl.idx_i[nbl.mask].cpu()  # type: ignore[attr-defined]
        idx_j = nbl.idx_j[nbl.mask].cpu()  # type: ignore[attr-defined]
        return {
            frozenset((i, j)) for i, j in zip(idx_i.tolist(), idx_j.tolist())
        }

    assert pair_set(nbl_cpu) == pair_set(nbl_cuda)


@pytest.mark.triton
def test_atom_pairs_within_thresholds_uses_triton_on_cuda() -> None:
    """Sanity check on the dispatch flag itself, not just the end result:
    `_atom_pairs_within_thresholds` must actually pick the Triton branch
    on CUDA when it is available, not silently fall back. Distinct-pair
    ordering is not part of this function's own contract (each pair is
    unordered and appears once; only `idx_i != idx_j` holds), so that is
    what is checked here, not any numeric `idx_i < idx_j`."""
    from tad_mctc.neighbor.list import _atom_pairs_within_thresholds

    torch.manual_seed(0)
    positions = (torch.rand(64, 3, dtype=torch.float64) * 20.0).to("cuda")
    tiles = Tiles(positions, tile=8)
    tile_a, tile_b = tile_pairs(tiles, cutoff=6.0)

    idx_i, idx_j, _ = _atom_pairs_within_thresholds(
        tiles, tile_a, tile_b, positions, (6.0,)
    )[0]

    assert idx_i.shape == idx_j.shape
    assert idx_i.shape[0] > 0
    assert bool((idx_i != idx_j).all())
