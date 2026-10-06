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
    _baddbmm_distance_squared,
    _broadcast_distance_squared,
    gather_rows,
    pair_distance_squared,
    pair_distance_squared_from_columns,
    position_columns,
    select_kernel,
    split_lattice,
)
from tad_mctc.neighbor._tiles import Tiles, tile_pairs
from tad_mctc.neighbor.list import (
    _atom_pairs_within_thresholds,
    build_neighborlist,
)
from tad_mctc.typing import Tensor

from ..utils import hydrogens, jacfwd, jacrev, run_compiled_or_skip

_CPU = torch.device("cpu")


def _unused_compute(positions_a: Tensor, positions_b: Tensor) -> Tensor:
    """Placeholder `compute`: these tests exercise `applicable`/`name`
    selection only and never call `compute` itself."""
    raise NotImplementedError


_always = DistanceKernel("always", lambda device: True, _unused_compute)
_never = DistanceKernel("never", lambda device: False, _unused_compute)


def test_select_kernel_picks_first_applicable_in_priority_order() -> None:
    """A kernel earlier in the list wins over a later, also-applicable
    one -- list order is the priority order, not just a filter."""
    first = DistanceKernel("first", lambda device: True, _unused_compute)
    second = DistanceKernel("second", lambda device: True, _unused_compute)

    chosen = select_kernel(_CPU, kernels=[first, second])

    assert chosen.name == "first"


def test_select_kernel_skips_inapplicable_kernels() -> None:
    """A kernel earlier in the list but not applicable is skipped, not
    just deprioritised."""
    chosen = select_kernel(_CPU, kernels=[_never, _always])

    assert chosen.name == "always"


def test_select_kernel_raises_if_none_applicable() -> None:
    """No silent fallback: if every candidate reports itself
    inapplicable, `select_kernel` must say so, not return something
    anyway."""
    with pytest.raises(RuntimeError, match="no distance kernel is applicable"):
        select_kernel(_CPU, kernels=[_never])


def test_select_kernel_force_overrides_priority_order() -> None:
    """`force` picks the named kernel even when an earlier, also-
    applicable one would otherwise win."""
    first = DistanceKernel("first", lambda device: True, _unused_compute)
    second = DistanceKernel("second", lambda device: True, _unused_compute)

    chosen = select_kernel(_CPU, force="second", kernels=[first, second])

    assert chosen.name == "second"


def test_select_kernel_force_unknown_name_raises() -> None:
    """Forcing a name that isn't in the candidate list must raise, not
    fall through to automatic selection."""
    with pytest.raises(ValueError, match="not a known kernel"):
        select_kernel(_CPU, force="nonexistent", kernels=[_always])


def test_select_kernel_force_inapplicable_kernel_raises() -> None:
    """Forcing a kernel that exists but reports itself inapplicable for
    this device must raise, not silently substitute a
    different kernel."""
    with pytest.raises(ValueError, match="is not applicable"):
        select_kernel(_CPU, force="never", kernels=[_never, _always])


def test_default_registry_picks_baddbmm_on_cpu() -> None:
    """The real, production registry: `baddbmm` is CPU's automatic
    choice."""
    assert select_kernel(_CPU).name == "baddbmm"


def test_default_registry_force_triton_on_cpu_raises() -> None:
    """Forcing `"triton"` on a CPU device must raise -- it is never
    applicable off CUDA, independent of whether the optional `triton`
    dependency happens to be installed."""
    with pytest.raises(ValueError, match="is not applicable"):
        select_kernel(_CPU, force="triton")


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
            hydrogens(positions), cutoff=3.0, distance_kernel="nonexistent"  # type: ignore[arg-type]
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


def test_baddbmm_matches_broadcast_at_large_coordinates() -> None:
    """`_baddbmm_distance_squared` computes ``|a|^2 + |b|^2 - 2 a.b``,
    which cancels catastrophically once ``|a|``/``|b|`` are large
    relative to the cutoff -- two ~1500 A positions differing by ~13 A
    in float32 leaves only a few significant digits for that difference.
    That is why the kernel measures each tile pair from its own first atom
    before computing any distance: it bounds ``|a|``/``|b|`` by the tile
    diameter plus `cutoff` regardless of where in the structure the pair
    sits, so `"baddbmm"` should agree with `"broadcast"` (direct
    subtraction, unaffected either way) exactly, even at this coordinate
    magnitude.

    This guards that per-tile-pair shift. A single global mean-subtraction
    is not enough: it leaves a large residual gap here (measured on a real
    1.7M-atom chain: 202.3M of 220.5M pairs against a true 187.8M),
    because a global mean does not bound any individual tile pair's
    coordinates when the structure is elongated rather than centred at its
    own mean.

    `"baddbmm"` is the CPU path whenever the native extension is missing,
    so this test must not depend on that extension."""
    torch.manual_seed(4)
    # Offset far from the origin, like one axis of a real, extended
    # structure (a ~213k-atom structure spans ~1500 A) -- small random
    # positions near the origin do not trigger
    # the cancellation and would make this test pass for the wrong
    # reason.
    offset = torch.tensor([1500.0, 0.0, 0.0], device="cpu")
    positions = offset + torch.randn(500, 3, device="cpu") * 20.0
    cutoff = 25.0

    tiles = Tiles(positions, tile=32)
    tile_a, tile_b = tile_pairs(tiles, cutoff)

    ((broadcast_i, broadcast_j, _),) = _atom_pairs_within_thresholds(
        tiles, tile_a, tile_b, positions, (cutoff,), distance_kernel="broadcast"
    )
    ((baddbmm_i, baddbmm_j, _),) = _atom_pairs_within_thresholds(
        tiles, tile_a, tile_b, positions, (cutoff,), distance_kernel="baddbmm"
    )

    assert broadcast_i.shape[0] > 0, "test needs a system with real pairs"
    assert torch.equal(broadcast_i, baddbmm_i)
    assert torch.equal(broadcast_j, baddbmm_j)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("offset", [0.0, 1e4, 1e5])
def test_baddbmm_error_does_not_grow_with_coordinates(
    dtype: torch.dtype, offset: float
) -> None:
    """The squared distances of `"baddbmm"` stay within a few `eps *
    cutoff^2` of the exact ones, wherever the atoms sit. Measured on
    random cells at offsets up to 1e5 Bohr, the error is at most 6 `eps *
    cutoff^2` in either dtype; without the per-tile-pair shift it would
    grow with `offset^2`."""
    torch.manual_seed(5)
    cutoff = 25.0
    eps = torch.finfo(dtype).eps
    exact_positions = torch.rand(600, 3, dtype=torch.float64) * 35.0 + offset
    positions = exact_positions.to(dtype)

    tiles = Tiles(positions, tile=32)
    tile_a, tile_b = tile_pairs(tiles, cutoff)
    positions_a = positions[tiles.index[tile_a]]
    positions_b = positions[tiles.index[tile_b]]

    # The exact squared distances of the same, rounded coordinates.
    exact = _broadcast_distance_squared(
        positions_a.double(), positions_b.double()
    )
    computed = _baddbmm_distance_squared(positions_a, positions_b).double()

    near = exact <= (1.2 * cutoff) ** 2
    error = (computed - exact).abs()[near]
    assert float(error.max()) <= 16 * eps * cutoff**2


def test_triton_import_failure_disables_triton(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib
    import sys

    from tad_mctc.neighbor import _distance_kernels

    # a `None` entry makes `import triton` raise ImportError
    monkeypatch.setitem(sys.modules, "triton", None)
    monkeypatch.setitem(sys.modules, "triton.language", None)
    try:
        reloaded = importlib.reload(_distance_kernels)
        assert reloaded.TRITON_AVAILABLE is False
    finally:
        monkeypatch.undo()
        importlib.reload(_distance_kernels)


# `pair_distance_squared` sums the three components by hand instead of with
# `.sum(-1)`, which is ~10x slower for an axis of length 3. These tests pin
# the values to the plain formula and keep the function usable under the
# transforms the sparse coordination number is promised to survive.


def _pair_inputs(
    n_atoms: int = 6, n_pairs: int = 40, *, periodic: bool = False
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Positions with a trailing phantom atom, random pair indices, integer
    shifts (all zero for a molecule), and a cell."""
    generator = torch.Generator().manual_seed(7)
    dd = {"dtype": torch.double}
    positions = torch.randn(n_atoms + 1, 3, generator=generator, **dd)
    idx_i = torch.randint(0, n_atoms, (n_pairs,), generator=generator)
    idx_j = torch.randint(0, n_atoms, (n_pairs,), generator=generator)
    if periodic:
        shift = torch.randint(-2, 3, (n_pairs, 3), generator=generator).short()
    else:
        shift = torch.zeros(n_pairs, 3, dtype=torch.int16)
    cell = torch.tensor(
        [[5.0, 0.3, 0.0], [0.2, 6.0, 0.4], [0.0, 0.1, 7.0]], **dd
    )
    return positions, idx_i, idx_j, shift, cell


def _reference(
    positions: Tensor,
    idx_i: Tensor,
    idx_j: Tensor,
    shift: Tensor,
    cell: Tensor | None,
) -> Tensor:
    """The textbook formula, with the reduction over the last axis."""
    difference = positions[idx_j] - positions[idx_i]
    if cell is not None:
        difference = difference + shift.to(cell.dtype) @ cell
    return (difference**2).sum(-1)


def test_pair_distance_squared_of_a_molecule() -> None:
    positions, idx_i, idx_j, shift, _ = _pair_inputs()
    result = pair_distance_squared(
        idx_i,
        idx_j,
        shift,
        positions,
        shared_lattice=None,
        system_lattices=None,
        atoms_per_system=positions.shape[0],
    )
    expected = _reference(positions, idx_i, idx_j, shift, None)

    assert result.shape == idx_i.shape
    assert torch.allclose(result, expected, rtol=1e-14, atol=0)


def test_pair_distance_squared_with_a_shared_cell() -> None:
    positions, idx_i, idx_j, shift, cell = _pair_inputs(periodic=True)
    result = pair_distance_squared(
        idx_i,
        idx_j,
        shift,
        positions,
        shared_lattice=cell,
        system_lattices=None,
        atoms_per_system=positions.shape[0],
    )
    expected = _reference(positions, idx_i, idx_j, shift, cell)

    assert torch.allclose(result, expected, rtol=1e-14, atol=0)


def test_pair_distance_squared_with_a_cell_per_system() -> None:
    """Two systems of three atoms, each pair taking the cell of its own
    system."""
    positions, _, _, shift, cell = _pair_inputs(periodic=True)
    cells = torch.stack([cell, 1.5 * cell])
    idx_i = torch.tensor([0, 1, 2, 3, 4, 5, 0, 3])
    idx_j = torch.tensor([1, 2, 0, 5, 3, 4, 2, 4])
    shift = shift[: idx_i.shape[0]]
    result = pair_distance_squared(
        idx_i,
        idx_j,
        shift,
        positions,
        shared_lattice=None,
        system_lattices=cells,
        atoms_per_system=3,
    )

    pair_cells = cells[idx_i // 3]
    difference = positions[idx_j] - positions[idx_i]
    difference = difference + (
        shift.to(cells.dtype).unsqueeze(-2) @ pair_cells
    ).squeeze(-2)
    assert torch.allclose(result, (difference**2).sum(-1), rtol=1e-14, atol=0)


def _distance_of(
    positions: Tensor,
    case: str,
    inputs: tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
) -> Tensor:
    """``pair_distance_squared`` of ``positions`` for the pairs in
    ``inputs``, which are made outside of any transform."""
    _, idx_i, idx_j, shift, cell = inputs
    return pair_distance_squared(
        idx_i,
        idx_j,
        shift,
        positions,
        shared_lattice=cell if case == "cell" else None,
        system_lattices=None,
        atoms_per_system=positions.shape[0],
    )


@pytest.mark.parametrize("case", ["molecule", "cell"])
def test_pair_distance_squared_under_vmap(case: str) -> None:
    """A batch of positions gives the same result as a loop over them."""
    inputs = _pair_inputs(periodic=case == "cell")
    positions = inputs[0]
    batch = torch.stack([positions, 1.1 * positions, positions + 0.3])

    result = torch.func.vmap(lambda p: _distance_of(p, case, inputs))(batch)
    expected = torch.stack([_distance_of(p, case, inputs) for p in batch])

    assert result.shape == (3, 40)
    assert torch.allclose(result, expected, rtol=1e-14, atol=0)


@pytest.mark.parametrize("case", ["molecule", "cell"])
def test_pair_distance_squared_derivatives(case: str) -> None:
    """``jacrev`` and ``jacfwd`` agree with each other and with the
    analytic derivative, ``2 * difference`` on atoms ``j`` and ``i``."""
    inputs = _pair_inputs(periodic=case == "cell")
    positions, idx_i, idx_j, shift, cell = inputs

    def func(p: Tensor) -> Tensor:
        return _distance_of(p, case, inputs)

    reverse = jacrev(func)(positions)
    forward = jacfwd(func)(positions)
    assert torch.allclose(reverse, forward, rtol=1e-12, atol=1e-12)

    difference = positions[idx_j] - positions[idx_i]
    if case == "cell":
        difference = difference + shift.to(cell.dtype) @ cell
    expected = torch.zeros_like(reverse)
    rows = torch.arange(idx_i.shape[0])
    # `index_put_` accumulates, so a pair of an atom with itself is right
    expected.index_put_((rows, idx_j), 2.0 * difference, accumulate=True)
    expected.index_put_((rows, idx_i), -2.0 * difference, accumulate=True)
    assert torch.allclose(reverse, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("case", ["molecule", "cell"])
def test_pair_distance_squared_compiles_as_one_graph(case: str) -> None:
    inputs = _pair_inputs(periodic=case == "cell")
    positions = inputs[0]

    result = run_compiled_or_skip(
        lambda p: _distance_of(p, case, inputs), positions
    )

    assert torch.allclose(
        result, _distance_of(positions, case, inputs), rtol=1e-14, atol=0
    )


def _three_cases() -> dict[str, tuple[Tensor, Tensor, Tensor, Tensor, dict]]:
    """Positions, ``int64`` indices, shifts and the lattice arguments of a
    molecule, a shared cell and two systems with a cell each."""
    positions, idx_i, idx_j, shift, cell = _pair_inputs(periodic=True)
    molecule = {"shared_lattice": None, "system_lattices": None}
    shared = {"shared_lattice": cell, "system_lattices": None}
    per_system = {
        "shared_lattice": None,
        "system_lattices": torch.stack([cell, 1.5 * cell]),
    }
    zero = torch.zeros_like(shift)
    return {
        "molecule": (positions, idx_i, idx_j, zero, molecule),
        "shared_cell": (positions, idx_i, idx_j, shift, shared),
        "cell_per_system": (positions, idx_i, idx_j, shift, per_system),
    }


def _row_formula(
    positions: Tensor,
    idx_i: Tensor,
    idx_j: Tensor,
    shift: Tensor,
    lattices: dict,
) -> Tensor:
    """The kernel before it gathered by column: rows of ``(n, 3)``
    positions, then the same per-component sum."""
    difference = positions.index_select(0, idx_j) - positions.index_select(
        0, idx_i
    )
    if lattices["shared_lattice"] is not None:
        difference = (
            difference + shift.to(positions.dtype) @ lattices["shared_lattice"]
        )
    elif lattices["system_lattices"] is not None:
        cells = lattices["system_lattices"].index_select(0, idx_i // 3)
        difference = difference + (
            shift.to(positions.dtype).unsqueeze(-2) @ cells
        ).squeeze(-2)
    x, y, z = difference[..., 0], difference[..., 1], difference[..., 2]
    return x * x + y * y + z * z


@pytest.mark.parametrize("case", ["molecule", "shared_cell", "cell_per_system"])
def test_column_gathers_give_the_row_formula_bit_for_bit(case: str) -> None:
    positions, idx_i, idx_j, shift, lattices = _three_cases()[case]

    result = pair_distance_squared(
        idx_i, idx_j, shift, positions, atoms_per_system=3, **lattices
    )

    assert torch.equal(
        result, _row_formula(positions, idx_i, idx_j, shift, lattices)
    )


@pytest.mark.parametrize("case", ["molecule", "shared_cell", "cell_per_system"])
def test_int32_indices_give_the_int64_bits(case: str) -> None:
    """Value and derivative, whichever dtype the list hands over."""
    positions, idx_i, idx_j, shift, lattices = _three_cases()[case]

    def distances(i: Tensor, j: Tensor) -> object:
        def f(p: Tensor) -> Tensor:
            return pair_distance_squared(
                i, j, shift, p, atoms_per_system=3, **lattices
            )

        return f(positions), jacrev(f)(positions), jacfwd(f)(positions)

    narrow = distances(idx_i.int(), idx_j.int())
    wide = distances(idx_i, idx_j)

    for a, b in zip(narrow, wide):  # type: ignore[call-overload]
        assert torch.equal(a, b)


def test_columns_split_once_give_the_same_bits() -> None:
    positions, idx_i, idx_j, shift, lattices = _three_cases()["shared_cell"]
    columns = position_columns(positions)

    assert all(c.shape == (positions.shape[0],) for c in columns)
    assert all(c.is_contiguous() for c in columns)
    chunks = [
        pair_distance_squared_from_columns(
            idx_i[start : start + 16].int(),
            idx_j[start : start + 16].int(),
            shift[start : start + 16],
            columns,
            atoms_per_system=3,
            **lattices,
        )
        for start in range(0, idx_i.shape[0], 16)
    ]

    assert torch.equal(
        torch.cat(chunks),
        pair_distance_squared(
            idx_i, idx_j, shift, positions, atoms_per_system=3, **lattices
        ),
    )


@pytest.mark.parametrize("case", ["molecule", "shared_cell", "cell_per_system"])
def test_int32_distances_under_vmap(case: str) -> None:
    positions, idx_i, idx_j, shift, lattices = _three_cases()[case]
    batch = torch.stack([positions, 1.1 * positions, positions + 0.3])

    def f(p: Tensor) -> Tensor:
        return pair_distance_squared(
            idx_i.int(), idx_j.int(), shift, p, atoms_per_system=3, **lattices
        )

    expected = torch.stack([f(p) for p in batch])
    assert torch.equal(torch.func.vmap(f)(batch), expected)


@pytest.mark.parametrize("case", ["molecule", "shared_cell", "cell_per_system"])
def test_int32_distances_compile(case: str) -> None:
    positions, idx_i, idx_j, shift, lattices = _three_cases()[case]

    def f(p: Tensor) -> Tensor:
        return pair_distance_squared(
            idx_i.int(), idx_j.int(), shift, p, atoms_per_system=3, **lattices
        )

    result = run_compiled_or_skip(f, positions)
    assert torch.allclose(result, f(positions), rtol=1e-14, atol=0)


def test_gather_rows_is_index_select_in_eager() -> None:
    table = torch.arange(12.0, dtype=torch.double).reshape(4, 3)
    index = torch.tensor([3, 3, 0, 2])
    assert torch.equal(gather_rows(table, index), table.index_select(0, index))
    assert torch.equal(gather_rows(table[:, 0], index), table[:, 0][index])


def test_compiled_gather_rows_gradient() -> None:
    """``torch.compile(jacrev(f))`` of an ``index_select`` is wrong on
    PyTorch 2.5 to 2.13: every row came out as the sum of all rows."""
    x = torch.arange(1.0, 5.0, dtype=torch.double)
    index = torch.tensor([3, 2, 1, 0])

    def f(p: Tensor) -> Tensor:
        return gather_rows(p, index) ** 2

    expected = torch.flip(torch.diag(2 * x.flip(0)), (1,))
    assert torch.equal(torch.func.jacrev(f)(x), expected)
    result = run_compiled_or_skip(torch.func.jacrev(f), x)
    assert torch.equal(result, expected)


@pytest.mark.parametrize("case", ["molecule", "shared_cell", "cell_per_system"])
def test_int32_distances_compiled_jacrev(case: str) -> None:
    """The compiled Jacobian of the distance kernel (column gathers and,
    per system, the cell gather) matches the eager one."""
    positions, idx_i, idx_j, shift, lattices = _three_cases()[case]

    def f(p: Tensor) -> Tensor:
        return pair_distance_squared(
            idx_i.int(), idx_j.int(), shift, p, atoms_per_system=3, **lattices
        )

    result = run_compiled_or_skip(torch.func.jacrev(f), positions)
    assert torch.allclose(
        result, torch.func.jacrev(f)(positions), rtol=1e-13, atol=1e-13
    )


@pytest.mark.parametrize("case", ["shared_cell", "cell_per_system"])
def test_lattice_gradient_matches_the_row_formula(case: str) -> None:
    """The translation is summed column by column: the values are the same
    bits as the matrix product, the lattice gradient the same to rounding."""
    positions, idx_i, idx_j, shift, lattices = _three_cases()[case]
    key = "shared_lattice" if case == "shared_cell" else "system_lattices"
    lattice = lattices[key].clone().requires_grad_(True)
    lattices = {**lattices, key: lattice}

    new = pair_distance_squared(
        idx_i.int(),
        idx_j.int(),
        shift,
        positions,
        atoms_per_system=3,
        **lattices,
    )
    old = _row_formula(positions, idx_i, idx_j, shift, lattices)
    assert torch.equal(new, old)

    (g_new,) = torch.autograd.grad(new.sum(), lattice)
    (g_old,) = torch.autograd.grad(old.sum(), lattice)
    assert torch.allclose(g_new, g_old, rtol=1e-13, atol=1e-13)
