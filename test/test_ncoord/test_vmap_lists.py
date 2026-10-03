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
`vmap` over *different* molecules, each with its own neighbour list.

`test_transforms.py` closes over one list built for the whole structure.
Here the list is mapped too: one fixed-capacity list per system is built
outside `vmap` (building is data-dependent), the index tensors are
stacked, and the list is rebuilt from its slices inside the mapped
function. The two molecules differ in size, so the second is padded
(`numbers == 0`) to the first's atom count. A list that overflowed its
capacity is rejected by `check_compatible` before it reaches `vmap`.
"""

from __future__ import annotations

import pytest
import torch

from tad_mctc.io.structure import Structure
from tad_mctc.ncoord.common import CNModel
from tad_mctc.neighbor.list import NeighborList, build_neighborlist
from tad_mctc.typing import DD, Tensor

from ..conftest import DEVICE
from ..utils import load_batch
from ._variants import VARIANTS

PAIR = (("mb16_43", "01"), ("mb16_43", "SiH4"))


def _systems(dtype: torch.dtype) -> list[Structure]:
    """The padded batch `PAIR` split into single systems of equal `nat`."""
    dd: DD = {"device": DEVICE, "dtype": dtype}
    batch = load_batch(PAIR, dd)
    systems = [
        Structure(numbers=batch.numbers[i], positions=batch.positions[i])
        for i in range(2)
    ]
    # The test only means something if the molecules really differ.
    assert int((systems[0].numbers != 0).sum()) != int(
        (systems[1].numbers != 0).sum()
    )
    return systems


def _lists(
    model: CNModel, systems: list[Structure], capacity: int | None = None
) -> list[NeighborList]:
    nbls = [
        build_neighborlist(s, model.cutoff, capacity=capacity) for s in systems
    ]
    if capacity is not None:
        return nbls
    # Auto-sized lists differ in length (a multiple of 4096 per system), but
    # `vmap` needs one: rebuild every list at the largest capacity.
    largest = max(nbl.idx_i.shape[0] for nbl in nbls)
    return [
        build_neighborlist(s, model.cutoff, capacity=largest) for s in systems
    ]


def _mapped_cn(model: CNModel, systems: list[Structure]):
    """Returns `f(numbers, positions)` and the stacked list tensors, such
    that `vmap(f)(numbers, positions, *stacked)` is the CN of every system
    through its own list."""
    nbls = _lists(model, systems)
    for nbl, s in zip(nbls, systems):
        nbl.check_compatible(s, model.cutoff)
    assert len({nbl.idx_i.shape for nbl in nbls}) == 1  # shared capacity

    ref = nbls[0]
    stacked = tuple(
        torch.stack([getattr(nbl, name) for nbl in nbls])
        for name in ("idx_i", "idx_j", "shift", "mask")
    )

    def f(
        numbers: Tensor,
        positions: Tensor,
        idx_i: Tensor,
        idx_j: Tensor,
        shift: Tensor,
        mask: Tensor,
    ) -> Tensor:
        nbl = NeighborList(
            idx_i,
            idx_j,
            shift,
            mask,
            build_positions=positions,
            cutoff=ref.cutoff,
            skin=ref.skin,
            overflow=False,
        )
        return model(Structure(numbers=numbers, positions=positions), nbl)

    return f, stacked


@pytest.mark.parametrize("variant_name", list(VARIANTS))
@pytest.mark.parametrize(
    "dtype,atol", [(torch.double, 1e-12), (torch.float, 1e-5)]
)
def test_vmap_over_systems_values(
    variant_name: str, dtype: torch.dtype, atol: float
) -> None:
    model = VARIANTS[variant_name].call
    systems = _systems(dtype)
    f, stacked = _mapped_cn(model, systems)
    numbers = torch.stack([s.numbers for s in systems])
    positions = torch.stack([s.positions for s in systems])

    out = torch.func.vmap(f)(
        numbers, positions, *stacked
    )  # pyright: ignore[reportPrivateImportUsage]

    # Each system through its own list, one at a time.
    nbls = _lists(model, systems)
    loop = torch.stack([model(s, n) for s, n in zip(systems, nbls)])
    assert torch.allclose(out, loop, atol=atol, rtol=0)

    # Padding atoms have no pairs, so no coordination number.
    assert (out[numbers == 0] == 0).all()


@pytest.mark.parametrize("variant_name", list(VARIANTS))
def test_vmap_over_systems_gradient_finite(variant_name: str) -> None:
    model = VARIANTS[variant_name].call
    systems = _systems(torch.double)
    f, stacked = _mapped_cn(model, systems)
    numbers = torch.stack([s.numbers for s in systems])
    positions = torch.stack([s.positions for s in systems])

    def total(pos: Tensor, num: Tensor, *rest: Tensor) -> Tensor:
        return f(num, pos, *rest).sum()

    grad = torch.func.vmap(
        torch.func.grad(total)
    )(  # pyright: ignore[reportPrivateImportUsage]
        positions, numbers, *stacked
    )
    jac = torch.func.vmap(  # pyright: ignore[reportPrivateImportUsage]
        torch.func.jacrev(
            lambda p, n, *r: f(n, p, *r)
        )  # pyright: ignore[reportPrivateImportUsage]
    )(positions, numbers, *stacked)

    assert torch.isfinite(grad).all()
    assert torch.isfinite(jac).all()
    # Padding atoms feel no force.
    assert (grad[numbers == 0] == 0).all()


def test_overflowed_list_is_rejected_before_vmap() -> None:
    model = VARIANTS["cn_d3"].call
    systems = _systems(torch.double)
    nbls = _lists(model, systems, capacity=4)
    assert nbls[0].overflow
    with pytest.raises(ValueError, match="overflow"):
        nbls[0].check_compatible(systems[0], model.cutoff)


########################################################################
# Periodic cells, each with its own list and lattice

CELL_PAIR = (("other", "periodic_triclinic"), ("other", "periodic_one_atom"))
# The EN-weighted variants are identically zero on the one-atom cell, but
# not on the triclinic one, so they are still checked there.


def _cells() -> list[Structure]:
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    batch = load_batch(CELL_PAIR, dd)
    assert batch.lattice is not None and batch.periodic is not None
    return [
        Structure(
            numbers=batch.numbers[i],
            positions=batch.positions[i],
            lattice=batch.lattice[i],
            periodic=batch.periodic[i],
        )
        for i in range(2)
    ]


def _mapped_cell_cn(model: CNModel, cells: list[Structure]):
    """`f(numbers, positions, lattice, periodic, *stacked)` and the stacked
    list tensors, for lists built per cell."""
    nbls = _lists(model, cells)
    for nbl, c in zip(nbls, cells):
        nbl.check_compatible(c, model.cutoff)
    ref = nbls[0]
    stacked = tuple(
        torch.stack([getattr(nbl, name) for nbl in nbls])
        for name in ("idx_i", "idx_j", "shift", "mask")
    )

    def f(
        numbers: Tensor,
        positions: Tensor,
        lattice: Tensor,
        periodic: Tensor,
        idx_i: Tensor,
        idx_j: Tensor,
        shift: Tensor,
        mask: Tensor,
    ) -> Tensor:
        nbl = NeighborList(
            idx_i,
            idx_j,
            shift,
            mask,
            build_positions=positions,
            cutoff=ref.cutoff,
            skin=ref.skin,
            overflow=False,
            lattice=lattice,
            periodic_axes=periodic,
        )
        structure = Structure(
            numbers=numbers,
            positions=positions,
            lattice=lattice,
            periodic=periodic,
        )
        return model(structure, nbl)

    return f, nbls, stacked


@pytest.mark.parametrize("variant_name", list(VARIANTS))
def test_vmap_over_cells_values(variant_name: str) -> None:
    model = VARIANTS[variant_name].call
    cells = _cells()
    f, nbls, stacked = _mapped_cell_cn(model, cells)
    args = tuple(
        torch.stack([getattr(c, name) for c in cells])
        for name in ("numbers", "positions", "lattice", "periodic")
    )

    out = torch.func.vmap(f)(
        *args, *stacked
    )  # pyright: ignore[reportPrivateImportUsage]

    loop = torch.stack([model(c, n) for c, n in zip(cells, nbls)])
    assert torch.allclose(out, loop, atol=1e-12, rtol=0)
    assert (out[args[0] == 0] == 0).all()


@pytest.mark.parametrize("variant_name", list(VARIANTS))
def test_vmap_over_cells_gradients(variant_name: str) -> None:
    """`vmap(grad)` with respect to positions and to the lattice, each
    cell through its own list, matches `grad` of each cell on its own."""
    model = VARIANTS[variant_name].call
    cells = _cells()
    f, nbls, stacked = _mapped_cell_cn(model, cells)
    numbers, positions, lattice, periodic = (
        torch.stack([getattr(c, name) for c in cells])
        for name in ("numbers", "positions", "lattice", "periodic")
    )

    def total(pos, lat, num, per, *rest):
        return f(num, pos, lat, per, *rest).sum()

    grad = torch.func.grad  # pyright: ignore[reportPrivateImportUsage]
    d_pos, d_lat = torch.func.vmap(
        grad(total, argnums=(0, 1))
    )(  # pyright: ignore[reportPrivateImportUsage]
        positions, lattice, numbers, periodic, *stacked
    )

    assert torch.isfinite(d_pos).all() and torch.isfinite(d_lat).all()

    for i, (cell, nbl) in enumerate(zip(cells, nbls)):

        def single(pos: Tensor, lat: Tensor) -> Tensor:
            return model(cell.replace(positions=pos, lattice=lat), nbl).sum()

        e_pos, e_lat = grad(single, argnums=(0, 1))(
            cell.positions, cell.lattice
        )
        assert torch.allclose(d_pos[i], e_pos, atol=1e-10, rtol=0)
        assert torch.allclose(d_lat[i], e_lat, atol=1e-10, rtol=0)
