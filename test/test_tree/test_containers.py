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
Test the containers built on `Node`: `Structure`, `PeriodicShifts`, `CNModel`
and `NeighborList`.
"""

from __future__ import annotations

import pytest
import torch
from torch.utils import _pytree as pytree

from tad_mctc.autograd import jacrev_matches_finite_diff
from tad_mctc.data import radii
from tad_mctc.io.structure import Structure
from tad_mctc.ncoord import cn_d3, cn_d4, cn_gfn2
from tad_mctc.ncoord.common import CNModel
from tad_mctc.neighbor.images import PeriodicShifts
from tad_mctc.neighbor.list import NeighborList, build_neighborlist
from tad_mctc.tools import is_compile_supported
from tad_mctc.tree import combine, leaf_paths

from ..utils import compile_fullgraph

# -- Structure ------------------------------------------------------------


def _water() -> tuple[torch.Tensor, torch.Tensor]:
    numbers = torch.tensor([8, 1, 1])
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 1.4, 1.1], [0.0, -1.4, 1.1]],
        dtype=torch.float64,
    )
    return numbers, positions


def _full_structure() -> Structure:
    numbers, positions = _water()
    return Structure(
        numbers=numbers,
        positions=positions,
        charge=torch.tensor(0.0, dtype=torch.float64),
        uhf=torch.tensor(0),
        lattice=20.0 * torch.eye(3, dtype=torch.float64),
        periodic=torch.tensor([True, True, True]),
        bonds=torch.tensor([[0, 1], [0, 2]]),
        bond_orders=torch.tensor([1.0, 1.0], dtype=torch.float64),
    )


def test_structure_round_trip() -> None:
    numbers, positions = _water()
    for structure in (
        Structure(numbers=numbers, positions=positions),
        _full_structure(),
    ):
        leaves, spec = pytree.tree_flatten(structure)
        new = pytree.tree_unflatten(leaves, spec)

        assert type(new) is Structure
        for name in structure._child_names:
            old, other = getattr(structure, name), getattr(new, name)
            assert (old is None) == (other is None)
            assert old is other


def test_structure_periodic_default() -> None:
    numbers, positions = _water()
    lattice = 20.0 * torch.eye(3, dtype=torch.float64)
    structure = Structure(numbers=numbers, positions=positions, lattice=lattice)

    assert structure.periodic is not None
    assert structure.periodic.dtype == torch.bool
    assert structure.periodic.shape == (3,)
    assert bool(structure.periodic.all())


def test_structure_casts_charge_to_positions_dtype() -> None:
    numbers, positions = _water()
    structure = Structure(
        numbers=numbers,
        positions=positions,
        charge=torch.tensor(1.0, dtype=torch.float32),
    )
    assert structure.charge is not None
    assert structure.charge.dtype == torch.float64


def test_structure_to_dtype() -> None:
    structure = _full_structure()
    new = structure.to(dtype=torch.float32)

    assert new.positions.dtype == torch.float32
    assert new.charge is not None and new.charge.dtype == torch.float32
    assert new.lattice is not None and new.lattice.dtype == torch.float32
    assert new.bond_orders is not None
    assert new.bond_orders.dtype == torch.float32

    assert new.numbers is structure.numbers
    assert new.uhf is structure.uhf
    assert new.periodic is structure.periodic
    assert new.bonds is structure.bonds


def test_structure_to_is_identity_if_unchanged() -> None:
    structure = _full_structure()
    assert structure.to() is structure


# -- PeriodicShifts -------------------------------------------------------


def _shifts() -> PeriodicShifts:
    return PeriodicShifts(
        shifts=torch.tensor([[0, 0, 0], [1, 0, 0], [-1, 0, 0]]),
        periodic_axes=torch.tensor([True, False, False]),
        cutoff=10.0,
    )


def test_periodic_shifts_round_trip() -> None:
    shifts = _shifts()
    leaves, spec = pytree.tree_flatten(shifts)
    new = pytree.tree_unflatten(leaves, spec)

    assert type(new) is PeriodicShifts
    assert new.shifts is shifts.shifts
    assert new.periodic_axes is shifts.periodic_axes
    assert new.cutoff == shifts.cutoff


def test_periodic_shifts_shape_error() -> None:
    with pytest.raises(
        RuntimeError, match=r"`shifts` must be an `\(n_shift, 3\)`"
    ):
        PeriodicShifts(
            shifts=torch.zeros(3, 2, dtype=torch.long),
            periodic_axes=torch.tensor([True, True, True]),
            cutoff=10.0,
        )


def test_periodic_shifts_to_meta() -> None:
    new = _shifts().to(device="meta")

    assert new.shifts.device.type == "meta"
    assert new.periodic_axes.device.type == "meta"
    assert new.shifts.dtype == torch.long
    assert new.periodic_axes.dtype == torch.bool


# -- CNModel --------------------------------------------------------------


def _water_structure() -> Structure:
    numbers, positions = _water()
    return Structure(numbers=numbers, positions=positions)


def test_cnmodel_round_trip() -> None:
    structure = _water_structure()
    for model in (cn_d3, cn_d4, cn_gfn2):
        leaves, spec = pytree.tree_flatten(model)
        new = pytree.tree_unflatten(leaves, spec)

        assert type(new) is CNModel
        assert torch.equal(new(structure), model(structure))


def test_cnmodel_tensor_rcov_is_leaf() -> None:
    structure = _water_structure()
    rcov = radii.COV_D3(dtype=torch.float64)
    model = cn_d3.replace(rcov=rcov)

    assert leaf_paths(model) == [".rcov"]

    def cn_of_rcov(r: torch.Tensor) -> torch.Tensor:
        return combine({".rcov": r}, model)(structure)

    assert torch.allclose(model(structure), cn_d3(structure))
    assert jacrev_matches_finite_diff(cn_of_rcov, rcov, eps=1e-5, atol=1e-6)


def test_cnmodel_replace_keeps_other_fields() -> None:
    new = cn_d3.replace(cutoff=40.0)

    assert new.cutoff == 40.0
    assert new.count is cn_d3.count
    assert new.cn_max is cn_d3.cn_max
    assert new.rcov is cn_d3.rcov
    assert new.en is cn_d3.en
    assert new.pair_weight is cn_d3.pair_weight


def test_cnmodel_cn_max_must_be_scalar() -> None:
    with pytest.raises(ValueError, match="scalar"):
        cn_d3.replace(cn_max=torch.ones(3))


def test_cnmodel_replace_same_structure() -> None:
    first = cn_d4.replace(cutoff=30.0)
    second = cn_d4.replace(cutoff=30.0)

    assert pytree.tree_structure(first) == pytree.tree_structure(second)


# -- NeighborList ---------------------------------------------------------


def _molecular_list() -> NeighborList:
    numbers, positions = _water()
    return build_neighborlist(
        Structure(numbers=numbers, positions=positions), 10.0, capacity=8
    )


def _periodic_list() -> NeighborList:
    numbers, positions = _water()
    lattice = 10.0 * torch.eye(3, dtype=torch.float64)
    structure = Structure(numbers=numbers, positions=positions, lattice=lattice)
    return build_neighborlist(structure, 8.0, capacity=64)


def test_neighborlist_molecular_to_meta_keeps_zero_stride_shift() -> None:
    nbl = _molecular_list()
    assert nbl.shift.stride() == (0, 1)

    new = nbl.to(device="meta")
    assert new.shift.device.type == "meta"
    assert new.shift.stride() == (0, 1)
    assert new.idx_i.device.type == "meta"


def test_neighborlist_round_trip() -> None:
    for nbl in (_molecular_list(), _periodic_list()):
        leaves, spec = pytree.tree_flatten(nbl)
        new = pytree.tree_unflatten(leaves, spec)

        assert type(new) is NeighborList
        for name in nbl._child_names:
            old, other = getattr(nbl, name), getattr(new, name)
            assert (old is None) == (other is None)
            assert old is other
        for name in nbl._context_names:
            assert getattr(new, name) == getattr(nbl, name)


def test_neighborlist_create_and_consistency() -> None:
    nbl = _periodic_list()
    assert nbl.periodic is True
    assert nbl.build_lattice is not None
    assert _molecular_list().periodic is False

    with pytest.raises(ValueError, match="periodic"):
        nbl.replace(periodic=False)


def test_cnmodel_replace_inside_compile() -> None:
    if not is_compile_supported():
        pytest.skip("torch.compile is not supported")

    structure = _water_structure()

    def fn(s: Structure) -> torch.Tensor:
        return cn_d3.replace(cutoff=10.0)(s)

    compiled = compile_fullgraph(fn)
    assert torch.allclose(compiled(structure), fn(structure))


def test_stacked_molecular_list_to_meta_keeps_batch_shape() -> None:
    from tad_mctc.tree import stack

    nbl = _molecular_list()
    new = stack([nbl, nbl]).to(device="meta")

    assert new.shift.shape == (2, *nbl.shift.shape)
    assert new.shift.stride()[:-1] == (0, 0)


def test_neighborlist_validate_periodic_mismatch() -> None:
    with pytest.raises(ValueError, match="periodic"):
        _periodic_list().replace(build_lattice=None)


def test_neighborlist_validate_numbers_shape_mismatch() -> None:
    nbl = _molecular_list()
    with pytest.raises(ValueError, match="numbers_shape"):
        nbl.replace(numbers_shape=(nbl.numbers_shape[0] + 1,))


def test_neighborlist_validate_index_dtype() -> None:
    nbl = _molecular_list()
    for name in ("idx_i", "idx_j"):
        with pytest.raises(ValueError, match=name):
            nbl.replace(**{name: getattr(nbl, name).to(torch.int64)})


def test_neighborlist_validate_mask_dtype() -> None:
    nbl = _molecular_list()
    with pytest.raises(ValueError, match="mask"):
        nbl.replace(mask=nbl.mask.to(torch.uint8))


def test_neighborlist_molecular_to_same_device_keeps_shift() -> None:
    nbl = _molecular_list()
    new = nbl.to(device=nbl.shift.device)

    assert new.shift.stride() == (0, 1)
    assert new.shift.device == nbl.shift.device
