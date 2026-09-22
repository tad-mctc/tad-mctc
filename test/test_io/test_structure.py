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
Test the `Structure` container: construction, validation wiring,
`.to()`/`.type()`, and its pytree registration under `vmap`, `jacrev` and
`torch.compile(fullgraph=True)`.

Validation itself (every branch of `structure_check`) is already covered at
the free-function level in `test/test_io/test_checks.py`; this module only
adds a smoke test proving `Structure.__post_init__` actually calls it,
plus construction-level and transform-level coverage that has no other
home.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch
from torch.utils._pytree import tree_flatten

from tad_mctc._version import __tversion__
from tad_mctc.autograd import jacrev, vmap
from tad_mctc.exceptions import DtypeError
from tad_mctc.io.structure import Structure, pack_structures
from tad_mctc.typing import Tensor

from ..utils import (
    DYNAMO_SUPPORTED,
    DYNAMO_UNSUPPORTED_REASON,
    run_compiled_or_skip,
)

_VMAP_PYTREE_UNSUPPORTED_REASON = (
    "vmap over a Structure needs torch.func (>= 2.0); the compat fallback "
    "(tad_mctc.autograd.vmap on older PyTorch) only accepts a raw Tensor "
    "as its first argument, not an arbitrary pytree"
)


def _water() -> tuple[Tensor, Tensor]:
    """Numbers and positions for a single, non-periodic water molecule."""
    numbers = torch.tensor([8, 1, 1], dtype=torch.long)
    positions = torch.tensor(
        [
            [0.00000000, 0.00000000, -0.14082324],
            [0.00000000, 1.43152878, 0.11185029],
            [0.00000000, -1.43152878, 0.11185029],
        ],
        dtype=torch.double,
    )
    return numbers, positions


def test_construction_minimal() -> None:
    """Only the two required fields need to be given; every optional field
    defaults to `None`."""
    numbers, positions = _water()

    structure = Structure(numbers=numbers, positions=positions)

    assert structure.numbers is numbers
    assert structure.positions is positions
    assert structure.charge is None
    assert structure.uhf is None
    assert structure.lattice is None
    assert structure.periodic is None


def test_construction_all_fields() -> None:
    """Every field can be set at once, including the periodic ones."""
    numbers, positions = _water()
    charge = torch.tensor(0.0)
    uhf = torch.tensor(0)
    lattice = 20.0 * torch.eye(3, dtype=torch.double)
    periodic = torch.tensor([True, True, True])
    bonds = torch.tensor([[0, 1], [0, 2]], dtype=torch.long)
    bond_orders = torch.tensor([1.0, 1.0], dtype=torch.double)

    structure = Structure(
        numbers=numbers,
        positions=positions,
        charge=charge,
        uhf=uhf,
        lattice=lattice,
        periodic=periodic,
        bonds=bonds,
        bond_orders=bond_orders,
    )

    assert structure.charge is charge
    assert structure.uhf is uhf
    assert structure.lattice is lattice
    assert structure.periodic is periodic
    assert structure.bonds is bonds
    assert structure.bond_orders is bond_orders


def test_construction_bonds_default_none() -> None:
    numbers, positions = _water()

    structure = Structure(numbers=numbers, positions=positions)

    assert structure.bonds is None
    assert structure.bond_orders is None


def test_construction_runs_structure_check() -> None:
    """`__post_init__` must wire into `structure_check`: a malformed
    periodicity dtype (one of `structure_check`'s own checks) has to raise
    from the constructor itself, not just from calling `structure_check`
    directly. This is a single smoke case, not a re-run of every branch
    already covered in `test_io/test_checks.py`."""
    numbers, positions = _water()
    wrong_dtype_periodic = torch.tensor([1, 1, 1])

    with pytest.raises(DtypeError):
        Structure(
            numbers=numbers, positions=positions, periodic=wrong_dtype_periodic
        )


##############################################################################
# periodicity mask
##############################################################################


@pytest.mark.parametrize("batch", [(), (2,)])
def test_lattice_without_mask_is_periodic_along_every_axis(
    batch: tuple[int, ...],
) -> None:
    """A lattice without a mask means periodic along all three axes, as
    in mctc-lib's `new_structure`; the mask is filled in, one row per
    lattice."""
    numbers, positions = _water()
    numbers = numbers.expand(*batch, -1)
    positions = positions.expand(*batch, -1, -1)
    lattice = 20.0 * torch.eye(3, dtype=torch.double).expand(*batch, 3, 3)

    structure = Structure(numbers=numbers, positions=positions, lattice=lattice)

    assert structure.periodic is not None
    assert structure.periodic.dtype == torch.bool
    assert structure.periodic.shape == (*batch, 3)
    assert structure.periodic.all()


def test_mask_without_lattice_raises() -> None:
    numbers, positions = _water()

    with pytest.raises(RuntimeError, match="without 'lattice'"):
        Structure(
            numbers=numbers,
            positions=positions,
            periodic=torch.ones(3, dtype=torch.bool),
        )


def test_mask_must_match_lattice_batch() -> None:
    """A `(3,)` mask applies to every lattice in a batch; a batched mask
    must match the lattice's batch shape."""
    numbers, positions = _water()
    numbers, positions = numbers.expand(2, -1), positions.expand(2, -1, -1)
    lattice = 20.0 * torch.eye(3, dtype=torch.double).expand(2, 3, 3)

    shared = Structure(
        numbers=numbers,
        positions=positions,
        lattice=lattice,
        periodic=torch.ones(3, dtype=torch.bool),
    )
    assert shared.periodic is not None and shared.periodic.shape == (3,)

    with pytest.raises(RuntimeError, match="batch shape"):
        Structure(
            numbers=numbers,
            positions=positions,
            lattice=lattice,
            periodic=torch.ones(5, 3, dtype=torch.bool),
        )


def test_replace_keeps_the_filled_in_mask() -> None:
    """`replace` copies the mask like any other field, so dropping the
    cell means clearing both fields."""
    numbers, positions = _water()
    lattice = 20.0 * torch.eye(3, dtype=torch.double)
    periodic = Structure(numbers=numbers, positions=positions, lattice=lattice)

    with pytest.raises(RuntimeError, match="without 'lattice'"):
        periodic.replace(lattice=None)

    molecule = periodic.replace(lattice=None, periodic=None)
    assert molecule.lattice is None and molecule.periodic is None


def test_vmap_fills_in_the_mask_per_lane() -> None:
    """A `Structure` built inside `vmap` gets a `(3,)` mask per lane."""
    numbers, positions = _water()
    lattices = torch.stack([20.0 * torch.eye(3), 30.0 * torch.eye(3)]).to(
        torch.double
    )

    def masked_diagonal(lat: Tensor) -> Tensor:
        structure = Structure(numbers=numbers, positions=positions, lattice=lat)
        assert structure.periodic is not None
        assert structure.periodic.shape == (3,)
        return torch.where(structure.periodic, lat.diagonal(), 0.0)

    result = vmap(masked_diagonal)(lattices)

    assert torch.equal(result, torch.stack([l.diagonal() for l in lattices]))


def test_compile_fullgraph_fills_in_the_mask() -> None:
    """Building a `Structure` with a lattice, and so its default mask,
    traces under `torch.compile(fullgraph=True)`."""
    numbers, positions = _water()
    lattice = 20.0 * torch.eye(3, dtype=torch.double)

    def masked_trace(lat: Tensor) -> Tensor:
        structure = Structure(numbers=numbers, positions=positions, lattice=lat)
        assert structure.periodic is not None
        return torch.where(structure.periodic, lat.diagonal(), 0.0).sum()

    result = run_compiled_or_skip(masked_trace, lattice)

    assert torch.allclose(result, masked_trace(lattice))


def test_frozen_instance_cannot_be_mutated() -> None:
    """`Structure` is a value, not a place to assign into."""
    numbers, positions = _water()
    structure = Structure(numbers=numbers, positions=positions)

    with pytest.raises(dataclasses.FrozenInstanceError):
        structure.numbers = numbers


def test_pytree_omits_absent_optional_fields() -> None:
    """The pytree leaves are exactly the *set* fields -- an absent optional
    field must not appear as a `None` leaf (that is the trap issue 18
    exists to avoid: `None` is itself a pytree leaf, which breaks `vmap`)."""
    numbers, positions = _water()
    structure = Structure(numbers=numbers, positions=positions)

    leaves, _ = tree_flatten(structure)

    assert len(leaves) == 2
    assert leaves[0] is numbers
    assert leaves[1] is positions


def test_pytree_includes_present_optional_fields() -> None:
    """A present optional field becomes a real leaf, in field-name order
    after the two required fields."""
    numbers, positions = _water()
    uhf = torch.tensor(0)
    structure = Structure(numbers=numbers, positions=positions, uhf=uhf)

    leaves, _ = tree_flatten(structure)

    assert len(leaves) == 3
    assert leaves[2] is uhf


def test_pytree_includes_bonds_after_periodic_fields() -> None:
    """`bonds`/`bond_orders` are the last two entries in `_OPTIONAL_FIELDS`,
    so they appear last among the leaves, after charge/uhf/lattice/
    periodic -- whichever of those are also present."""
    numbers, positions = _water()
    lattice = 20.0 * torch.eye(3, dtype=torch.double)
    periodic = torch.tensor([True, True, True])
    bonds = torch.tensor([[0, 1], [0, 2]], dtype=torch.long)
    bond_orders = torch.tensor([1.0, 1.0])
    structure = Structure(
        numbers=numbers,
        positions=positions,
        lattice=lattice,
        periodic=periodic,
        bonds=bonds,
        bond_orders=bond_orders,
    )

    leaves, _ = tree_flatten(structure)

    assert len(leaves) == 6
    assert leaves[2] is lattice
    assert leaves[3] is periodic
    assert leaves[4] is bonds
    assert leaves[5] is bond_orders


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_to_moves_only_floating_fields_to_new_dtype(dtype: torch.dtype) -> None:
    """`.to(dtype=...)` must cast `positions`/`charge`/`lattice`/
    `bond_orders`, but leave `numbers`/`uhf`/`periodic`/`bonds`
    (integer/boolean) untouched -- mirroring `NeighborList.to()`."""
    numbers, positions = _water()
    charge = torch.tensor(0.0)
    uhf = torch.tensor(0)
    lattice = 20.0 * torch.eye(3, dtype=torch.double)
    periodic = torch.tensor([True, True, True])
    bonds = torch.tensor([[0, 1], [0, 2]], dtype=torch.long)
    bond_orders = torch.tensor([1.0, 1.0], dtype=torch.double)

    structure = Structure(
        numbers=numbers,
        positions=positions,
        charge=charge,
        uhf=uhf,
        lattice=lattice,
        periodic=periodic,
        bonds=bonds,
        bond_orders=bond_orders,
    )

    moved = structure.to(dtype=dtype)

    assert moved.positions.dtype == dtype
    assert moved.charge is not None and moved.charge.dtype == dtype
    assert moved.lattice is not None and moved.lattice.dtype == dtype
    assert moved.bond_orders is not None
    assert moved.bond_orders.dtype == dtype
    assert moved.numbers.dtype == torch.long
    assert moved.uhf is not None and moved.uhf.dtype == torch.long
    assert moved.periodic is not None and moved.periodic.dtype == torch.bool
    assert moved.bonds is not None and moved.bonds.dtype == torch.long


def test_to_with_no_arguments_returns_self() -> None:
    """`.to()` with neither `device` nor `dtype` given is a no-op."""
    numbers, positions = _water()
    structure = Structure(numbers=numbers, positions=positions)

    assert structure.to() is structure


def test_type_delegates_to_to() -> None:
    """`.type(dtype)` is `.to(dtype=dtype)` keeping the current device."""
    numbers, positions = _water()
    structure = Structure(numbers=numbers, positions=positions.float())

    converted = structure.type(torch.float64)

    assert converted.positions.dtype == torch.float64
    assert converted.numbers.dtype == torch.long


@pytest.mark.cuda
def test_to_moves_every_set_field_to_cuda() -> None:
    """`.to(device=...)` must move every *set* field, required and
    optional, to the requested device -- a partial move would leave
    `structure_check`'s device-consistency check failing on the next
    construction. All four optional fields are set here (not just one),
    since `.to()` returning successfully at all already proves the
    constructor's device check passed across every field that was moved."""
    numbers, positions = _water()
    charge = torch.tensor(0.0)
    uhf = torch.tensor(0)
    lattice = 20.0 * torch.eye(3, dtype=torch.double)
    periodic = torch.tensor([True, True, True])
    structure = Structure(
        numbers=numbers,
        positions=positions,
        charge=charge,
        uhf=uhf,
        lattice=lattice,
        periodic=periodic,
    )

    moved = structure.to(device=torch.device("cuda"))

    assert moved.numbers.device.type == "cuda"
    assert moved.positions.device.type == "cuda"
    assert moved.charge is not None and moved.charge.device.type == "cuda"
    assert moved.uhf is not None and moved.uhf.device.type == "cuda"
    assert moved.lattice is not None and moved.lattice.device.type == "cuda"
    assert moved.periodic is not None and moved.periodic.device.type == "cuda"


@pytest.mark.skipif(
    __tversion__ < (2, 0, 0), reason=_VMAP_PYTREE_UNSUPPORTED_REASON
)
def test_vmap_over_batch_with_optional_field_absent() -> None:
    """`vmap(f, in_dims=0)` must work over a batch of instances that all
    uniformly omit an optional field -- the pytree treespec then has no
    `None` leaf to trip over."""
    batch_size = 4
    single_numbers, single_positions = _water()
    numbers = single_numbers.unsqueeze(0).expand(batch_size, -1).clone()
    positions = single_positions.unsqueeze(0).expand(batch_size, -1, -1).clone()
    batched = Structure(numbers=numbers, positions=positions)

    def total_charge_free_energy(structure: Structure) -> Tensor:
        return structure.positions.pow(2).sum()

    result = vmap(total_charge_free_energy, in_dims=0)(batched)
    expected = positions.pow(2).sum(dim=(-1, -2))

    assert torch.allclose(result, expected)


@pytest.mark.skipif(
    __tversion__ < (2, 0, 0), reason=_VMAP_PYTREE_UNSUPPORTED_REASON
)
def test_vmap_over_batch_with_optional_field_present() -> None:
    """The same batched `vmap`, but with an optional field (`uhf`)
    uniformly present across the batch instead of uniformly absent."""
    batch_size = 4
    single_numbers, single_positions = _water()
    numbers = single_numbers.unsqueeze(0).expand(batch_size, -1).clone()
    positions = single_positions.unsqueeze(0).expand(batch_size, -1, -1).clone()
    uhf = torch.zeros(batch_size, dtype=torch.long)
    batched = Structure(numbers=numbers, positions=positions, uhf=uhf)

    def uhf_plus_position_sum(structure: Structure) -> Tensor:
        assert structure.uhf is not None
        return structure.positions.sum() + structure.uhf.to(positions.dtype)

    result = vmap(uhf_plus_position_sum, in_dims=0)(batched)
    expected = positions.sum(dim=(-1, -2))

    assert torch.allclose(result, expected)


def test_jacrev_with_respect_to_positions() -> None:
    """`jacrev` through a function that builds a `Structure` from a
    differentiable `positions` tensor must see the same gradient as
    differentiating `positions` directly -- constructing (and validating)
    the container must not block autograd."""
    _, positions = _water()
    fixed_numbers = torch.tensor([8, 1, 1], dtype=torch.long)

    def sum_of_squares(pos: Tensor) -> Tensor:
        structure = Structure(numbers=fixed_numbers, positions=pos)
        return structure.positions.pow(2).sum()

    jacobian = jacrev(sum_of_squares)(positions)

    # d/dx sum(x**2) = 2*x, a closed-form identity, not a fabricated
    # reference value.
    assert torch.allclose(
        jacobian, 2.0 * positions
    )  # pyright: ignore[reportArgumentType]


def test_jacrev_with_respect_to_lattice() -> None:
    """Same guarantee as above, but differentiating with respect to
    `lattice` instead of `positions`."""
    numbers, positions = _water()
    lattice = 20.0 * torch.eye(3, dtype=torch.double)

    def sum_of_squares(lat: Tensor) -> Tensor:
        structure = Structure(numbers=numbers, positions=positions, lattice=lat)
        assert structure.lattice is not None
        return structure.lattice.pow(2).sum()

    jacobian = jacrev(sum_of_squares)(lattice)

    assert torch.allclose(
        jacobian, 2.0 * lattice
    )  # pyright: ignore[reportArgumentType]


def _leaf_op(structure: Structure) -> Tensor:
    """A trivial, differentiable reduction that never routes through
    `storch.cdist` (issue 17's already-tracked `torch.compile` blocker),
    so a `fullgraph=True` failure here would point at the container
    itself, not at that unrelated limitation."""
    total = structure.positions.sum()
    if structure.lattice is not None:
        total = total + structure.lattice.sum()
    return total


@pytest.mark.skipif(not DYNAMO_SUPPORTED, reason=DYNAMO_UNSUPPORTED_REASON)
def test_compile_fullgraph_without_lattice() -> None:
    """`torch.compile(fullgraph=True)` must trace through a `Structure`
    with only its required fields set."""
    numbers, positions = _water()
    structure = Structure(numbers=numbers, positions=positions)

    result = run_compiled_or_skip(_leaf_op, structure)

    assert torch.allclose(result, _leaf_op(structure))


@pytest.mark.skipif(not DYNAMO_SUPPORTED, reason=DYNAMO_UNSUPPORTED_REASON)
def test_compile_fullgraph_with_lattice() -> None:
    """Same guarantee, but with `lattice` also set -- a different pytree
    treespec, so `torch.compile` specializes and traces it separately."""
    numbers, positions = _water()
    lattice = 20.0 * torch.eye(3, dtype=torch.double)
    structure = Structure(numbers=numbers, positions=positions, lattice=lattice)

    result = run_compiled_or_skip(_leaf_op, structure)

    assert torch.allclose(result, _leaf_op(structure))


def _periodic_water(edge: float) -> Structure:
    """`_water()` in a cubic cell of edge length `edge`, periodic in all
    three directions."""
    numbers, positions = _water()
    return Structure(
        numbers=numbers,
        positions=positions,
        lattice=edge * torch.eye(3, dtype=torch.double),
        periodic=torch.ones(3, dtype=torch.bool),
    )


def test_pack_structures_pads_atoms_and_stacks_cell() -> None:
    """Per-atom fields are zero-padded to the largest structure, the
    per-structure cell fields are stacked."""
    water = _periodic_water(10.0)
    small_lattice = 5.0 * torch.eye(3, dtype=torch.double)
    single_atom = Structure(
        numbers=water.numbers[:1],
        positions=water.positions[:1],
        lattice=small_lattice,
        periodic=torch.tensor([True, True, False]),
    )

    batch = pack_structures([water, single_atom])

    assert batch.numbers.tolist() == [[8, 1, 1], [8, 0, 0]]
    assert batch.positions.shape == (2, 3, 3)
    assert (batch.positions[1, 1:] == 0).all()
    assert batch.lattice is not None and batch.periodic is not None
    assert torch.equal(batch.lattice[1], small_lattice)
    assert batch.periodic.tolist() == [[True, True, True], [True, True, False]]
    assert batch.charge is None


def test_pack_structures_reads_unset_charge_and_uhf_as_zero() -> None:
    """An unset charge/uhf means neutral/closed-shell, so it packs as zero
    next to a structure that sets it."""
    numbers, positions = _water()
    neutral = Structure(numbers=numbers, positions=positions)
    uhf = torch.tensor(1)
    radical = Structure(
        numbers=numbers[:2],
        positions=positions[:2],
        charge=torch.tensor(1.0, dtype=torch.double),
        uhf=uhf,
    )

    batch = pack_structures([neutral, radical])

    assert batch.lattice is None
    assert batch.charge is not None and batch.charge.tolist() == [0.0, 1.0]
    assert batch.uhf is not None and batch.uhf.tolist() == [0, 1]
    assert batch.uhf.dtype == uhf.dtype


def test_pack_structures_rejects_molecule_next_to_cell() -> None:
    numbers, positions = _water()
    molecule = Structure(numbers=numbers, positions=positions)

    with pytest.raises(ValueError, match="molecules with periodic"):
        pack_structures([molecule, _periodic_water(10.0)])


def test_pack_structures_accepts_cell_without_explicit_mask() -> None:
    """A cell whose mask was filled in packs next to one that set it."""
    numbers, positions = _water()
    implicit = Structure(
        numbers=numbers,
        positions=positions,
        lattice=12.0 * torch.eye(3, dtype=torch.double),
    )

    batch = pack_structures([implicit, _periodic_water(10.0)])

    assert batch.periodic is not None
    assert batch.periodic.tolist() == [[True] * 3, [True] * 3]


def test_pack_structures_accepts_molecule_as_non_periodic_cell() -> None:
    """The documented way to batch a molecule with periodic cells: any
    lattice together with an all-`False` periodic mask."""
    numbers, positions = _water()
    molecule = Structure(
        numbers=numbers,
        positions=positions,
        lattice=torch.eye(3, dtype=torch.double),
        periodic=torch.zeros(3, dtype=torch.bool),
    )

    batch = pack_structures([molecule, _periodic_water(10.0)])

    assert batch.periodic is not None
    assert batch.periodic.tolist() == [[False] * 3, [True] * 3]


def test_pack_structures_rejects_empty_and_batched_input() -> None:
    with pytest.raises(ValueError, match="empty"):
        pack_structures([])

    batch = pack_structures([_periodic_water(10.0), _periodic_water(12.0)])
    with pytest.raises(ValueError, match="unbatched"):
        pack_structures([batch])


def test_pack_structures_rejects_bonds() -> None:
    numbers, positions = _water()
    molecule = Structure(
        numbers=numbers,
        positions=positions,
        bonds=torch.tensor([[0, 1], [0, 2]]),
    )

    with pytest.raises(NotImplementedError):
        pack_structures([molecule, molecule])
