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
from tad_mctc.io.structure import Structure
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

    structure = Structure(
        numbers=numbers,
        positions=positions,
        charge=charge,
        uhf=uhf,
        lattice=lattice,
        periodic=periodic,
    )

    assert structure.charge is charge
    assert structure.uhf is uhf
    assert structure.lattice is lattice
    assert structure.periodic is periodic


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


def test_frozen_instance_cannot_be_mutated() -> None:
    """`Structure` is a value, not a place to assign into."""
    numbers, positions = _water()
    structure = Structure(numbers=numbers, positions=positions)

    with pytest.raises(dataclasses.FrozenInstanceError):
        structure.numbers = numbers  # type: ignore[misc]


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


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_to_moves_only_floating_fields_to_new_dtype(dtype: torch.dtype) -> None:
    """`.to(dtype=...)` must cast `positions`/`charge`/`lattice`, but leave
    `numbers`/`uhf`/`periodic` (integer/boolean) untouched -- mirroring
    `NeighborList.to()`."""
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

    moved = structure.to(dtype=dtype)

    assert moved.positions.dtype == dtype
    assert moved.charge is not None and moved.charge.dtype == dtype
    assert moved.lattice is not None and moved.lattice.dtype == dtype
    assert moved.numbers.dtype == torch.long
    assert moved.uhf is not None and moved.uhf.dtype == torch.long
    assert moved.periodic is not None and moved.periodic.dtype == torch.bool


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
