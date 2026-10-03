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
Test the checks for the numbers and positions given to the reader and writer.
"""

import pytest
import torch

from tad_mctc.exceptions import (
    DeviceError,
    DtypeError,
    StructureError,
    StructureWarning,
)
from tad_mctc.io import checks
from tad_mctc.io.structure import Structure, pack_structures
from tad_mctc.typing import MockTensor

natoms = 4
ncart = 3


def _molecule(numbers: list[int], positions: list[list[float]]) -> Structure:
    return Structure(
        numbers=torch.tensor(numbers), positions=torch.tensor(positions)
    )


def test_coldfusion() -> None:
    # distances above threshold
    apart = _molecule([1, 2], [[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]])
    assert checks.coldfusion_check(apart)

    # distances below threshold
    close = _molecule([1, 2], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.1]])
    with pytest.raises(StructureError):
        checks.coldfusion_check(close, threshold=0.5)


def test_coldfusion_default_threshold() -> None:
    # atoms 0.1 Bohr apart must be caught with no explicit `threshold`
    close = _molecule([1, 2], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.1]])

    with pytest.raises(StructureError):
        checks.coldfusion_check(close)


def test_coldfusion_threshold_already_a_tensor() -> None:
    # `threshold` given as a Tensor must be used as-is, not re-wrapped
    close = _molecule([1, 2], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.1]])

    with pytest.raises(StructureError):
        checks.coldfusion_check(close, threshold=torch.tensor(0.5))


def test_content_checks_leave_distances_alone() -> None:
    numbers = torch.tensor([1, 2])
    positions_close = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.1]])

    assert checks.content_checks(numbers, positions_close)


def test_coldfusion_cutoff_smaller_than_threshold() -> None:
    # `cutoff` must not limit the check: a clash beyond it is still caught
    close = _molecule([1, 2], [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    with pytest.raises(StructureError):
        checks.coldfusion_check(close, threshold=5.0, cutoff=0.1)


def test_coldfusion_sparse_cutoff_smaller_than_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # as above, through the neighbour list a large molecule would use
    monkeypatch.setattr(checks.structure, "_COLDFUSION_DENSE_MAX_ATOMS", 0)
    close = _molecule([1, 2], [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    with pytest.raises(StructureError):
        checks.coldfusion_check(close, threshold=5.0, cutoff=0.1)


def test_coldfusion_small_molecule_builds_no_neighbor_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # a small molecule is compared densely, so reading one never loads
    # (or compiles) the native neighbour-list extension
    def fail(*args: object) -> None:
        raise AssertionError("the neighbour list was built")

    monkeypatch.setattr(checks.structure, "_coldfusion_check_sparse", fail)
    close = _molecule([1, 2], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.1]])

    with pytest.raises(StructureError):
        checks.coldfusion_check(close)


def test_coldfusion_sparse_matches_dense_on_padded_batch() -> None:
    # a padded batch and its individual systems must agree on a passing
    # case with real padding
    numbers = torch.tensor([[1, 8, 0], [1, 1, 1]])
    positions = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.5], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.5], [0.0, 1.5, 0.0]],
        ]
    )
    assert checks.coldfusion_check(
        Structure(numbers=numbers, positions=positions)
    )
    for n, p in zip(numbers, positions):
        assert checks.coldfusion_check(Structure(numbers=n, positions=p))


def test_coldfusion_sparse_ignores_zero_padding_clash() -> None:
    # a single, unbatched structure with a zero-padded row must not compare
    # that phantom row against a real atom sitting at the origin
    padded = _molecule(
        [1, 8, 0], [[0.0, 0.0, 0.0], [0.0, 0.0, 1.5], [0.0, 0.0, 0.0]]
    )
    assert checks.coldfusion_check(padded)


def test_coldfusion_sparse_ignores_zero_padding_clash_in_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # as above, through the neighbour list a large molecule would use
    monkeypatch.setattr(checks.structure, "_COLDFUSION_DENSE_MAX_ATOMS", 0)
    padded = _molecule(
        [1, 8, 0], [[0.0, 0.0, 0.0], [0.0, 0.0, 1.5], [0.0, 0.0, 0.0]]
    )
    assert checks.coldfusion_check(padded)


def test_coldfusion_sparse_moderate_scale() -> None:
    # a molecule large enough to be checked through a non-trivial
    # neighbour-list build
    torch.manual_seed(0)
    nat = 1500
    assert nat > checks.structure._COLDFUSION_DENSE_MAX_ATOMS
    numbers = torch.ones(nat, dtype=torch.long)

    # Jittered grid, 2.5 Bohr apart: random positions at this density would
    # themselves put some pair closer than the default threshold (0.5 Bohr).
    axis = torch.arange(12, dtype=torch.double)
    grid = torch.cartesian_prod(axis, axis, axis)[:nat]
    jitter = 0.5 * (torch.rand((nat, 3), dtype=torch.double) - 0.5)
    positions = 2.5 * grid + jitter
    assert checks.coldfusion_check(
        Structure(numbers=numbers, positions=positions)
    )

    positions[1] = positions[0]
    with pytest.raises(StructureError):
        checks.coldfusion_check(Structure(numbers=numbers, positions=positions))


def _cell_with_contact_across_boundary(periodic: list[bool]) -> Structure:
    """Two atoms 0.2 Bohr apart through the x boundary of a 10 Bohr cell,
    but 9.8 Bohr apart inside it."""
    return Structure(
        numbers=torch.tensor([1, 1]),
        positions=torch.tensor(
            [[0.1, 5.0, 5.0], [9.9, 5.0, 5.0]], dtype=torch.double
        ),
        lattice=10.0 * torch.eye(3, dtype=torch.double),
        periodic=torch.tensor(periodic),
    )


def test_coldfusion_catches_contact_across_periodic_boundary() -> None:
    with pytest.raises(StructureError):
        checks.coldfusion_check(_cell_with_contact_across_boundary(3 * [True]))


def test_coldfusion_ignores_boundary_along_open_axis() -> None:
    """Along a non-periodic axis there is no image, so no contact."""
    wire_along_y = _cell_with_contact_across_boundary([False, True, True])
    assert checks.coldfusion_check(wire_along_y)


def test_coldfusion_catches_contact_in_batch_of_cells() -> None:
    fine = _cell_with_contact_across_boundary([False, True, True])
    fused = _cell_with_contact_across_boundary(3 * [True])

    assert checks.coldfusion_check(pack_structures([fine, fine]))
    with pytest.raises(StructureError):
        checks.coldfusion_check(pack_structures([fine, fused]))


def test_coldfusion_batch_sharing_one_cell() -> None:
    """One `(1, 3, 3)` cell is shared by every system of the batch."""
    fused = _cell_with_contact_across_boundary(3 * [True])
    assert fused.lattice is not None
    fine_positions = fused.positions.clone()
    fine_positions[1, 0] = 5.0  # 4.9 Bohr from atom 0, also across the cell

    def shared(*positions: torch.Tensor) -> Structure:
        return Structure(
            numbers=torch.tensor([[1, 1]] * len(positions)),
            positions=torch.stack(positions),
            lattice=fused.lattice.unsqueeze(0),
            periodic=fused.periodic,
        )

    assert checks.coldfusion_check(shared(*[fine_positions] * 3))
    with pytest.raises(StructureError):
        checks.coldfusion_check(
            shared(fine_positions, fused.positions, fine_positions)
        )


def test_coldfusion_large_batch_of_small_molecules_is_sparse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The all-pairs matrix of a whole trajectory is not built: past the
    size of one large molecule, the batch takes the neighbour list."""
    frames = 300
    atoms = 100  # 300 * 100**2 entries, past 1024**2 / 3
    numbers = torch.ones(frames, atoms, dtype=torch.long)
    positions = torch.zeros(frames, atoms, 3, dtype=torch.double)
    positions[..., 0] = 2.0 * torch.arange(atoms, dtype=torch.double)
    ok = Structure(numbers=numbers, positions=positions)
    assert checks.structure._coldfusion_uses_neighborlist(ok)

    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("the dense all-pairs matrix was built")

    monkeypatch.setattr(torch, "cdist", fail)
    assert checks.coldfusion_check(ok)

    fused = positions.clone()
    fused[-1, 1, 0] = fused[-1, 0, 0] + 0.1
    with pytest.raises(StructureError):
        checks.coldfusion_check(Structure(numbers=numbers, positions=fused))


def test_content() -> None:
    # Valid case
    numbers = torch.tensor([1, 8])
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]])
    assert checks.content_checks(numbers, positions)

    # Invalid case: atomic number too large (larger than pse.MAX_ELEMENT)
    numbers_large = torch.tensor([1, 200])
    with pytest.raises(StructureError):
        checks.content_checks(numbers_large, positions)

    # Invalid case: atomic number too small
    numbers_small = torch.tensor([1, 0])
    with pytest.raises(StructureError):
        checks.content_checks(numbers_small, positions, allow_batched=False)


def test_deflatable() -> None:
    positions = torch.tensor([[0.0, 0.0, 1.5], [0.0, 0.0, 0.0]])

    with pytest.warns(StructureWarning):
        checks.deflatable_check(positions, raise_padding_warning=True)


def test_deflatable_skip_warning() -> None:
    positions = torch.tensor([[0.0, 0.0, 1.5], [0.0, 0.0, 0.0]])
    assert checks.deflatable_check(positions, raise_padding_warning=False)


def test_shape_valid() -> None:
    # Valid shapes
    numbers = torch.zeros((natoms,))
    positions = torch.zeros((natoms, ncart))
    assert checks.shape_checks(numbers, positions, allow_batched=True)


def test_shape_mismatched_shapes() -> None:
    # Mismatched shapes between numbers and positions
    numbers = torch.zeros((natoms + 1,))
    positions = torch.zeros((natoms, ncart))
    with pytest.raises(ValueError):
        checks.shape_checks(numbers, positions)

    numbers = torch.zeros(0)
    positions = torch.zeros((1,))
    with pytest.raises(ValueError):
        checks.shape_checks(numbers, positions)


def test_shape_incorrect_dimensions_numbers() -> None:
    nbatch = 10

    # batch dimension
    numbers = torch.zeros((nbatch, natoms))
    positions = torch.zeros((nbatch, natoms, ncart))
    with pytest.raises(ValueError):
        checks.shape_checks(numbers, positions, allow_batched=False)


def test_shape_incorrect_cartesian_directions() -> None:
    # Incorrect size for the last dimension of positions
    numbers = torch.zeros(natoms)
    positions = torch.zeros((natoms, ncart - 1))
    with pytest.raises(ValueError):
        checks.shape_checks(numbers, positions)


def test_dimensions() -> None:

    numbers = torch.tensor([1, 8])
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]])
    charge = torch.tensor(0.0)

    assert checks.dimension_check(numbers, min_ndim=1, max_ndim=2)
    assert checks.dimension_check(positions, min_ndim=2, max_ndim=3)
    assert checks.dimension_check(charge, min_ndim=0, max_ndim=1)


def test_dimensions_fail() -> None:
    with pytest.raises(TypeError):
        checks.dimension_check(1, min_ndim=1, max_ndim=1)

    numbers = torch.tensor([[[1]]])
    with pytest.raises(RuntimeError):
        checks.dimension_check(numbers, max_ndim=2)

    positions = torch.tensor([0.0, 0.0, 0.0])
    with pytest.raises(RuntimeError):
        checks.dimension_check(positions, min_ndim=2)


###############################################################################
# structure_check (replaces the removed `Mol.checks`)
###############################################################################


def test_structure_check_valid() -> None:
    numbers = torch.randint(1, 118, (5,))
    positions = torch.randn((5, 3))
    assert checks.structure_check(numbers, positions) is True


def test_structure_check_shape_mismatch() -> None:
    numbers = torch.randint(1, 118, (5,))
    positions = torch.randn((5, 3))

    with pytest.raises(RuntimeError):
        checks.structure_check(torch.randint(1, 118, (1,)), positions)

    with pytest.raises(RuntimeError):
        checks.structure_check(numbers, torch.randn((4, 3)))


def test_structure_check_too_many_dimensions() -> None:
    positions = torch.randn((5, 3))
    numbers = torch.randint(1, 118, (5,))

    with pytest.raises(RuntimeError):
        checks.structure_check(torch.randint(1, 118, (1, 2, 3)), positions)

    with pytest.raises(RuntimeError):
        checks.structure_check(numbers, torch.randn(1, 2, 3, 4))


def test_structure_check_wrong_numbers_dtype() -> None:
    numbers = torch.randint(1, 118, (5,)).type(torch.float32)
    positions = torch.randn((5, 3))

    with pytest.raises(DtypeError):
        checks.structure_check(numbers, positions)


def test_structure_check_device_mismatch() -> None:
    numbers = torch.randint(1, 118, (5,), device=torch.device("cpu"))

    positions = MockTensor(torch.randn((5, 3)))
    positions.device = torch.device("cuda")

    with pytest.raises(DeviceError):
        checks.structure_check(numbers, positions)


def test_structure_check_lattice_wrong_shape() -> None:
    numbers = torch.randint(1, 118, (5,))
    positions = torch.randn((5, 3))
    bad_lattice = torch.eye(4)

    with pytest.raises(RuntimeError):
        checks.structure_check(numbers, positions, lattice=bad_lattice)


def test_structure_check_periodic_wrong_shape() -> None:
    numbers = torch.randint(1, 118, (5,))
    positions = torch.randn((5, 3))
    bad_periodic = torch.tensor([True, True])

    with pytest.raises(RuntimeError):
        checks.structure_check(numbers, positions, periodic=bad_periodic)


def test_structure_check_periodic_wrong_dtype() -> None:
    numbers = torch.randint(1, 118, (5,))
    positions = torch.randn((5, 3))
    bad_periodic = torch.ones(3)

    with pytest.raises(DtypeError):
        checks.structure_check(numbers, positions, periodic=bad_periodic)


def test_structure_check_bonds_valid() -> None:
    numbers = torch.randint(1, 118, (5,))
    positions = torch.randn((5, 3))
    bonds = torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.long)
    bond_orders = torch.tensor([1.0, 2.0, 1.0])

    assert checks.structure_check(
        numbers, positions, bonds=bonds, bond_orders=bond_orders
    )


def test_structure_check_bonds_alone_valid() -> None:
    """`bond_orders` is optional even when `bonds` is given -- unordered
    connectivity alone is a valid use case."""
    numbers = torch.randint(1, 118, (5,))
    positions = torch.randn((5, 3))
    bonds = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)

    assert checks.structure_check(numbers, positions, bonds=bonds)


def test_structure_check_bonds_wrong_last_dim() -> None:
    numbers = torch.randint(1, 118, (5,))
    positions = torch.randn((5, 3))
    bad_bonds = torch.tensor([[0, 1, 2]], dtype=torch.long)

    with pytest.raises(RuntimeError):
        checks.structure_check(numbers, positions, bonds=bad_bonds)


def test_structure_check_bonds_wrong_dtype() -> None:
    numbers = torch.randint(1, 118, (5,))
    positions = torch.randn((5, 3))
    bad_bonds = torch.tensor([[0, 1], [1, 2]], dtype=torch.float32)

    with pytest.raises(DtypeError):
        checks.structure_check(numbers, positions, bonds=bad_bonds)


def test_structure_check_bond_orders_without_bonds() -> None:
    """A bond order without the corresponding connectivity is meaningless."""
    numbers = torch.randint(1, 118, (5,))
    positions = torch.randn((5, 3))
    bond_orders = torch.tensor([1.0, 2.0])

    with pytest.raises(RuntimeError):
        checks.structure_check(numbers, positions, bond_orders=bond_orders)


def test_structure_check_bond_orders_count_mismatch() -> None:
    numbers = torch.randint(1, 118, (5,))
    positions = torch.randn((5, 3))
    bonds = torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.long)
    bond_orders = torch.tensor([1.0, 2.0])

    with pytest.raises(RuntimeError):
        checks.structure_check(
            numbers, positions, bonds=bonds, bond_orders=bond_orders
        )


def test_structure_check_lattice_periodic_valid() -> None:
    numbers = torch.randint(1, 118, (5,))
    positions = torch.randn((5, 3))
    lattice = 20.0 * torch.eye(3)
    periodic = torch.tensor([True, True, True])

    assert checks.structure_check(
        numbers, positions, lattice=lattice, periodic=periodic
    )


###############################################################################
# functorch short-circuit branches
###############################################################################


def test_coldfusion_functorch_via_jacrev() -> None:
    """
    Inside torch.func.jacrev, positions is a grad-tracking functorch tensor.
    coldfusion_check must short-circuit and return True rather than raising,
    even though the atoms are far too close to pass normally.
    """
    numbers = torch.tensor([1, 2])
    positions_close = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1e-10]])

    def f(pos: torch.Tensor) -> torch.Tensor:
        structure = Structure(numbers=numbers, positions=pos)
        assert checks.coldfusion_check(structure) is True
        return pos.sum()

    _ = torch.func.jacrev(f)(  # pyright: ignore[reportPrivateImportUsage]
        positions_close
    )


def test_coldfusion_functorch_via_vmap() -> None:
    """
    Inside torch.func.vmap, numbers is a batched functorch tensor.
    coldfusion_check must short-circuit on numbers before ever inspecting
    positions, so dangerously-close atoms do not raise.
    """
    numbers_batch = torch.tensor([[1, 2], [1, 2]], dtype=torch.long)
    positions_close = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1e-10]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1e-10]],
        ]
    )

    def f(nums: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        structure = Structure(numbers=nums, positions=pos)
        assert checks.coldfusion_check(structure) is True
        return pos.sum()

    _ = torch.func.vmap(f)(  # pyright: ignore[reportPrivateImportUsage]
        numbers_batch, positions_close
    )


def test_content_functorch_via_vmap() -> None:
    """
    Inside torch.func.vmap, numbers is a batched functorch tensor.
    `content_checks` must skip the max-element guard so that an oversized atomic
    number in one batch element does not raise.
    """
    numbers_batch = torch.tensor([[1, 2], [1, 999]], dtype=torch.long)
    positions_batch = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]],
        ]
    )

    def f(nums: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        assert checks.content_checks(nums, pos) is True
        return pos.sum()

    _ = torch.func.vmap(f)(  # pyright: ignore[reportPrivateImportUsage]
        numbers_batch, positions_batch
    )
