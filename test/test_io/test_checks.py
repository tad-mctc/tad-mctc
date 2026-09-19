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

from tad_mctc._version import __tversion__
from tad_mctc.exceptions import (
    DeviceError,
    DtypeError,
    StructureError,
    StructureWarning,
)
from tad_mctc.io import checks
from tad_mctc.typing import MockTensor

natoms = 4
ncart = 3


def test_coldfusion() -> None:
    # distances above threshold
    numbers = torch.tensor([1, 2])
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]])

    # Should pass without error
    assert checks.coldfusion_check(numbers, positions)

    # distances below threshold
    positions_close = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.1]])
    with pytest.raises(StructureError):
        checks.coldfusion_check(numbers, positions_close, threshold=0.5)


def test_coldfusion_threshold_already_a_tensor() -> None:
    # `threshold` given as a Tensor must be used as-is, not re-wrapped
    numbers = torch.tensor([1, 2])
    positions_close = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.1]])

    with pytest.raises(StructureError):
        checks.coldfusion_check(
            numbers, positions_close, threshold=torch.tensor(0.5)
        )


def test_coldfusion_check_disabled() -> None:
    numbers = torch.tensor([1, 2])
    positions_close = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.1]])

    assert checks.coldfusion_check(
        numbers, positions_close, threshold=0.5, check=False
    )
    assert checks.content_checks(
        numbers, positions_close, check_coldfusion=False
    )


def test_coldfusion_cutoff_smaller_than_threshold() -> None:
    # a clash just past the default cutoff must still be caught: `cutoff`
    # is raised internally to `threshold` whenever `threshold` is larger
    numbers = torch.tensor([1, 2])
    positions_close = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    with pytest.raises(StructureError):
        checks.coldfusion_check(
            numbers, positions_close, threshold=5.0, cutoff=0.1
        )


def test_coldfusion_sparse_matches_dense_on_padded_batch() -> None:
    # the O(nat) path (ndim == 2) and the dense fallback (ndim == 3, as used
    # for a padded batch) must agree on a passing case with real padding
    numbers = torch.tensor([[1, 8, 0], [1, 1, 1]])
    positions = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.5], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.5], [0.0, 1.5, 0.0]],
        ]
    )
    assert checks.coldfusion_check(numbers, positions)
    for n, p in zip(numbers, positions):
        assert checks.coldfusion_check(n, p)


def test_coldfusion_sparse_ignores_zero_padding_clash() -> None:
    # a single, unbatched structure with a zero-padded row must not compare
    # that phantom row against a real atom sitting at the origin
    numbers = torch.tensor([1, 8, 0])
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.5], [0.0, 0.0, 0.0]]
    )
    assert checks.coldfusion_check(numbers, positions)


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


@pytest.mark.skipif(__tversion__ < (2, 0, 0), reason="Requires torch>=2.0.0")
def test_coldfusion_functorch_via_jacrev() -> None:
    """
    Inside torch.func.jacrev, positions is a grad-tracking functorch tensor.
    coldfusion_check must short-circuit and return True rather than raising,
    even though the atoms are far too close to pass normally.
    """
    numbers = torch.tensor([1, 2])
    positions_close = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1e-10]])

    def f(pos: torch.Tensor) -> torch.Tensor:
        assert checks.coldfusion_check(numbers, pos) is True
        return pos.sum()

    _ = torch.func.jacrev(f)(  # pyright: ignore[reportPrivateImportUsage]
        positions_close
    )


@pytest.mark.skipif(__tversion__ < (2, 0, 0), reason="Requires torch>=2.0.0")
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
        assert checks.coldfusion_check(nums, pos) is True
        return pos.sum()

    _ = torch.func.vmap(f)(  # pyright: ignore[reportPrivateImportUsage]
        numbers_batch, positions_close
    )


@pytest.mark.skipif(__tversion__ < (2, 0, 0), reason="Requires torch>=2.0.0")
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
