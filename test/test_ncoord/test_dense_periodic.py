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
The dense periodic quadrant: `CNModel.__call__` on a `Structure` with a
lattice, with a shift table built on every call (`pairs=None`) and with a
caller-built shift table (`pairs=shifts`).

Covers the shift table, and leading-batch-dimension batches (including
batches that mix bulk with slabs and wires), also under `vmap`. Cell
geometry that every evaluation path must handle alike lives in
`test_periodic_cells.py`. Agreement of both dense periodic paths with the
Fortran references is checked in `test_reference.py` and `test_grad/`, and
`vmap`, `jacrev` and `torch.compile` with a fixed shift table in
`test_transforms.py` and `test_compile.py`.
"""

from __future__ import annotations

import pytest
import torch
from torch.func import vmap

from tad_mctc.autograd import jacrev_matches_finite_diff
from tad_mctc.batch import pack
from tad_mctc.io.structure import Structure, pack_structures
from tad_mctc.ncoord import cn_d3, cn_d4, cn_eeq
from tad_mctc.ncoord.common import CNModel
from tad_mctc.ncoord.count import erf_count
from tad_mctc.neighbor.images import (
    build_periodic_shifts,
    build_shared_periodic_shifts,
)
from tad_mctc.typing import DD, Tensor

from ..conftest import DEVICE
from ..utils import load_structure
from .samples import PLACEHOLDER, carbon_pair

PERIODIC_CASES = [
    [True, True, False],  # slab
    [True, False, False],  # wire
    [False, False, False],  # molecule in a box
]


def _load_periodic_sample(collection: str, record: str, dd: DD) -> Structure:
    """`load_structure`, asserting the two periodic fields every test here
    needs are actually set."""
    structure = load_structure(collection, record, dd)
    assert structure.lattice is not None and structure.periodic is not None
    return structure


def _bulk_and_lower_dim(dd: DD, periodic: list[bool]) -> Structure:
    """A bulk cell packed with a system that is periodic only along
    `periodic`, with placeholder lattice vectors on its open axes."""
    bulk = carbon_pair(dd, 4.7 * torch.eye(3, **dd), [True, True, True])

    lattice = 4.7 * torch.eye(3, **dd)
    for axis, is_periodic in enumerate(periodic):
        if not is_periodic:
            lattice[axis, axis] = PLACEHOLDER
    lower_dim = carbon_pair(dd, lattice, periodic)

    return pack_structures([bulk, lower_dim])


def _single_system_cns(batch: Structure) -> Tensor:
    assert batch.lattice is not None and batch.periodic is not None
    return torch.stack(
        [
            cn_d3(
                Structure(
                    numbers=batch.numbers[i],
                    positions=batch.positions[i],
                    lattice=batch.lattice[i],
                    periodic=batch.periodic[i],
                )
            )
            for i in range(batch.numbers.shape[0])
        ]
    )


def _silicon_small_and_large() -> tuple[Structure, Structure]:
    """Two silicon cells of different atom counts and sizes. At a 9 Bohr
    cutoff the small, dense cell needs strictly more image rings than the
    large, sparse one."""
    small = Structure(
        numbers=torch.tensor([14, 14]),
        positions=torch.tensor(
            [[0.0, 0.0, 0.0], [1.5, 1.5, 1.5]], dtype=torch.double
        ),
        lattice=6.0 * torch.eye(3, dtype=torch.double),
    )
    large = Structure(
        numbers=torch.tensor([14, 14, 14]),
        positions=torch.tensor(
            [[0.0, 0.0, 0.0], [3.0, 3.0, 3.0], [6.0, 0.0, 0.0]],
            dtype=torch.double,
        ),
        lattice=20.0 * torch.eye(3, dtype=torch.double),
    )
    return small, large


########################################################################
# The shift table


def test_shifts_without_lattice_raise() -> None:
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
        model(structure, pairs=shifts)


def test_rejects_short_cutoff() -> None:
    """A `PeriodicShifts` bundle whose
    `.cutoff` falls short of the model's own is rejected, instead of
    silently under-counting the CN."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    structure = _load_periodic_sample("other", "periodic_one_atom", dd)
    assert structure.lattice is not None and structure.periodic is not None

    short_shifts = build_periodic_shifts(
        structure.lattice, structure.periodic, cutoff=cn_d3.cutoff / 2
    )

    with pytest.raises(ValueError):
        cn_d3(structure, pairs=short_shifts)


def test_rejects_missing_periodic_axis() -> None:
    """A `PeriodicShifts` bundle built without one of the structure's
    periodic axes is rejected, instead of silently dropping the images
    along that axis."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    structure = _load_periodic_sample("other", "periodic_one_atom", dd)
    assert structure.lattice is not None and structure.periodic is not None
    assert structure.periodic.all()

    slab_periodic = torch.tensor([True, True, False], device=DEVICE)
    slab_shifts = build_periodic_shifts(
        structure.lattice, slab_periodic, cutoff=cn_d3.cutoff
    )

    with pytest.raises(ValueError):
        cn_d3(structure, pairs=slab_shifts)


def test_structure_mask_not_table_mask_decides_wrap() -> None:
    """One shift table built for all three axes may serve a slab: the
    structure's own `periodic` mask, not the table's, decides which axes
    are folded into the primary cell. An atom moved by one whole lattice
    vector along the slab's vacuum axis must change the result, and
    marking that axis periodic in the structure must fold it back.
    `test_periodic_cells.py` checks the translation itself for every
    evaluation path."""
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

    baseline = cn_d3(slab(positions, periodic_slab), pairs=shifts)
    correct_slab = cn_d3(slab(displaced, periodic_slab), pairs=shifts)
    wrong_all_periodic = cn_d3(slab(displaced, periodic_all), pairs=shifts)

    # Moving atom 1 a full 40 Bohr along the vacuum axis is a real
    # physical change once z is correctly left unwrapped.
    assert not torch.allclose(baseline, correct_slab)
    # Treating z as periodic anyway silently folds the displacement away.
    assert torch.allclose(baseline, wrong_all_periodic, atol=1e-11, rtol=0)


########################################################################
# Batches with a leading batch dimension


def test_call_batched_numbers_matches_single_system_loop() -> None:
    """A leading batch dimension on `structure.numbers`/`structure.
    positions` routes to the batched dense periodic path and matches
    calling `__call__` once per system in a Python loop."""
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
    for each system individually. The shared table is sized to the more
    demanding system; the less demanding system's extra shift entries are
    masked out by the ordinary cutoff check rather than causing any error.
    Parametrized over a `pair_weight` preset (`cn_d4`) and a `cn_max`
    preset (`cn_eeq`) too, not just a bare `CNModel`: both branches are
    otherwise only exercised by the single-system path."""
    small, large = _silicon_small_and_large()
    assert small.lattice is not None and large.lattice is not None
    nat_small = small.numbers.shape[0]

    structure = Structure(
        numbers=pack([small.numbers, large.numbers]),
        positions=pack([small.positions, large.positions]),
        lattice=torch.stack([small.lattice, large.lattice]),
    )
    batched = model(structure)

    assert torch.allclose(
        batched[0, :nat_small], model(small), atol=1e-11, rtol=0
    )
    assert torch.allclose(batched[1], model(large), atol=1e-11, rtol=0)
    # Padding atoms contribute nothing.
    assert torch.allclose(
        batched[0, nat_small:], torch.zeros_like(batched[0, nat_small:])
    )


def test_call_batched_jacrev_wrt_positions_matches_finite_differences() -> None:
    """`jacrev` with respect to `positions` on the batched dense periodic
    path matches a finite-difference Jacobian for a batch that actually
    has a padding atom, so the atom-count/shift-count padding and masking
    do not silently break autodiff at batch scale (the failure mode
    `_cn_dense_per`'s own "mask before the square root" comment guards
    against for the single-system path). The padded slot in the smaller
    system lands at `[0, 0, 0]` -- `pack`'s zero padding -- which
    coincides exactly with that system's own atom 0, the same
    zero-distance situation the single-system comment warns about."""
    model = CNModel(count=erf_count, cutoff=9.0)

    small, large = _silicon_small_and_large()
    assert small.lattice is not None and large.lattice is not None
    numbers = pack([small.numbers, large.numbers])
    positions = pack([small.positions, large.positions])
    lattice = torch.stack([small.lattice, large.lattice])

    def f(p: torch.Tensor) -> torch.Tensor:
        structure = Structure(numbers=numbers, positions=p, lattice=lattice)
        return model(structure)

    assert jacrev_matches_finite_diff(f, positions)


########################################################################
# Batches mixing bulk with lower-dimensional systems


@pytest.mark.parametrize("periodic", PERIODIC_CASES)
def test_batch_matches_single_for_mixed_periodicity(
    periodic: list[bool],
) -> None:
    """A batch shares one shift table built for the union of its periodic
    axes. A system that is not periodic along one of those axes must
    still get the same coordination number as it does on its own."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    batch = _bulk_and_lower_dim(dd, periodic)

    batched = cn_d3(batch)

    expected = _single_system_cns(batch)
    assert torch.allclose(batched, expected, atol=1e-12, rtol=0)


@pytest.mark.parametrize("periodic", PERIODIC_CASES)
def test_vmap_matches_single_for_mixed_periodicity(
    periodic: list[bool],
) -> None:
    """Under `vmap` every lane is a single system, but the shift table is
    still shared and built for the union of the lanes' periodic axes."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    batch = _bulk_and_lower_dim(dd, periodic)
    assert batch.lattice is not None and batch.periodic is not None

    shifts = build_shared_periodic_shifts(
        batch.lattice, batch.periodic, cutoff=cn_d3.cutoff
    )

    def cn_one(
        numbers: Tensor, positions: Tensor, lattice: Tensor, mask: Tensor
    ) -> Tensor:
        structure = Structure(
            numbers=numbers, positions=positions, lattice=lattice, periodic=mask
        )
        return cn_d3(structure, pairs=shifts)

    vmapped = vmap(cn_one)(
        batch.numbers, batch.positions, batch.lattice, batch.periodic
    )

    expected = _single_system_cns(batch)
    assert torch.allclose(vmapped, expected, atol=1e-12, rtol=0)
