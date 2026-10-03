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
Periodic cells beyond a plain right-handed bulk cell, through every
evaluation path (see `_paths.py`): atoms outside the primary cell, slabs
and wires, placeholder lattice vectors on open axes, and left-handed
cells. These are properties of the cell, not of how the coordination
number is summed, so every path must give the same answers.

Tests specific to one path, such as batches mixing bulk with slabs or the
periodic shifts' own mask, live in the quadrant modules.
"""

from __future__ import annotations

import pytest
import torch

from tad_mctc.io.structure import Structure
from tad_mctc.ncoord import cn_d3
from tad_mctc.typing import DD

from ..conftest import DEVICE
from ..utils import load_structure
from ._paths import Bind, bind_dense, bind_precomputed, bind_sparse
from .samples import PLACEHOLDER, carbon_pair

# Every path accepts a single periodic structure.
path_params = pytest.mark.parametrize(
    "bind",
    [bind_dense, bind_precomputed, bind_sparse],
    ids=["dense", "precomputed", "sparse"],
)

LOWER_DIMENSIONAL = [
    ([True, True, False], [6.0, 6.0, 40.0]),  # slab
    ([True, False, False], [6.0, 40.0, 40.0]),  # wire
]
"""Periodic masks with an orthorhombic cell whose open axes carry a long
vacuum vector."""


def _silicon_pair(
    dd: DD, lengths: list[float], periodic: list[bool]
) -> Structure:
    """Two silicon atoms in an orthorhombic cell with edge `lengths`."""
    return Structure(
        numbers=torch.tensor([14, 14], device=DEVICE),
        positions=torch.tensor([[0.0, 0.0, 0.0], [1.5, 1.5, 2.0]], **dd),
        lattice=torch.diag(torch.tensor(lengths, **dd)),
        periodic=torch.tensor(periodic, device=DEVICE),
    )


@path_params
def test_unwrapped_positions(bind: Bind) -> None:
    """An atom written 20 lattice vectors away from the origin (a stand-in
    for an unwrapped MD trajectory) must not change the result. Image
    search only covers a cutoff sphere anchored at the primary cell, so
    every path first folds positions into it -- the same invariant
    mctc-lib's `wrap_to_central_cell` establishes for its callers."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    structure = load_structure("other", "periodic_cubic", dd)
    assert structure.lattice is not None

    unwrapped_positions = structure.positions.clone()
    unwrapped_positions[0] += 20 * structure.lattice[0]
    unwrapped = structure.replace(positions=unwrapped_positions)

    wrapped_cn = bind(cn_d3, structure)(structure)
    unwrapped_cn = bind(cn_d3, unwrapped)(unwrapped)

    assert torch.allclose(wrapped_cn, unwrapped_cn, atol=1e-11, rtol=0)


@path_params
@pytest.mark.parametrize(
    "periodic,lengths", LOWER_DIMENSIONAL, ids=["slab", "wire"]
)
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_translation_by_lattice_vector(
    bind: Bind, periodic: list[bool], lengths: list[float], axis: int
) -> None:
    """Moving one atom by a whole lattice vector gives the same crystal
    along a periodic axis, but a different, more isolated system along an
    open one. Folding positions along an open axis would hide that."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    structure = _silicon_pair(dd, lengths, periodic)
    assert structure.lattice is not None

    moved_positions = structure.positions.clone()
    moved_positions[1] += structure.lattice[axis]
    moved = structure.replace(positions=moved_positions)

    baseline = bind(cn_d3, structure)(structure)
    after_move = bind(cn_d3, moved)(moved)

    same = torch.allclose(baseline, after_move, atol=1e-11, rtol=0)
    assert same == periodic[axis]


@path_params
@pytest.mark.parametrize(
    "periodic,lengths", LOWER_DIMENSIONAL, ids=["slab", "wire"]
)
def test_open_axis_lattice_vector_is_ignored(
    bind: Bind, periodic: list[bool], lengths: list[float]
) -> None:
    """A short placeholder vector on an open axis puts its images well
    inside the cutoff. Those images must be masked out, so the result
    matches the same system with a long vacuum vector."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    placeholder_lengths = [
        length if is_periodic else PLACEHOLDER
        for length, is_periodic in zip(lengths, periodic)
    ]
    vacuum = _silicon_pair(dd, lengths, periodic)
    placeholder = _silicon_pair(dd, placeholder_lengths, periodic)

    vacuum_cn = bind(cn_d3, vacuum)(vacuum)
    placeholder_cn = bind(cn_d3, placeholder)(placeholder)

    assert torch.allclose(vacuum_cn, placeholder_cn, atol=1e-11, rtol=0)


@path_params
@pytest.mark.parametrize(
    "periodic",
    [
        [True, True, True],  # bulk
        [True, True, False],  # slab, as a left-handed `$lattice` block
    ],
    ids=["bulk", "slab"],
)
def test_left_handed_cell_matches_right_handed(
    bind: Bind, periodic: list[bool]
) -> None:
    """Swapping the first two lattice vectors flips the handedness of the
    cell but describes the same crystal, so the coordination number must
    not change."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    lattice = torch.tensor(
        [[4.7, 0.0, 0.0], [0.6, 5.1, 0.0], [0.0, 0.0, 5.5]], **dd
    )
    if not periodic[2]:
        lattice[2, 2] = PLACEHOLDER

    right_handed = carbon_pair(dd, lattice, periodic)
    left_handed = right_handed.replace(lattice=lattice[[1, 0, 2]])
    assert left_handed.lattice is not None
    assert torch.linalg.det(left_handed.lattice) < 0

    expected = bind(cn_d3, right_handed)(right_handed)
    got = bind(cn_d3, left_handed)(left_handed)
    assert torch.allclose(got, expected, atol=1e-12, rtol=0)
