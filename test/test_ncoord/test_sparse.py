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
`CNModel` with `pairs=nbl` for both geometries: the call-time checks on
the structure and the pre-built list. Molecular and periodic sparse
evaluation share one kernel, so these checks are not split by geometry;
the geometry-specific tests live in `test_sparse_molecular.py` and
`test_sparse_periodic.py`.
"""

from __future__ import annotations

import pytest
import torch

from tad_mctc.io.structure import Structure
from tad_mctc.ncoord import cn_d3, erf_count
from tad_mctc.ncoord.common import CNModel
from tad_mctc.neighbor.images import build_periodic_shifts
from tad_mctc.neighbor.list import build_neighborlist
from tad_mctc.typing import DD

from ..conftest import DEVICE
from ..utils import load_structure
from .samples import PLACEHOLDER, carbon_pair


def _dimer(dd: DD) -> Structure:
    """Two hydrogen atoms one Bohr apart, the smallest valid molecule."""
    numbers = torch.tensor([1, 1], device=dd["device"])
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        device=dd["device"],
        dtype=dd["dtype"],
    )
    return Structure(numbers=numbers, positions=positions)


def _cloud(nat: int, dd: DD) -> Structure:
    """`nat` atoms scattered over a ~24 Bohr box, so that a 6 Bohr cutoff
    leaves many but far from all pairs."""
    generator = torch.Generator(device=dd["device"]).manual_seed(1)
    positions = torch.randn(
        nat, 3, device=dd["device"], dtype=dd["dtype"], generator=generator
    )
    numbers = torch.randint(
        1, 10, (nat,), device=dd["device"], generator=generator
    )
    return Structure(numbers=numbers, positions=positions * 12.0)


def test_pairs_can_be_passed_positionally() -> None:
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    model = CNModel(count=erf_count, cutoff=5.0)
    structure = _dimer(dd)
    nbl = build_neighborlist(structure, model.cutoff)

    assert torch.equal(model(structure, nbl), model(structure, pairs=nbl))


def test_unknown_mode_raises() -> None:
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    model = CNModel(count=erf_count, cutoff=5.0)
    structure = _dimer(dd)
    nbl = build_neighborlist(structure, model.cutoff)

    with pytest.raises(ValueError, match="mode"):
        model(structure, pairs=nbl, mode="not-a-mode")  # type: ignore[arg-type]


@pytest.mark.parametrize("source", ["none", "shifts"])
def test_recompute_without_list_raises(source: str) -> None:
    """`mode="recompute"` chunks a neighbour list's pair loop, so it has
    nothing to apply to for the all-pairs paths."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    model = CNModel(count=erf_count, cutoff=5.0)
    cell = load_structure("other", "periodic_cubic", dd)
    assert cell.lattice is not None and cell.periodic is not None
    pairs = (
        None
        if source == "none"
        else build_periodic_shifts(
            cell.lattice, cell.periodic, cutoff=model.cutoff
        )
    )

    with pytest.raises(ValueError, match="NeighborList"):
        model(cell, pairs=pairs, mode="recompute")


def test_list_for_other_atoms_raises() -> None:
    """A list indexes the atoms of the structure it was built from; a
    single system's list must not serve a batch."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    model = CNModel(count=erf_count, cutoff=5.0)
    single = _dimer(dd)
    nbl = build_neighborlist(single, model.cutoff)
    batch = Structure(
        numbers=single.numbers.repeat(2, 1),
        positions=single.positions.repeat(2, 1, 1),
    )

    with pytest.raises(ValueError, match="shape"):
        model(batch, pairs=nbl)


def test_periodic_list_without_lattice_raises() -> None:
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    model = CNModel(count=erf_count, cutoff=5.0)
    periodic = load_structure("other", "periodic_cubic", dd)
    nbl = build_neighborlist(periodic, model.cutoff)
    molecule = periodic.replace(lattice=None, periodic=None)

    with pytest.raises(ValueError, match="lattice"):
        model(molecule, pairs=nbl)


def test_molecular_list_with_lattice_raises() -> None:
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    model = CNModel(count=erf_count, cutoff=5.0)
    periodic = load_structure("other", "periodic_cubic", dd)
    molecule = periodic.replace(lattice=None, periodic=None)
    molecular_nbl = build_neighborlist(molecule, model.cutoff)

    with pytest.raises(ValueError, match="molecular list"):
        model(periodic, pairs=molecular_nbl)


def test_list_smaller_than_cutoff_raises() -> None:
    """A list built at a smaller cutoff is missing pairs; using it would
    silently under-count."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    model = CNModel(count=erf_count, cutoff=6.0)
    structure = _cloud(60, dd)
    small = build_neighborlist(structure, model.cutoff / 2)

    with pytest.raises(ValueError, match="cutoff"):
        model(structure, pairs=small)


def test_overflowed_list_raises() -> None:
    """A list built with an explicit `capacity` too small for the real
    pair count sets `nbl.overflow`; summing its truncated list would
    silently give a wrong coordination number."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    model = CNModel(count=erf_count, cutoff=6.0)
    structure = _cloud(150, dd)

    full = build_neighborlist(structure, model.cutoff, tile=16)
    npair = int(full.mask.sum().item())
    assert npair > 8, "test needs a system with more than 8 real pairs"

    overflowed = build_neighborlist(
        structure, model.cutoff, tile=16, capacity=npair // 2
    )
    assert overflowed.overflow is True

    with pytest.raises(ValueError, match="overflow"):
        model(structure, pairs=overflowed)


def test_list_with_spare_capacity_does_not_raise() -> None:
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    model = CNModel(count=erf_count, cutoff=6.0)
    structure = _cloud(150, dd)

    full = build_neighborlist(structure, model.cutoff, tile=16)
    npair = int(full.mask.sum().item())

    generous = build_neighborlist(
        structure, model.cutoff, tile=16, capacity=npair + 64
    )
    assert generous.overflow is False

    model(structure, pairs=generous)


def test_list_built_for_other_periodic_axes_raises() -> None:
    """A slab evaluated with a list built periodic along all three axes
    would count images along the short placeholder axis: without the
    check, the coordination number nearly doubles, silently."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    lattice = torch.diag(
        torch.tensor([4.0, 4.0, PLACEHOLDER], device=DEVICE, dtype=torch.double)
    )
    slab = carbon_pair(dd, lattice, [True, True, False])
    all_axes = torch.tensor([True, True, True], device=DEVICE)
    nbl = build_neighborlist(slab.replace(periodic=all_axes), cn_d3.cutoff)

    with pytest.raises(ValueError, match="periodic_axes"):
        cn_d3(slab, pairs=nbl)

    own_axes = build_neighborlist(slab, cn_d3.cutoff)
    sparse = cn_d3(slab, pairs=own_axes)
    assert torch.allclose(sparse, cn_d3(slab), atol=1e-11, rtol=0)
