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
`CNModel.with_precomputed_shifts`: the dense periodic path with a
caller-built shift table, so the table can stay fixed across a `vmap` over
positions or a `jacrev` w.r.t. the lattice. Agreement with the Fortran
reference itself is checked for every sample in `test_reference.py`,
through the plain `__call__`.
"""

from __future__ import annotations

import pytest
import torch

from tad_mctc.autograd import jacrev, vmap
from tad_mctc.io.structure import Structure
from tad_mctc.ncoord import cn_d3
from tad_mctc.neighbor.images import build_periodic_shifts
from tad_mctc.typing import DD, Tensor

from ._variants import VARIANTS
from ..conftest import DEVICE
from ..utils import load_structure


def _load_periodic_sample(collection: str, record: str, dd: DD) -> Structure:
    """`load_structure`, asserting the two periodic fields every test here
    needs are actually set."""
    structure = load_structure(collection, record, dd)
    assert structure.lattice is not None and structure.periodic is not None
    return structure


@pytest.mark.parametrize("variant_name", list(VARIANTS))
@pytest.mark.parametrize(
    "collection,record",
    [("other", "periodic_triclinic"), ("x23", "acetic")],
)
def test_matches_call(collection: str, record: str, variant_name: str) -> None:
    """Precomputed shifts give the same CN as `__call__`, which builds
    them itself. ``x23/acetic`` has atoms outside the primary cell, so it
    also covers the internal wrap."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    model = VARIANTS[variant_name].call

    structure = _load_periodic_sample(collection, record, dd)
    assert structure.lattice is not None and structure.periodic is not None

    shifts = build_periodic_shifts(
        structure.lattice, structure.periodic, cutoff=model.cutoff
    )
    cn = model.with_precomputed_shifts(structure, shifts=shifts)

    assert torch.allclose(cn, model(structure), atol=1e-12, rtol=0)


def test_periodic_cn_d3_dense_lattice_jacobian_is_finite() -> None:
    """`dCN/dlattice` through the dense path's `shifts.to(dtype) @
    lattice` term stays finite and well-defined -- the torch-func-
    friendly contract requires every physical input, including the
    lattice, to survive `jacrev`, not just `positions`. `dCN/dposition`
    for the periodic path, against the Fortran reference and for every
    variant, is checked in `test_grad/test_jacobian.py::
    test_matches_fortran_reference` instead, via the auto-build periodic
    dispatch rather than `with_precomputed_shifts` -- no need to
    duplicate that check here against this file's own `with_
    precomputed_shifts`-specific `shifts`."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    structure = _load_periodic_sample("other", "periodic_cubic", dd)
    assert structure.lattice is not None and structure.periodic is not None

    shifts = build_periodic_shifts(
        structure.lattice, structure.periodic, cutoff=cn_d3.cutoff
    )

    def f(lat: Tensor) -> Tensor:
        return cn_d3.with_precomputed_shifts(
            structure.replace(lattice=lat), shifts=shifts
        )

    jacobian = jacrev(f)(structure.lattice)
    assert isinstance(jacobian, Tensor)

    assert torch.isfinite(jacobian).all()


def test_vmap_over_positions_dense_periodic_with_one_shared_shifts() -> None:
    """`vmap` over a batch of `positions`, with one shared `lattice` and
    `shifts` table, matches a plain Python loop -- the other half of the
    ``vmap``-friendly claim in `_cn_dense_per`'s docstring, which
    `test_model.py`'s own vmap test only checks for `lattice`."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    structure = _load_periodic_sample("other", "periodic_cubic", dd)
    assert structure.lattice is not None and structure.periodic is not None

    shifts = build_periodic_shifts(
        structure.lattice, structure.periodic, cutoff=cn_d3.cutoff
    )
    batch = torch.stack(
        [
            structure.positions,
            structure.positions + 0.01,
            structure.positions - 0.01,
        ]
    )

    def f(p: Tensor) -> Tensor:
        return cn_d3.with_precomputed_shifts(
            structure.replace(positions=p), shifts=shifts
        )

    batched = vmap(f)(batch)
    looped = torch.stack([f(p) for p in batch])

    assert torch.allclose(batched, looped, atol=1e-12, rtol=0)


def test_periodic_cn_dense_rejects_short_cutoff() -> None:
    """`with_precomputed_shifts` raises for a `PeriodicShifts` bundle whose
    `.cutoff` falls short of the model's own, instead of silently
    under-counting the CN."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    structure = _load_periodic_sample("other", "periodic_one_atom", dd)
    assert structure.lattice is not None and structure.periodic is not None

    short_shifts = build_periodic_shifts(
        structure.lattice, structure.periodic, cutoff=cn_d3.cutoff / 2
    )

    with pytest.raises(ValueError):
        cn_d3.with_precomputed_shifts(structure, shifts=short_shifts)


def test_periodic_cn_dense_rejects_missing_periodic_axis() -> None:
    """`with_precomputed_shifts` raises for a `PeriodicShifts` bundle built
    without one of the structure's periodic axes, instead of silently
    dropping the images along that axis."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    structure = _load_periodic_sample("other", "periodic_one_atom", dd)
    assert structure.lattice is not None and structure.periodic is not None
    assert structure.periodic.all()

    slab_periodic = torch.tensor([True, True, False], device=DEVICE)
    slab_shifts = build_periodic_shifts(
        structure.lattice, slab_periodic, cutoff=cn_d3.cutoff
    )

    with pytest.raises(ValueError):
        cn_d3.with_precomputed_shifts(structure, shifts=slab_shifts)


def test_periodic_cn_dense_matches_for_unwrapped_positions() -> None:
    """`build_periodic_shifts`' ring count only covers a cutoff sphere
    anchored at the primary cell, so it is only valid once every atom
    lies inside that cell -- the same invariant mctc-lib's own
    `wrap_to_central_cell` (`mctc-lib/src/mctc/cutoff.f90`) establishes
    for its callers (s-dftd3 applies it to every structure entering its
    API). `_cn_dense_per` must therefore give the same answer whether or
    not the caller wrapped `positions` first: an
    atom written 20 lattice vectors away from the origin (a stand-in for
    an unwrapped MD trajectory) must not change the result."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    structure = _load_periodic_sample("other", "periodic_cubic", dd)
    assert structure.lattice is not None and structure.periodic is not None

    unwrapped_positions = structure.positions.clone()
    unwrapped_positions[0] = unwrapped_positions[0] + 20 * structure.lattice[0]
    unwrapped = structure.replace(positions=unwrapped_positions)

    shifts = build_periodic_shifts(
        structure.lattice, structure.periodic, cutoff=cn_d3.cutoff
    )

    wrapped_cn = cn_d3.with_precomputed_shifts(structure, shifts=shifts)
    unwrapped_cn = cn_d3.with_precomputed_shifts(unwrapped, shifts=shifts)

    assert torch.allclose(wrapped_cn, unwrapped_cn, atol=1e-11, rtol=0)
