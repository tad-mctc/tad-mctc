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
Test the uniform k-point grids `make_kpoints`, against PySCF's
`Cell.make_kpts`.
"""

from __future__ import annotations

import pytest
import torch
from torch.func import jacfwd, jacrev, vmap

from tad_mctc.kpoints import make_kpoints, to_fractional
from tad_mctc.typing import Tensor

from ..utils import compile_fullgraph
from .samples import LATTICE, dd


@pytest.mark.parametrize("mesh", [(1, 1, 1), (2, 3, 1), (3, 2, 4)])
@pytest.mark.parametrize("with_gamma_point", [True, False])
@pytest.mark.parametrize("wrap_around", [False, True])
@pytest.mark.parametrize("scaled_center", [None, (0.1, -0.25, 0.0)])
def test_pyscf(
    mesh: tuple[int, int, int],
    with_gamma_point: bool,
    wrap_around: bool,
    scaled_center: tuple[float, float, float] | None,
) -> None:
    """The points, their order and weights equal PySCF's."""
    pbcgto = pytest.importorskip("pyscf.pbc.gto")
    cell = pbcgto.M(
        atom=[("H", (0.0, 0.0, 0.0))],
        a=LATTICE.cpu().numpy(),
        unit="Bohr",
        basis="sto-3g",
        spin=1,
        verbose=0,
    )
    ref = cell.make_kpts(
        mesh,
        with_gamma_point=with_gamma_point,
        wrap_around=wrap_around,
        scaled_center=scaled_center,
    )
    kpts = make_kpoints(
        LATTICE,
        mesh,
        with_gamma_point=with_gamma_point,
        wrap_around=wrap_around,
        scaled_center=scaled_center,
    )
    assert torch.allclose(kpts.points.cpu(), torch.from_numpy(ref), atol=1e-12)
    nk = mesh[0] * mesh[1] * mesh[2]
    assert torch.allclose(kpts.weights, torch.full((nk,), 1 / nk, **dd))


def test_monkhorst_pack() -> None:
    """Without the Gamma point, the original Monkhorst-Pack fractions."""
    kpts = make_kpoints(LATTICE, (4, 1, 1), with_gamma_point=False)
    fractional = to_fractional(kpts.points, LATTICE)
    expected = torch.tensor([-3, -1, 1, 3], **dd) / 8
    assert torch.allclose(fractional[:, 0], expected)
    assert torch.allclose(fractional[:, 1:], torch.zeros(4, 2, **dd))


def test_wrap_around() -> None:
    """Wrapped fractions lie in [-1/2, 1/2), in the order of the grid."""
    kpts = make_kpoints(LATTICE, (4, 1, 1), wrap_around=True)
    fractional = to_fractional(kpts.points, LATTICE)
    expected = torch.tensor([0, 1, -2, -1], **dd) / 4
    assert torch.allclose(fractional[:, 0], expected)


def test_low_dimensional() -> None:
    """Non-periodic axes take one point at zero, and nothing else."""
    periodic = torch.tensor([True, False, True])
    kpts = make_kpoints(LATTICE, (2, 1, 3), periodic=periodic)
    fractional = to_fractional(kpts.points, LATTICE)
    assert torch.allclose(fractional[:, 1], torch.zeros(6, **dd))

    with pytest.raises(ValueError, match="Axis 1 is not periodic"):
        make_kpoints(LATTICE, (2, 2, 3), periodic=periodic)
    with pytest.raises(ValueError, match="Axis 1 is not periodic"):
        make_kpoints(
            LATTICE,
            (2, 1, 3),
            periodic=[True, False, True],
            scaled_center=(0.0, 0.5, 0.0),
        )


def test_low_dimensional_batch() -> None:
    """In a batch, an axis is non-periodic if it is in any structure."""
    lattice = torch.stack([LATTICE, 1.1 * LATTICE])
    periodic = torch.tensor([[True, True, True], [True, True, False]])
    kpts = make_kpoints(lattice, (2, 3, 1), periodic=periodic)
    assert kpts.points.shape == (2, 6, 3)
    with pytest.raises(ValueError, match="Axis 2 is not periodic"):
        make_kpoints(lattice, (1, 1, 2), periodic=periodic)


def test_invalid() -> None:
    """Meshes and centres of the wrong size are rejected."""
    with pytest.raises(ValueError, match="three positive"):
        make_kpoints(LATTICE, (2, 2))
    with pytest.raises(ValueError, match="three positive"):
        make_kpoints(LATTICE, (2, 0, 1))
    with pytest.raises(ValueError, match="three entries"):
        make_kpoints(LATTICE, (2, 2, 2), scaled_center=(0.0, 0.0))


def test_batch() -> None:
    """A batch of lattices gets one grid each."""
    lattices = torch.stack([LATTICE, 1.3 * LATTICE.flip(0)])
    kpts = make_kpoints(lattices, (2, 2, 1))
    assert kpts.points.shape == (2, 4, 3)
    for b in range(2):
        assert torch.allclose(
            kpts.points[b], make_kpoints(lattices[b], (2, 2, 1)).points
        )

    via_vmap = vmap(lambda lat: make_kpoints(lat, (2, 2, 1)).points)(lattices)
    assert torch.allclose(via_vmap, kpts.points)


def test_derivatives() -> None:
    """The grid moves with the lattice: fixed fractional coordinates."""

    def points(lattice: Tensor) -> Tensor:
        return make_kpoints(lattice, (3, 2, 2)).points

    rev = jacrev(points)(LATTICE)
    fwd = jacfwd(points)(LATTICE)
    assert torch.allclose(rev, fwd)

    # fractional coordinates do not change with the lattice
    frac = jacrev(lambda lat: to_fractional(points(lat), lat))(LATTICE)
    assert torch.allclose(frac, torch.zeros_like(frac), atol=1e-12)


def test_compile() -> None:
    """The grid compiles as one graph and equals eager."""

    def points(lattice: Tensor) -> Tensor:
        return make_kpoints(lattice, (3, 2, 2), with_gamma_point=False).points

    compiled = compile_fullgraph(points)
    assert torch.allclose(compiled(LATTICE), points(LATTICE))
