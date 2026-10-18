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
Test the Ewald splitting parameter against a line-by-line port of the
Fortran search (`search_alpha` of multicharge's `ewald.f90`).
"""

from __future__ import annotations

import math

import pytest
import torch
from torch.func import vmap

from tad_mctc.autograd import no_vmap_fallback
from tad_mctc.data.structures import get_structure
from tad_mctc.ewald import get_alpha
from tad_mctc.typing import DD, Tensor

from ..conftest import DEVICE

dd: DD = {"device": DEVICE, "dtype": torch.float64}

CELLS = [("x23", "CO2"), ("x23", "urea"), ("x23", "oxacb"), ("ice10", "vi")]


def _fortran_alpha(lattice: Tensor) -> float:
    """`search_alpha` of multicharge, with Python floats"""
    lat = lattice.cpu().double()
    volume = abs(torch.linalg.det(lat).item())
    rec = 2.0 * math.pi * torch.linalg.inv(lat).mT
    rlen = rec.norm(dim=-1).min().item()
    dlen = lat.norm(dim=-1).min().item()
    tol = math.sqrt(torch.finfo(torch.float64).eps)

    def rec_term(g: float, a: float) -> float:
        return (
            4.0 * math.pi * (math.exp(-0.25 * g * g / a**2) / (volume * g * g))
        )

    def dir_term(r: float, a: float) -> float:
        return math.erfc(a * r) / r

    def diff(a: float) -> float:
        return (rec_term(4 * rlen, a) - rec_term(5 * rlen, a)) - (
            dir_term(2 * dlen, a) - dir_term(3 * dlen, a)
        )

    alpha0 = 1.0e-8
    alpha = alpha0
    d = diff(alpha)
    while d < -tol:
        alpha *= 2.0
        d = diff(alpha)
    if alpha == alpha0:
        return 0.25

    alpl = 0.5 * alpha
    while d < tol:
        alpha *= 2.0
        d = diff(alpha)
    alpr = alpha

    alpha = (alpl + alpr) * 0.5
    ibs = 0
    d = diff(alpha)
    while abs(d) > tol and ibs <= 30:
        if d < 0:
            alpl = alpha
        else:
            alpr = alpha
        alpha = (alpl + alpr) * 0.5
        d = diff(alpha)
        ibs += 1
    if ibs > 30:
        return 0.25
    return alpha


@pytest.mark.parametrize("cell", CELLS, ids="-".join)
def test_matches_fortran(cell: tuple[str, str]) -> None:
    lattice = get_structure(*cell, **dd).lattice
    assert lattice is not None
    alpha = get_alpha(lattice)
    assert alpha.shape == ()
    assert alpha.item() == pytest.approx(_fortran_alpha(lattice), abs=1e-15)


def test_triclinic() -> None:
    lattice = torch.tensor(
        [[5.5, 0.0, 0.0], [1.0, 6.0, 0.0], [0.5, -0.7, 6.5]], **dd
    )
    assert get_alpha(lattice).item() == pytest.approx(
        _fortran_alpha(lattice), abs=1e-15
    )


def test_scales_inversely() -> None:
    """The search depends on the cell's shape and size only through
    lengths, so scaling the cell scales `alpha` inversely (up to the
    tolerance of the bisection)."""
    lattice = get_structure("x23", "urea", **dd).lattice
    assert lattice is not None
    alpha = get_alpha(lattice)
    assert (2.0 * get_alpha(2.0 * lattice)).item() == pytest.approx(
        alpha.item(), rel=1e-6
    )


def test_fallback_without_bracket() -> None:
    """A cell so large that the very first `alpha` is already balanced:
    the Fortran code falls back to 0.25."""
    lattice = 1e9 * torch.eye(3, **dd)
    assert _fortran_alpha(lattice) == 0.25
    assert get_alpha(lattice).item() == 0.25


def test_batch() -> None:
    lattices = torch.stack(
        [get_structure(*cell, **dd).lattice for cell in CELLS]  # type: ignore[misc]
    )
    alpha = get_alpha(lattices)
    assert alpha.shape == (len(CELLS),)
    for a, lattice in zip(alpha, lattices):
        assert a.item() == pytest.approx(_fortran_alpha(lattice), abs=1e-15)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dtype(dtype: torch.dtype) -> None:
    """The search runs in double precision, the result has the lattice's
    dtype."""
    lattice = get_structure("x23", "CO2", **dd).lattice
    assert lattice is not None
    alpha = get_alpha(lattice.to(dtype))
    assert alpha.dtype == dtype
    assert alpha.item() == pytest.approx(_fortran_alpha(lattice), rel=1e-6)


def test_no_gradient() -> None:
    lattice = get_structure("x23", "CO2", **dd).lattice
    assert lattice is not None
    alpha = get_alpha(lattice.clone().requires_grad_(True))
    assert not alpha.requires_grad


def test_vmap() -> None:
    lattices = torch.stack(
        [get_structure(*cell, **dd).lattice for cell in CELLS]  # type: ignore[misc]
    )
    with no_vmap_fallback():
        alpha = vmap(get_alpha)(lattices)
    assert torch.equal(alpha, get_alpha(lattices))
