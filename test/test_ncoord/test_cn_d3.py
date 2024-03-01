# This file is part of tad-mctc.
#
# SPDX-Identifier: LGPL-3.0
# Copyright (C) 2023 Marvin Friede
#
# tad-mctc is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# tad_mctc is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with tad-mctc. If not, see <https://www.gnu.org/licenses/>.
"""
Test calculation of DFT-D4 coordination number.
"""
from __future__ import annotations

import pytest
import torch

from tad_mctc.autograd import jacrev
from tad_mctc.batch import pack
from tad_mctc.convert import reshape_fortran, tensor_to_numpy
from tad_mctc.data import radii
from tad_mctc.ncoord import cn_d3 as get_cn
from tad_mctc.typing import DD, Tensor

from ..conftest import DEVICE
from .samples import samples

sample_list = ["SiH4", "PbH4-BiH3", "C6H5I-CH3SH", "MB16_43_01"]


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_single(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)

    rcov = radii.COV_D3.to(**dd)[numbers]
    cutoff = torch.tensor(30.0, **dd)
    ref = sample["cn_d3"].to(**dd)

    cn = get_cn(numbers, positions, rcov=rcov, cutoff=cutoff)
    assert pytest.approx(ref.cpu()) == cn.cpu()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", sample_list)
@pytest.mark.parametrize("name2", sample_list)
def test_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    sample1, sample2 = samples[name1], samples[name2]
    numbers = pack(
        (
            sample1["numbers"].to(DEVICE),
            sample2["numbers"].to(DEVICE),
        )
    )
    positions = pack(
        (
            sample1["positions"].to(**dd),
            sample2["positions"].to(**dd),
        )
    )
    ref = pack(
        (
            sample1["cn_d3"].to(**dd),
            sample2["cn_d3"].to(**dd),
        )
    )

    cn = get_cn(numbers, positions)
    assert pytest.approx(ref.cpu()) == cn.cpu()


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", ["SiH4", "MB16_43_01"])
def test_jacobian(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}
    tol = 1e-7

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)

    ref = sample["dcn3dr"].to(**dd)
    ref = reshape_fortran(ref, (3, *2 * (numbers.shape[0],)))
    ref = torch.einsum("xij->jix", ref)

    numgrad = calc_numgrad(numbers, positions)

    # variable to be differentiated
    positions.requires_grad_(True)

    fjac = jacrev(get_cn, argnums=1)
    jacobian: Tensor = fjac(numbers, positions)  # type: ignore
    jac_np = tensor_to_numpy(jacobian)

    assert pytest.approx(ref.cpu(), abs=tol * 10.5) == jac_np
    assert pytest.approx(ref.cpu(), abs=tol * 10) == numgrad.cpu()
    assert pytest.approx(numgrad.cpu(), abs=tol) == jac_np


def calc_numgrad(numbers: Tensor, positions: Tensor) -> Tensor:
    n_atoms = positions.shape[0]
    gradient = torch.zeros(
        (n_atoms, n_atoms, 3), device=positions.device, dtype=positions.dtype
    )
    step = 1.0e-6

    for i in range(n_atoms):
        for j in range(3):
            positions[i, j] += step
            cnr = get_cn(numbers, positions)

            positions[i, j] -= 2 * step
            cnl = get_cn(numbers, positions)

            positions[i, j] += step
            gradient[:, i, j] = 0.5 * (cnr - cnl) / step

    return gradient
