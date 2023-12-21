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
Test calculation of EEQ coordination number.
"""
from __future__ import annotations

import pytest
import torch

from tad_mctc.batch import pack
from tad_mctc.data import radii
from tad_mctc.ncoord import cn_eeq as get_cn
from tad_mctc.typing import DD, Tensor

from ..conftest import DEVICE
from .samples import samples

sample_list = ["MB16_43_01", "MB16_43_02"]


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_single(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)

    rcov = radii.COV_D3.to(**dd)[numbers]
    cutoff = torch.tensor(30.0, **dd)
    ref = sample["cn_eeq"].to(**dd)

    cn = get_cn(numbers, positions, cutoff=cutoff, rcov=rcov, cn_max=None)
    assert pytest.approx(ref.cpu()) == cn.cpu()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("cn_max", [49, 51.0, torch.tensor(49)])
def test_single_cnmax(dtype: torch.dtype, cn_max: int | float | Tensor) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    sample = samples["MB16_43_01"]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)

    ref = sample["cn_eeq"].to(**dd)

    cn = get_cn(numbers, positions, cn_max=cn_max, cutoff=None)
    assert pytest.approx(ref.cpu(), abs=1e-5) == cn.cpu()


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
            sample1["cn_eeq"].to(**dd),
            sample2["cn_eeq"].to(**dd),
        )
    )

    cutoff = torch.tensor(30.0, **dd)
    cn = get_cn(numbers, positions, cutoff=cutoff, cn_max=None)
    assert pytest.approx(ref.cpu()) == cn.cpu()
