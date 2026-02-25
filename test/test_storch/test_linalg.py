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
Test linalg safeops.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tad_mctc import storch
from tad_mctc.autograd import dgradcheck
from tad_mctc.convert import numpy_to_tensor, symmetrize
from tad_mctc.typing import DD, Literal, Tensor

from ..conftest import DEVICE, FAST_MODE

# Hamiltonian of LiH from last step
hamiltonian = torch.tensor(
    [
        [
            -0.27474006548256,
            -0.00000000000000,
            -0.00000000000000,
            -0.00000000000000,
            -0.22679941570507,
            0.07268461913372,
        ],
        [
            -0.00000000000000,
            -0.17641725918816,
            -0.00000000000000,
            -0.00000000000000,
            0.00000000000000,
            0.00000000000000,
        ],
        [
            -0.00000000000000,
            -0.00000000000000,
            -0.17641725918816,
            -0.00000000000000,
            -0.28474359171632,
            0.02385107216679,
        ],
        [
            -0.00000000000000,
            -0.00000000000000,
            -0.00000000000000,
            -0.17641725918816,
            0.00000000000000,
            0.00000000000000,
        ],
        [
            -0.22679941570507,
            0.00000000000000,
            -0.28474359171632,
            0.00000000000000,
            -0.33620576141638,
            0.00000000000000,
        ],
        [
            0.07268461913372,
            0.00000000000000,
            0.02385107216679,
            0.00000000000000,
            0.00000000000000,
            -0.01268791523447,
        ],
    ],
    dtype=torch.float64,
    device=DEVICE,
)


@pytest.mark.parametrize("broadening", ["none", "cond", "lorn"])
@pytest.mark.parametrize("dtype", [torch.double])
def test_eighb(
    broadening: Literal["cond", "lorn", "none"], dtype: torch.dtype
) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}
    a = numpy_to_tensor(np.random.rand(8, 8), **dd)
    a.requires_grad_(True)

    def eigen_proxy(m: Tensor) -> tuple[Tensor, Tensor]:
        m = symmetrize(m, force=True)
        return storch.linalg.eighb(a=m, broadening_method=broadening)

    assert dgradcheck(eigen_proxy, a, fast_mode=FAST_MODE)


@pytest.mark.xfail
@pytest.mark.parametrize("broadening", ["none", "cond", "lorn"])
@pytest.mark.parametrize("dtype", [torch.double])
def test_eighb_degen(
    broadening: Literal["cond", "lorn", "none"], dtype: torch.dtype
) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}
    hamiltonian.detach_().to(**dd).requires_grad_(True)

    def eigen_proxy(m: Tensor) -> tuple[Tensor, Tensor]:
        m = symmetrize(m, force=True)
        return storch.linalg.eighb(a=m, broadening_method=broadening)

    assert dgradcheck(eigen_proxy, hamiltonian, fast_mode=FAST_MODE)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_gershgorin(dtype: torch.dtype) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    amat = torch.tensor(
        [
            [
                [-1.1258, -0.1794, 0.1126],
                [-0.1794, 0.5988, 0.1490],
                [0.1126, 0.1490, 0.4681],
            ],
            [
                [-0.1577, 0.6080, -0.3301],
                [0.6080, 1.5863, 0.9391],
                [-0.3301, 0.9391, 1.2590],
            ],
        ],
        **dd,
    )

    _min, _max = storch.linalg.estimate_minmax(amat)

    ref_min = torch.tensor([-1.4178, -1.0958], **dd)
    ref_max = torch.tensor([0.9272, 3.1334], **dd)

    assert pytest.approx(ref_min.cpu(), abs=1e-4, rel=1e-4) == _min.cpu()
    assert pytest.approx(ref_max.cpu(), abs=1e-4, rel=1e-4) == _max.cpu()

    evals = torch.linalg.eigh(amat)[0]
    _emin = evals.min(-1)[0]
    _emax = evals.max(-1)[0]

    ref_emin = torch.tensor([-1.1543, -0.5760], **dd)
    ref_emax = torch.tensor([0.7007, 2.4032], **dd)

    assert pytest.approx(ref_emin.cpu(), abs=1e-4, rel=1e-4) == _emin.cpu()
    assert pytest.approx(ref_emax.cpu(), abs=1e-4, rel=1e-4) == _emax.cpu()

    # Gershgorin works?
    assert pytest.approx(_emin.cpu(), abs=0.7, rel=0.5) == _min.cpu()
    assert pytest.approx(_emax.cpu(), abs=0.7, rel=0.5) == _emax.cpu()
