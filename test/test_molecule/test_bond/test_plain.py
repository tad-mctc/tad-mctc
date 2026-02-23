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
Test bond order functionality for random examples.
"""

import pytest
import torch

from tad_mctc.batch import pack
from tad_mctc.molecule.bond import guess_bond_length, guess_bond_order
from tad_mctc.typing import DD

from ...conftest import DEVICE


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_bond_length(dtype: torch.dtype) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    numbers = torch.tensor([6, 8, 7, 1, 1, 1], device=DEVICE)
    cn = torch.tensor(
        [3.0059586, 1.0318390, 3.0268824, 1.0061584, 1.0036336, 0.9989871], **dd
    )
    ref = torch.tensor(
        [
            [2.5983, 2.2588, 2.5871, 1.9833, 1.9828, 1.9820],
            [2.2588, 2.1631, 2.2855, 1.5542, 1.5538, 1.5531],
            [2.5871, 2.2855, 2.5589, 1.8902, 1.8897, 1.8890],
            [1.9833, 1.5542, 1.8902, 1.4750, 1.4746, 1.4737],
            [1.9828, 1.5538, 1.8897, 1.4746, 1.4741, 1.4733],
            [1.9820, 1.5531, 1.8890, 1.4737, 1.4733, 1.4724],
        ],
        **dd
    )

    bond_length = guess_bond_length(numbers, cn)
    assert pytest.approx(ref.cpu(), abs=1e-4) == bond_length.cpu()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_bo_single(dtype: torch.dtype) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    numbers = torch.tensor([7, 7, 1, 1, 1, 1, 1, 1], device=DEVICE)
    positions = torch.tensor(
        [
            [-2.98334550857544, -0.08808205276728, +0.00000000000000],
            [+2.98334550857544, +0.08808205276728, +0.00000000000000],
            [-4.07920360565186, +0.25775116682053, +1.52985656261444],
            [-1.60526800155640, +1.24380481243134, +0.00000000000000],
            [-4.07920360565186, +0.25775116682053, -1.52985656261444],
            [+4.07920360565186, -0.25775116682053, -1.52985656261444],
            [+1.60526800155640, -1.24380481243134, +0.00000000000000],
            [+4.07920360565186, -0.25775116682053, +1.52985656261444],
        ],
        **dd
    )
    cn = torch.tensor([3.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], **dd)

    ref = torch.tensor(
        [
            [0.0000, 0.0000, 0.4403, 0.4334, 0.4403, 0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.4403, 0.4334, 0.4403],
            [0.4403, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            [0.4334, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            [0.4403, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            [0.0000, 0.4403, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            [0.0000, 0.4334, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            [0.0000, 0.4403, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        ],
        **dd
    )

    mask = torch.tensor(
        [
            [False, False, True, True, True, False, False, False],
            [False, False, False, False, False, True, True, True],
            [True, False, False, False, False, False, False, False],
            [True, False, False, False, False, False, False, False],
            [True, False, False, False, False, False, False, False],
            [False, True, False, False, False, False, False, False],
            [False, True, False, False, False, False, False, False],
            [False, True, False, False, False, False, False, False],
        ],
        device=DEVICE,
    )

    bo = guess_bond_order(numbers, positions, cn)
    assert pytest.approx(ref.cpu(), abs=1e-4) == bo.cpu()

    ref_mask = bo > 0.3
    assert (mask == ref_mask).all()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_bo_batch(dtype: torch.dtype) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    numbers = pack(
        (
            torch.tensor([7, 1, 1, 1], device=DEVICE),
            torch.tensor([6, 8, 8, 1, 1], device=DEVICE),
        )
    )
    positions = pack(
        (
            torch.tensor(
                [
                    [+0.00000000000000, +0.00000000000000, -0.54524837997150],
                    [-0.88451840382282, +1.53203081565085, +0.18174945999050],
                    [-0.88451840382282, -1.53203081565085, +0.18174945999050],
                    [+1.76903680764564, +0.00000000000000, +0.18174945999050],
                ],
                **dd
            ),
            torch.tensor(
                [
                    [-0.53424386915034, -0.55717948166537, +0.00000000000000],
                    [+0.21336223456096, +1.81136801357186, +0.00000000000000],
                    [+0.82345103924195, -2.42214694643037, +0.00000000000000],
                    [-2.59516465056138, -0.70672678063558, +0.00000000000000],
                    [+2.09259524590881, +1.87468519515944, +0.00000000000000],
                ],
                **dd
            ),
        )
    )
    cn = torch.tensor(
        [
            [2.9901006, 0.9977214, 0.9977214, 0.9977214, 0.0000000],
            [3.0093639, 2.0046251, 1.0187057, 0.9978270, 1.0069743],
        ],
        **dd
    )

    ref = pack(
        (
            torch.tensor(
                [
                    [0.0000, 0.4392, 0.4392, 0.4392, 0.0000],
                    [0.4392, 0.0000, 0.0000, 0.0000, 0.0000],
                    [0.4392, 0.0000, 0.0000, 0.0000, 0.0000],
                    [0.4392, 0.0000, 0.0000, 0.0000, 0.0000],
                    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                ],
                **dd
            ),
            torch.tensor(
                [
                    [0.0000, 0.5935, 0.4043, 0.3262, 0.0000],
                    [0.5935, 0.0000, 0.0000, 0.0000, 0.3347],
                    [0.4043, 0.0000, 0.0000, 0.0000, 0.0000],
                    [0.3262, 0.0000, 0.0000, 0.0000, 0.0000],
                    [0.0000, 0.3347, 0.0000, 0.0000, 0.0000],
                ],
                **dd
            ),
        )
    )

    bond_order = guess_bond_order(numbers, positions, cn)
    assert pytest.approx(ref.cpu(), abs=1e-4) == bond_order.cpu()
