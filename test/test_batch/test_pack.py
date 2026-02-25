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
Test the packing utility functions.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tad_mctc.autograd import dgradcheck
from tad_mctc.batch import pack
from tad_mctc.convert import normalize_device
from tad_mctc.typing import DD

from ..conftest import DEVICE
from ..utils import _rng

mol1 = torch.tensor([1, 1])  # H2
mol2 = torch.tensor([8, 1, 1])  # H2O


def test_single_tensor() -> None:
    # dummy test: only give single tensor
    assert (mol1 == pack(mol1)).all()


def test_standard() -> None:
    # standard packing
    ref = torch.tensor(
        [
            [1, 1, 0],  # H2
            [8, 1, 1],  # H2O
        ],
    )
    packed = pack([mol1, mol2])
    assert (packed == ref).all()


def test_axis() -> None:
    ref = torch.tensor(
        [
            [1, 1, 0],  # H2
            [8, 1, 1],  # H2O
        ],
    )

    # different axis
    packed = pack([mol1, mol2], axis=-1)
    assert (packed == ref.T).all()


def test_size() -> None:
    ref = torch.tensor(
        [
            [1, 1, 0, 0],  # H2
            [8, 1, 1, 0],  # H2O
        ],
    )

    # one additional column of padding
    packed = pack([mol1, mol2], size=[4])
    assert (packed == ref).all()


def test_return_mask() -> None:
    packed, mask = pack(
        [
            torch.tensor([1.0]),
            torch.tensor([2.0, 2.0]),
            torch.tensor([3.0, 3.0, 3.0]),
        ],
        return_mask=True,
    )

    ref_packed = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [2.0, 2.0, 0.0],
            [3.0, 3.0, 3.0],
        ]
    )

    ref_mask = torch.tensor(
        [
            [True, False, False],
            [True, True, False],
            [True, True, True],
        ]
    )

    assert (packed == ref_packed).all()
    assert (mask == ref_mask).all()


def test_return_mask_axis() -> None:
    packed, mask = pack(
        [
            torch.tensor([1.0]),
            torch.tensor([2.0, 2.0]),
            torch.tensor([3.0, 3.0, 3.0]),
        ],
        axis=-1,
        return_mask=True,
    )

    ref_packed = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [2.0, 2.0, 0.0],
            [3.0, 3.0, 3.0],
        ]
    )

    ref_mask = torch.tensor(
        [
            [True, False, False],
            [True, True, False],
            [True, True, True],
        ]
    )

    # different axis
    assert (packed == ref_packed.T).all()
    assert (mask == ref_mask.T).all()


###############################################################################


def test_pack():
    """Sanity test of batch packing operation."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    # Generate matrix list
    sizes = np.random.randint(2, 8, (10,))
    matrices = [_rng((i, i), dd) for i in sizes]

    # Pack matrices into a single tensor
    packed = pack(matrices)

    # Construct a numpy equivalent
    max_size = max(packed.shape[1:])
    ref = np.stack(
        [np.pad(i.cpu().numpy(), (0, max_size - len(i))) for i in matrices]
    )

    equivalent = np.all((packed.cpu().numpy() - ref) < 1e-12)
    assert equivalent, "Check pack method against numpy"

    same_device = packed.device == normalize_device(DEVICE)
    assert same_device, "Device persistence check (packed tensor)"

    # Check that the mask is correct
    *_, mask = pack(
        [
            torch.rand(1, device=DEVICE),
            torch.rand(2, device=DEVICE),
            torch.rand(3, device=DEVICE),
        ],
        return_mask=True,
    )

    ref_mask = torch.tensor(
        [[1, 0, 0], [1, 1, 0], [1, 1, 1]], dtype=torch.bool, device=DEVICE
    )

    same_device_mask = mask.device == normalize_device(DEVICE)
    eq = torch.all(torch.eq(mask, ref_mask))

    assert eq, "Mask yielded an unexpected result"
    assert same_device_mask, "Device persistence check (mask)"


@pytest.mark.grad
def test_pack_grad():
    """Gradient stability test of batch packing operation."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    sizes = np.random.randint(2, 6, (3,))
    tensors = [_rng((i, i), dd).requires_grad_(True) for i in sizes]

    def proxy(*args):
        # Proxy function is used to prevent an undiagnosed error from occurring.
        return pack(list(args))

    grad_is_safe = dgradcheck(proxy, tensors, raise_exception=False)
    assert grad_is_safe, "Gradient stability test"
