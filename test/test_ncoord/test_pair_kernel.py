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
Tests of the pair kernel every evaluation path shares: the masked count
of one pair. The paths themselves, including the orientation of an
antisymmetric pair weight (``cn_eeq_en`` against the Fortran references),
are tested through :class:`tad_mctc.ncoord.common.CNModel` in the quadrant
modules.
"""

from __future__ import annotations

import torch

from tad_mctc.ncoord.common import _masked_pair_counts
from tad_mctc.ncoord.count import exp_count
from tad_mctc.typing import DD, Tensor

from ..conftest import DEVICE


def _distance_plus_radius(r: Tensor, r0: Tensor) -> Tensor:
    """A counting function whose value shows which distance and radius
    sum it received."""
    return r + r0


def test_counts_only_valid_pairs_within_cutoff() -> None:
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    # distances 2, 3 (exactly the cutoff), just beyond 3, and 1 (masked)
    distance_squared = torch.tensor([4.0, 9.0, 9.01, 1.0], **dd)
    rcov_pair_sum = torch.tensor([10.0, 20.0, 30.0, 40.0], **dd)
    valid = torch.tensor([True, True, True, False], device=DEVICE)

    counts = _masked_pair_counts(
        distance_squared,
        rcov_pair_sum,
        valid,
        count=_distance_plus_radius,
        cutoff=3.0,
    )

    expected = torch.tensor([12.0, 23.0, 0.0, 0.0], **dd)
    torch.testing.assert_close(counts, expected, atol=1e-14, rtol=0)


def test_masked_zero_distance_has_finite_derivatives() -> None:
    """An atom paired with itself sits at distance zero, where `sqrt` has
    an infinite derivative. Masked, it must contribute exactly zero to the
    first and second derivatives, not `nan`."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    distance_squared = torch.tensor([0.0, 4.0], **dd, requires_grad=True)
    rcov_pair_sum = torch.tensor([3.0, 3.0], **dd)
    valid = torch.tensor([False, True], device=DEVICE)

    counts = _masked_pair_counts(
        distance_squared, rcov_pair_sum, valid, count=exp_count, cutoff=25.0
    )
    (first,) = torch.autograd.grad(
        counts.sum(), distance_squared, create_graph=True
    )
    (second,) = torch.autograd.grad(first.sum(), distance_squared)

    assert first[0] == 0.0 and second[0] == 0.0
    assert torch.isfinite(first).all() and torch.isfinite(second).all()
    assert first[1] != 0.0
