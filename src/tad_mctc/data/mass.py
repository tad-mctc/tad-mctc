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
Data: Masses
============

This module contains masses.
"""

from __future__ import annotations

import torch

from ..units.mass import GMOL2AU

__all__ = ["ATOMIC_MASS"]


def ATOMIC_MASS(
    device: torch.device | None = None, dtype: torch.dtype | None = torch.double
) -> torch.Tensor:
    """
    Isotope-averaged atom masses in atomic units (in g/mol) from
    https://www.angelo.edu/faculty/kboudrea/periodic/structure_mass.htm.

    Parameters
    ----------
    dtype : torch.dtype, optional
        Floating point precision for tensor. Defaults to `torch.double`.
    device : Optional[torch.device], optional
        Device of tensor. Defaults to None.

    Returns
    -------
    Tensor
        Atomic masses in atomic units.
    """
    if dtype is None:
        dtype = torch.double

    _ATOMIC = [
        0.0,  # dummy
        1.00797,
        4.00260,
        6.941,
        9.01218,
        10.81,
        12.011,
        14.0067,
        15.9994,
        18.998403,
        20.179,
        22.98977,
        24.305,
        26.98154,
        28.0855,
        30.97376,
        32.06,
        35.453,
        39.948,
        39.0983,
        40.08,
        44.9559,
        47.90,
        50.9415,
        51.996,
        54.9380,
        55.847,
        58.9332,
        58.70,
        63.546,
        65.38,
        69.72,
        72.59,
        74.9216,
        78.96,
        79.904,
        83.80,
        85.4678,
        87.62,
        88.9059,
        91.22,
        92.9064,
        95.94,
        98.0,
        101.07,
        102.9055,
        106.4,
        107.868,
        112.41,
        114.82,
        118.69,
        121.75,
        126.9045,
        127.60,
        131.30,
        132.9054,
        137.33,
        138.9055,
        140.12,
        140.9077,
        144.24,
        145,
        150.4,
        151.96,
        157.25,
        158.9254,
        162.50,
        164.9304,
        167.26,
        168.9342,
        173.04,
        174.967,
        178.49,
        180.9479,
        183.85,
        186.207,
        190.2,
        192.22,
        195.09,
        196.9665,
        200.59,
        204.37,
        207.2,
        208.9804,
        209,
        210,
        222,
    ]
    return GMOL2AU * torch.tensor(
        _ATOMIC, dtype=dtype, device=device, requires_grad=False
    )
