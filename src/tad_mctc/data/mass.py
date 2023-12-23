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
Data: Masses
============

This module contains masses.
"""
from __future__ import annotations

import torch

from ..units.mass import GMOL2AU

__all__ = ["ATOMIC"]


ATOMIC = GMOL2AU * torch.tensor(
    [
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
)
"""
Isotope-averaged atom masses in atomic units (save as g/mol) from
https://www.angelo.edu/faculty/kboudrea/periodic/structure_mass.htm.
"""
