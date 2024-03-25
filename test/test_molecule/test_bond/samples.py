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
Molecules for testing bond orders.
"""

from __future__ import annotations

import torch

from tad_mctc.data.molecules import merge_nested_dicts, mols
from tad_mctc.typing import Molecule, Tensor, TypedDict


class Refs(TypedDict):
    """Format for reference values."""

    bo: Tensor
    """Reference bond orders."""

    cn: Tensor
    """DFT-D3 coordination number."""


class Record(Molecule, Refs):
    """Store for molecular information and reference values."""


refs: dict[str, Refs] = {
    "PbH4-BiH3": {
        "bo": torch.tensor(
            [
                0.6533,
                0.6533,
                0.6533,
                0.6499,
                0.6533,
                0.6533,
                0.6533,
                0.6499,
                0.5760,
                0.5760,
                0.5760,
                0.5760,
                0.5760,
                0.5760,
            ],
        ),
        "cn": torch.tensor(
            [
                3.9388208389,
                0.9832025766,
                0.9832026958,
                0.9832026958,
                0.9865897894,
                2.9714603424,
                0.9870455265,
                0.9870456457,
                0.9870455265,
            ],
        ),
    },
    "C6H5I-CH3SH": {
        "bo": torch.tensor(
            [
                0.4884,
                0.5180,
                0.4006,
                0.4884,
                0.4884,
                0.4012,
                0.4884,
                0.5181,
                0.4006,
                0.5181,
                0.5144,
                0.4453,
                0.5144,
                0.5145,
                0.4531,
                0.5180,
                0.5145,
                0.4453,
                0.4531,
                0.4012,
                0.4453,
                0.4006,
                0.4453,
                0.4006,
                0.6041,
                0.3355,
                0.6041,
                0.3355,
                0.5645,
                0.5673,
                0.5670,
                0.5645,
                0.5673,
                0.5670,
            ]
        ),
        "cn": torch.tensor(
            [
                3.1393690109,
                3.1313166618,
                3.1393768787,
                3.3153429031,
                3.1376547813,
                3.3148119450,
                1.5363609791,
                1.0035246611,
                1.0122337341,
                1.0036621094,
                1.0121959448,
                1.0036619902,
                2.1570565701,
                0.9981809855,
                3.9841127396,
                1.0146225691,
                1.0123561621,
                1.0085891485,
            ],
        ),
    },
}


samples: dict[str, Record] = merge_nested_dicts(mols, refs)
