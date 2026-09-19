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
Data: Structures - Solids
==========================

Real periodic bulk solids with a verified crystallographic origin, but not
from mstore: mstore's own periodic coverage (see
:mod:`tad_mctc.data.structures.mstore`) is X23's discrete, hydrogen-bonded
organic molecular crystals and ice10's hydrogen-bonded ice polymorphs, with
no covalent-network or ionic bulk solid among them. `diamond` and `nacl`
fill that gap with the two bonding types X23/ice10 do not cover, mirroring
the same pair CP2K's own regression tests use as their standard bulk
examples (``c_32_diamond.inp``, ``c_32_NaCl.inp``).

Each is the conventional cubic cell (8 atoms) at its standard,
room-temperature experimental lattice constant, built here from that one
number plus the structure's textbook fractional coordinates rather than
hand-converted to Bohr, so the cited source stays checkable against the
positions actually used.
"""

from __future__ import annotations

import torch

from ...convert import symbol_to_number
from ...units.length import AA2AU

__all__ = ["solids"]


# Standard room-temperature experimental lattice constants; each fully
# determines its cubic conventional cell.
_A_DIAMOND = 3.567 * AA2AU
_A_NACL = 5.6402 * AA2AU


def _cubic_lattice(a: float) -> torch.Tensor:
    """Lattice vectors (rows) of a cubic cell with side length `a`."""
    return torch.tensor(
        [[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]], dtype=torch.double
    )


# Diamond cubic structure (space group Fd-3m): an FCC lattice with a
# two-atom basis at (0,0,0) and (1/4,1/4,1/4), i.e. 8 atoms in the
# conventional cubic cell.
_DIAMOND_FRAC = torch.tensor(
    [
        [0.00, 0.00, 0.00],
        [0.25, 0.25, 0.25],
        [0.00, 0.50, 0.50],
        [0.25, 0.75, 0.75],
        [0.50, 0.00, 0.50],
        [0.75, 0.25, 0.75],
        [0.50, 0.50, 0.00],
        [0.75, 0.75, 0.25],
    ],
    dtype=torch.double,
)

# Rock-salt structure (space group Fm-3m): an FCC lattice with a two-atom
# basis, Na at (0,0,0) and Cl at (1/2,1/2,1/2), i.e. 4 Na + 4 Cl atoms in
# the conventional cubic cell.
_NACL_FRAC = torch.tensor(
    [
        [0.00, 0.00, 0.00],
        [0.00, 0.50, 0.50],
        [0.50, 0.00, 0.50],
        [0.50, 0.50, 0.00],
        [0.50, 0.50, 0.50],
        [0.50, 0.00, 0.00],
        [0.00, 0.50, 0.00],
        [0.00, 0.00, 0.50],
    ],
    dtype=torch.double,
)


def _periodic_xyz() -> torch.Tensor:
    """A fresh `[True, True, True]` tensor -- each solid gets its own
    rather than sharing one mutable tensor between dict entries."""
    return torch.tensor([True, True, True], dtype=torch.bool)


solids: dict[str, dict[str, torch.Tensor]] = {
    "diamond": {
        "numbers": symbol_to_number(["C"] * 8),
        "positions": _DIAMOND_FRAC * _A_DIAMOND,
        "lattice": _cubic_lattice(_A_DIAMOND),
        "periodic": _periodic_xyz(),
    },
    "nacl": {
        "numbers": symbol_to_number(["Na"] * 4 + ["Cl"] * 4),
        "positions": _NACL_FRAC * _A_NACL,
        "lattice": _cubic_lattice(_A_NACL),
        "periodic": _periodic_xyz(),
    },
}
