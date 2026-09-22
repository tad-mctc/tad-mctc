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
Data: Structures - Other - Periodic
=====================================

Every periodic (lattice-bearing) record in
:mod:`tad_mctc.data.structures.other`, split out purely for file-level
navigability; see that package's own docstring for why this split follows
content shape rather than mstore's per-source convention. Reached the
same way as any other bespoke record, through
:func:`tad_mctc.data.structures.get_structure` with collection
``"other"``; nothing outside
``other`` should import this module directly.

Two different reasons a record ends up here
--------------------------------------------
`diamond` and `nacl` are real bulk solids: the conventional cubic cell (8
atoms) at its standard, room-temperature experimental lattice constant,
built here from that one number plus the structure's textbook fractional
coordinates rather than hand-converted to Bohr, so the cited source stays
checkable against the positions actually used -- mirroring the same pair
CP2K's own regression tests use as their standard bulk examples
(``c_32_diamond.inp``, ``c_32_NaCl.inp``). mstore's own periodic coverage
(X23's discrete organic crystals, ice10's ice polymorphs) has no
covalent-network or ionic bulk solid; these two fill that gap.

`periodic_cubic`, `periodic_triclinic` and `periodic_one_atom` are the
opposite: synthetic cells with made-up coordinates, authored only to
exercise the PBC algorithm itself (a triclinic cell; a single atom whose
only neighbours are its own periodic images) -- there is no real crystal
to check them against, because none was intended.

Both kinds are periodic `Structure`s and nothing downstream distinguishes
between them -- periodicity is a property of a `Structure` (a set
`lattice`), never of which record produced it, so the two live in one
file and one `collection` (``"other"``) rather than two.
"""

from __future__ import annotations

import torch

from ....convert import symbol_to_number
from ....units.length import AA2AU

__all__ = ["periodic"]


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
    """A fresh `[True, True, True]` tensor -- each record gets its own
    rather than sharing one mutable tensor between dict entries."""
    return torch.tensor([True, True, True], dtype=torch.bool)


periodic: dict[str, dict[str, torch.Tensor]] = {
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
    "periodic_cubic": {
        "numbers": torch.tensor([3, 8, 14, 16, 17, 9]),
        "positions": torch.tensor(
            [
                [0.4, 0.7, 1.1],
                [2.3, 5.1, 0.8],
                [4.0, 2.6, 3.9],
                [1.2, 4.4, 5.0],
                [5.3, 0.9, 2.2],
                [3.1, 3.3, 4.6],
            ],
            dtype=torch.double,
        ),
        "lattice": torch.tensor(
            [[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]],
            dtype=torch.double,
        ),
        "periodic": torch.tensor([True, True, True], dtype=torch.bool),
    },
    "periodic_triclinic": {
        "numbers": torch.tensor([6, 7, 8, 15, 16]),
        "positions": torch.tensor(
            [
                [0.5, 0.6, 0.4],
                [2.1, 3.4, 1.0],
                [4.6, 1.2, 2.8],
                [1.8, 5.0, 4.1],
                [3.3, 2.7, 3.6],
            ],
            dtype=torch.double,
        ),
        "lattice": torch.tensor(
            [
                [7.0, 0.0, 0.0],
                [1.2, 6.5, 0.0],
                [0.6, 0.9, 6.0],
            ],
            dtype=torch.double,
        ),
        "periodic": torch.tensor([True, True, True], dtype=torch.bool),
    },
    "periodic_one_atom": {
        # A single Si atom in a 5 Bohr simple-cubic cell: every neighbour
        # within the search cutoff is a periodic image of this same atom,
        # so the `NeighborList` this builds is entirely self-image
        # entries. This exercises the folded-self-image consumer rule
        # (see `neighbor.list.NeighborList`'s docstring) exhaustively,
        # rather than as a small fraction of a larger cell's pairs.
        "numbers": torch.tensor([14]),
        "positions": torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.double),
        "lattice": torch.tensor(
            [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
            dtype=torch.double,
        ),
        "periodic": torch.tensor([True, True, True], dtype=torch.bool),
    },
}
