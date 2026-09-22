# SPDX-Identifier: CC0-1.0
import torch

import tad_mctc as mctc
from tad_mctc.io.structure import Structure
from tad_mctc.neighbor.images import build_periodic_shifts

# a small cubic cell with a few different elements
numbers = torch.tensor([3, 8, 14, 16, 17, 9])

# coordinates in Bohr, inside the cell below
positions = torch.tensor(
    [
        [0.4, 0.7, 1.1],
        [2.3, 5.1, 0.8],
        [4.0, 2.6, 3.9],
        [1.2, 4.4, 5.0],
        [5.3, 0.9, 2.2],
        [3.1, 3.3, 4.6],
    ]
)

# cubic lattice vectors as rows, in Bohr
lattice = 8.0 * torch.eye(3)
periodic = torch.tensor([True, True, True])

# `torch.func.vmap` over `lattice` cannot go through `CNModel.__call__`:
# rebuilding the periodic-image table *inside* the trace hits
# `torch.func`'s data-dependent-control-flow restriction. Build the table
# once, outside the trace, at a cutoff that safely covers every lattice
# the trace will see, and reuse it with `with_precomputed_shifts` instead.
shifts = build_periodic_shifts(lattice, periodic, cutoff=mctc.ncoord.cn_d3.cutoff)

# three slightly expanded copies of the same cell, e.g. as `vmap` would
# see them while scanning a lattice-relaxation trajectory
batch_lattice = torch.stack([lattice * scale for scale in (1.00, 1.01, 1.02)])


def cn_of_lattice(lat: torch.Tensor) -> torch.Tensor:
    structure = Structure(numbers=numbers, positions=positions, lattice=lat)
    return mctc.ncoord.cn_d3.with_precomputed_shifts(structure, shifts=shifts)


cn = torch.func.vmap(cn_of_lattice)(batch_lattice)
torch.set_printoptions(precision=10)
print(cn)
