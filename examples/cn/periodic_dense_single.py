# SPDX-Identifier: CC0-1.0
import torch

import tad_mctc as mctc
from tad_mctc.io.structure import Structure

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

structure = Structure(numbers=numbers, positions=positions, lattice=lattice)

# `structure.lattice` being set routes `CNModel.__call__` to the periodic
# path: it auto-builds the periodic-image table from `structure.lattice`
# on every call, so a changing lattice (e.g. cell relaxation) is always
# tracked for free. See `periodic_dense_vmap.py` for the alternative,
# `vmap`/`jacrev`-over-`lattice`-friendly entry point.
cn = mctc.ncoord.cn_d3(structure)
torch.set_printoptions(precision=10)
print(cn)
