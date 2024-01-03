# SPDX-Identifier: CC0-1.0
from pathlib import Path

import tad_mctc as mctc
from tad_mctc.data.molecules import mols

mol = mols["H2O"]
numbers = mol["numbers"]
positions = mol["positions"]

path = Path(__file__).resolve().parent / "h2o.xyz"
mctc.io.write_xyz_to_path(path, numbers, positions)

# write a second structure to the same file with "append" mode
mctc.io.write_xyz_to_path(path, numbers, positions, mode="a")
