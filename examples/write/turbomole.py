# SPDX-Identifier: CC0-1.0
from pathlib import Path

import tad_mctc as mctc
from tad_mctc.data.molecules import mols

mol = mols["H2O"]
numbers = mol["numbers"]
positions = mol["positions"]

path = Path(__file__).resolve().parent / "coord"
mctc.io.write_turbomole_to_path(path, numbers, positions)
