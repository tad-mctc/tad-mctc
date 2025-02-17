# SPDX-Identifier: CC0-1.0
from pathlib import Path

from tad_mctc.data.molecules import mols
from tad_mctc.io import write

mol = mols["H2O"]
numbers = mol["numbers"]
positions = mol["positions"]

path = Path(__file__).resolve().parent / "coord"
write.write_turbomole(path, numbers, positions)
