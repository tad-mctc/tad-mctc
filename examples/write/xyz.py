# SPDX-Identifier: CC0-1.0
"""Writing atomic numbers and positions to an XYZ file, twice, appending the
second structure rather than overwriting the first."""

from pathlib import Path

from tad_mctc.data.structures import structures
from tad_mctc.io import write

# `tad_mctc.data.molecules.mols["H2O"]` no longer exists (that dataset was
# removed); `CO2` is the smallest bespoke structure the refactor kept.
structure = structures["CO2"]
numbers = structure.numbers
positions = structure.positions

path = Path(__file__).resolve().parent / "co2.xyz"
write.write_xyz(path, numbers, positions)

# write a second structure to the same file with "append" mode
write.write_xyz(path, numbers, positions, mode="a")
