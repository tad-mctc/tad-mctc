# SPDX-Identifier: CC0-1.0
"""Writing atomic numbers and positions to an XYZ file, twice, appending the
second structure rather than overwriting the first."""

from pathlib import Path

from tad_mctc.data.structures import get_structure
from tad_mctc.io import write

# A small molecule from the bundled test structures.
structure = get_structure("other", "CO2")
numbers = structure.numbers
positions = structure.positions

path = Path(__file__).resolve().parent / "co2.xyz"
write.write_xyz(path, numbers, positions)

# write a second structure to the same file with "append" mode
write.write_xyz(path, numbers, positions, mode="a")
