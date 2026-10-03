# SPDX-Identifier: CC0-1.0
"""Writing atomic numbers and positions to a Turbomole `coord` file."""

from pathlib import Path

from tad_mctc.data.structures import get_structure
from tad_mctc.io import write

# A small molecule from the bundled test structures.
structure = get_structure("other", "CO2")
numbers = structure.numbers
positions = structure.positions

path = Path(__file__).resolve().parent / "coord"
write.write_turbomole(path, numbers, positions)
