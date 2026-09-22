# SPDX-Identifier: CC0-1.0
"""Writing atomic numbers and positions to a Turbomole `coord` file."""

from pathlib import Path

from tad_mctc.data.structures import get_structure
from tad_mctc.io import write

# `tad_mctc.data.molecules.mols["H2O"]` no longer exists (that dataset was
# removed); `CO2` is the smallest bespoke structure the refactor kept.
structure = get_structure("other", "CO2")
numbers = structure.numbers
positions = structure.positions

path = Path(__file__).resolve().parent / "coord"
write.write_turbomole(path, numbers, positions)
