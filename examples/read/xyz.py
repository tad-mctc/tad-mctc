# SPDX-Identifier: CC0-1.0
"""Reading atomic numbers and positions from an XYZ file."""

from pathlib import Path

from tad_mctc.io import read

path = Path(__file__).resolve().parent / "co2.xyz"
structure = read.read_structure(path)

print(structure.numbers)
print(structure.positions)
