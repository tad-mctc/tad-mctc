# SPDX-Identifier: CC0-1.0
from pathlib import Path

from tad_mctc.io import read

path = Path(__file__).resolve().parent / "co2.xyz"
numbers, positions = read.read_xyz(path)

print(numbers)
print(positions)
