# SPDX-Identifier: CC0-1.0
from pathlib import Path

from tad_mctc.io import read

path = Path(__file__).resolve().parent / "h2o.coord"
numbers, positions = read.read_turbomole_from_path(path)

print(numbers)
print(positions)
