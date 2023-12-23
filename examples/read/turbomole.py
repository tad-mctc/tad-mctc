# SPDX-Identifier: CC0-1.0
from pathlib import Path

import tad_mctc as mctc

path = Path(__file__).resolve().parent / "h2o.coord"
numbers, positions = mctc.io.read_turbomole_from_path(path)

print(numbers)
print(positions)
