# SPDX-Identifier: CC0-1.0
from pathlib import Path

import tad_mctc as mctc

path = Path(__file__).resolve().parent / "co2.xyz"
numbers, positions = mctc.io.read_xyz_from_path(path)

print(numbers)
print(positions)
