# This file is part of tad-mctc.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2024 Grimme Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Packs a `glu_ala_a_0001_to_2048` checkout (the smaller of the two size
ladders at https://www.ergoscf.org/xyz/gluala.php) into the compressed
`src/tad_mctc/data/structures/glu_ala/data.npz` this package ships.

This tool holds parsing/packing logic only. It never contains a structure
itself; every structure it emits is read out of the directory passed on
the command line.

Usage
-----
    python tools/glu_ala/convert.py <path-to-glu_ala_a_0001_to_2048>

Positions are converted from the source xyz files' angstrom to this
library's atomic-unit convention and stored as `float32`: this is a
size-scaling benchmark, not a reference-energy dataset like `mstore`'s,
and the source coordinates carry no more than `float32` worth of
significant digits anyway. Rerunning against the same download produces a
byte-identical file: record order follows the ladder's own filename order,
and there is no other source of nondeterminism.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from tad_mctc.data import pse
from tad_mctc.units import length

OUTPUT_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "tad_mctc"
    / "data"
    / "structures"
    / "glu_ala"
    / "data.npz"
)


def parse_xyz(
    path: Path,
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.float32]]:
    """Atomic numbers (`uint8`) and positions (`float32`, bohr) from one
    xyz file, converted from the source file's angstrom."""
    with open(path, encoding="utf-8") as fh:
        natoms = int(fh.readline().strip())
        fh.readline()  # comment line, unused

        symbols: list[str] = []
        coords: list[tuple[float, float, float]] = []
        for _ in range(natoms):
            symbol, x, y, z = fh.readline().split()[:4]
            symbols.append(symbol.title())
            coords.append((float(x), float(y), float(z)))

    numbers = np.array([pse.S2Z[s] for s in symbols], dtype=np.uint8)
    positions = np.array(coords, dtype=np.float32) * np.float32(length.AA2AU)
    return numbers, positions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "ladder", type=Path, help="path to glu_ala_a_0001_to_2048"
    )
    args = parser.parse_args()

    paths = sorted(args.ladder.glob("*.xyz"))
    if not paths:
        raise SystemExit(f"No .xyz files found in {args.ladder}")

    arrays: dict[str, npt.NDArray[np.uint8] | npt.NDArray[np.float32]] = {}
    for path in paths:
        label = path.stem
        numbers, positions = parse_xyz(path)
        arrays[f"{label}_numbers"] = numbers
        arrays[f"{label}_positions"] = positions

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    # `**arrays`' value type is a union, not `bool`, so neither type checker
    # can rule out an `allow_pickle` collision by itself; there is none. The
    # `cast` documents that instead of an inline ignore comment.
    np.savez_compressed(OUTPUT_PATH, **cast(dict[str, Any], arrays))
    print(f"{len(paths)} records -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
