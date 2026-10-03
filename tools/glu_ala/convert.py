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
ladders at https://www.ergoscf.org/xyz/gluala.php), plus a handful of the
larger `glu_ala_b_512_to_65536` structures named in `EXTRA_LADDER_B_LABELS`
below, into the compressed
`src/tad_mctc/data/structures/glu_ala/data.npz` this package ships.

This tool holds parsing/packing logic only. It never contains a structure
itself; every structure it emits is read out of the directories passed on
the command line.

Usage
-----
    python tools/glu_ala/convert.py <path-to-glu_ala_a_0001_to_2048> \\
        [path-to-glu_ala_b_512_to_65536]

The second, optional path packs `EXTRA_LADDER_B_LABELS` on top of the full
`a` ladder; omitting it packs `a` alone, same as before this option
existed.

Positions are converted from the source xyz files' angstrom to this
library's atomic-unit convention and stored as `float32`: this is a
size-scaling benchmark, not a reference-energy dataset like `mstore`'s,
and the source coordinates carry no more than `float32` worth of
significant digits anyway. The archive is LZMA- rather than the more
common deflate-compressed (see `_write_npz_lzma`): deflate barely
compresses float32 mantissas, and LZMA's larger window does noticeably
better on the same, otherwise-incompressible coordinates. Rerunning
against the same downloads produces a byte-identical file: record order
follows each ladder's own filename order, ladder `a` first, and there is
no other source of nondeterminism.
"""

from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path
from typing import Mapping

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

# `glu_ala_b`'s labels at or below ladder `a`'s own maximum (53,250 atoms,
# label "2048") are identical structures under a different filename, not
# distinct data (see `examples/scaling/glu_ala.py`'s `discover_structures`,
# which already dedupes on this). Only genuinely larger labels are worth
# packing at all. Of those, "4096" (106,498 atoms) and "8192" (212,994
# atoms) keep the packaged file to 7.4 MB (measured with `_write_npz_lzma`
# below); adding "16384" (425,986 atoms) would push it to 10.7 MB. This is
# the one and only list of which extra `glu_ala_b` structures get packaged
# -- raise the package-size budget before extending it, not the other way
# around.
EXTRA_LADDER_B_LABELS = ["4096", "8192"]


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


def _write_npz_lzma(
    path: Path,
    arrays: Mapping[str, npt.NDArray[np.uint8] | npt.NDArray[np.float32]],
) -> None:
    """Write an `.npz` archive exactly like `np.savez_compressed`, except
    each array is LZMA-compressed rather than the hardcoded deflate:
    `np.savez_compressed` offers no algorithm choice, and deflate barely
    compresses `float32` mantissas (measured: 8.48 MB). LZMA's larger,
    context-mixing window does noticeably better on the same, otherwise
    close-to-incompressible coordinates (7.41 MB). The format read back
    by `np.load` is unaffected either way: `zipfile.ZIP_LZMA` is a
    standard per-entry compression method, same lazy, per-array loading
    as before."""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_LZMA) as zf:
        for name, arr in arrays.items():
            buf = io.BytesIO()
            np.lib.format.write_array(buf, arr, allow_pickle=False)
            zf.writestr(f"{name}.npy", buf.getvalue())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "ladder_a", type=Path, help="path to glu_ala_a_0001_to_2048"
    )
    parser.add_argument(
        "ladder_b",
        type=Path,
        nargs="?",
        default=None,
        help="path to glu_ala_b_512_to_65536 (optional)",
    )
    args = parser.parse_args()

    paths = sorted(args.ladder_a.glob("*.xyz"))
    if not paths:
        raise SystemExit(f"No .xyz files found in {args.ladder_a}")

    if args.ladder_b is not None:
        for label in EXTRA_LADDER_B_LABELS:
            path = args.ladder_b / f"{label}.xyz"
            if not path.is_file():
                raise SystemExit(f"Expected {path} to exist")
            paths.append(path)

    arrays: dict[str, npt.NDArray[np.uint8] | npt.NDArray[np.float32]] = {}
    for path in paths:
        label = path.stem
        numbers, positions = parse_xyz(path)
        arrays[f"{label}_numbers"] = numbers
        arrays[f"{label}_positions"] = positions

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _write_npz_lzma(OUTPUT_PATH, arrays)
    print(f"{len(paths)} records -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
