# This file is part of tad-mctc.
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
Regenerate ``test/references/<collection>/<record>.json``, one JSON file per
molecule or periodic test cell. See README.md in this directory for when and
how to run this.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from tad_mctc.data.structures import get_structure
from tad_mctc.io.structure import Structure

HERE = Path(__file__).resolve().parent
# fpm's and Meson's own, already-fixed install paths -- see README.md for
# how to build either one; whichever exists is used.
TOOL_CANDIDATES = [
    HERE / "_install_fpm" / "bin" / "gen_refs_fortran",
    HERE / "_install_meson" / "bin" / "gen_refs_fortran",
]
OUT_DIR = Path(__file__).resolve().parents[2] / "test" / "references"

# Every structure that gets a reference, as a `(collection, record)` pair for
# `get_structure`. The tests load whatever JSON files this writes, so adding
# an entry here and rerunning is all a new reference needs.
SAMPLE_LIST: list[tuple[str, str]] = [
    ("mb16_43", "SiH4"),
    ("heavy28", "pbh4_bih3"),
    ("other", "C6H5I-CH3SH"),
    ("mb16_43", "01"),
    ("mb16_43", "02"),
    ("mb16_43", "03"),
    ("other", "periodic_cubic"),
    ("other", "periodic_triclinic"),
    ("other", "periodic_one_atom"),
    ("other", "diamond"),
    ("other", "nacl"),
    ("x23", "acetic"),
    ("x23", "ammonia"),
    ("x23", "anthracene"),
]


def _sample(entry: tuple[str, str]) -> tuple[Path, Structure]:
    """Resolve one `SAMPLE_LIST` entry to the output path its reference
    JSON is written to, plus the structure itself."""
    collection, record = entry
    out = OUT_DIR / collection / f"{record}.json"
    return out, get_structure(collection, record)


def find_tool() -> Path:
    for candidate in TOOL_CANDIDATES:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "gen_refs_fortran is not built -- see README.md in this directory "
        f"(looked in {', '.join(str(c) for c in TOOL_CANDIDATES)})"
    )


def run_tool(
    tool: Path,
    numbers: list[int],
    positions: list[list[float]],
    lattice: list[list[float]] | None = None,
    periodic: list[bool] | None = None,
) -> Any:
    lines = [str(len(numbers))]
    for z, (x, y, zz) in zip(numbers, positions):
        lines.append(f"{z} {x!r} {y!r} {zz!r}")

    if lattice is None:
        lines.append("0")
    else:
        assert periodic is not None
        lines.append("1")
        for row in lattice:
            lines.append(" ".join(repr(v) for v in row))
        lines.append(" ".join("1" if p else "0" for p in periodic))

    result = subprocess.run(
        [str(tool)],
        input="\n".join(lines),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def main() -> None:
    tool = find_tool()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for entry in SAMPLE_LIST:
        out, sample = _sample(entry)

        if sample.lattice is None:
            data = run_tool(
                tool, sample.numbers.tolist(), sample.positions.tolist()
            )
        else:
            assert sample.periodic is not None
            data = run_tool(
                tool,
                sample.numbers.tolist(),
                sample.positions.tolist(),
                lattice=sample.lattice.tolist(),
                periodic=sample.periodic.tolist(),
            )

        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(data, indent=2) + "\n")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
