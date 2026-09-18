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
Converts an mstore (https://github.com/grimme-lab/mstore) checkout's Fortran
dataset files into the ``dict[str, dict[str, Tensor]]`` modules under
``src/tad_mctc/data/structures/mstore/``.

This tool holds parsing and code-generation logic only. It never contains
structures itself; every structure it emits is read out of the mstore
checkout passed on the command line.

Usage
-----
    python tools/mstore/convert.py <path-to-mstore-checkout>

Writes one Python module per dataset into
``src/tad_mctc/data/structures/mstore/``, then formats them with ``black``
and ``isort`` (both already project dependencies) so the generated files
match the rest of the repository's style. Rerunning on the same checkout
produces no diff: record order follows mstore's own registration order in
each dataset file, and there is no other source of nondeterminism.

Each dataset file is parsed independently, and any record it cannot parse
fully -- an element count mismatch, an unknown element symbol, or an
unrecognized ``call new(...)`` shape -- raises immediately. A record is
never silently skipped or approximated.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "src" / "tad_mctc" / "data" / "structures" / "mstore"

sys.path.insert(0, str(REPO_ROOT / "src"))
from tad_mctc.data.pse import S2Z  # noqa: E402

# Dataset name -> (Fortran source file, registration subroutine). The name
# is both the generated module's file stem and its dict key in `datasets`,
# matching the four datasets already mirrored before this tool existed
# (`mb16_43`, `heavy28`, `amino20x4`, `x23`).
DATASETS: dict[str, tuple[str, str]] = {
    "amino20x4": ("amino20x4.f90", "get_amino20x4_records"),
    "amylose": ("amylose.f90", "get_amylose_records"),
    "but14diol": ("but14diol.f90", "get_but14diol_records"),
    "f_block": ("f_block.f90", "get_f_block_records"),
    "heavy28": ("heavy28.f90", "get_heavy28_records"),
    "ice10": ("ice10.f90", "get_ice10_records"),
    "il16": ("il16.f90", "get_il16_records"),
    "mb16_43": ("mb16_43.f90", "get_mb16_43_records"),
    "polyalanine": ("polyalanine.f90", "get_polyalanine_records"),
    "rc21": ("rc21.f90", "get_rc21_records"),
    "upu23": ("upu23.f90", "get_upu23_records"),
    "x23": ("x23.f90", "get_x23_records"),
}


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------


@dataclass
class ParsedRecord:
    """One mstore record, in tad-mctc's conventions (row-vector lattice,
    Bohr coordinates, atomic numbers)."""

    id: str
    symbols: list[str] | None
    """Element symbols, if the source used ``sym(nat)``. ``None`` for the
    rare record that gives atomic numbers directly via ``num(nat)``."""
    numbers: list[int]
    positions: list[tuple[float, float, float]]
    lattice: list[tuple[float, float, float]] | None
    charge: float | None
    uhf: int | None


def _find_balanced_bracket(text: str, start: int) -> str:
    """Return the content between the first ``[`` at or after `start` and
    its matching ``]``. Safe here because mstore's numeric and string array
    literals never nest square brackets."""
    open_idx = text.index("[", start)
    close_idx = text.index("]", open_idx)
    return text[open_idx + 1 : close_idx]


def _registration_order(
    module_text: str, getter_name: str
) -> list[tuple[str, str]]:
    """Return ``[(record_id, subroutine_name), ...]`` in the order mstore
    itself registers them via ``new_record(...)``."""
    pattern = re.compile(
        rf"subroutine\s+{re.escape(getter_name)}\s*\(records\)(.*?)"
        rf"end\s+subroutine\s+{re.escape(getter_name)}",
        re.DOTALL,
    )
    match = pattern.search(module_text)
    if match is None:
        raise ValueError(f"registration subroutine {getter_name} not found")

    # mstore mixes single- and double-quoted record ids across files.
    pairs = re.findall(
        r"""new_record\(\s*['"]([^'"]+)['"]\s*,\s*(\w+)\s*\)""", match.group(1)
    )
    if not pairs:
        raise ValueError(f"no new_record(...) entries found in {getter_name}")
    return pairs


def _subroutine_body(module_text: str, name: str) -> str:
    pattern = re.compile(
        rf"\bsubroutine\s+{re.escape(name)}\s*\(self\)(.*?)"
        rf"end\s+subroutine\s+{re.escape(name)}\b",
        re.DOTALL,
    )
    match = pattern.search(module_text)
    if match is None:
        raise ValueError(f"subroutine {name} not found")
    return match.group(1)


def _parse_floats(fragment: str) -> list[float]:
    """Extract every ``_wp``-suffixed Fortran real literal, in order."""
    continuation_free = fragment.replace("&", " ")
    literals = re.findall(
        r"[-+]?\d+\.\d+(?:[eEdD][-+]?\d+)?_wp", continuation_free
    )
    return [float(literal.split("_wp")[0]) for literal in literals]


def _parse_nat(body: str) -> int:
    match = re.search(r"\bnat\s*=\s*(\d+)", body)
    if match is None:
        raise ValueError("nat not found")
    return int(match.group(1))


def _parse_symbols(body: str) -> list[str] | None:
    match = re.search(r"sym\(nat\)\s*=\s*\[\s*character\(len=\d+\)\s*::", body)
    if match is None:
        return None
    content = _find_balanced_bracket(body, match.start())
    return re.findall(r'"([^"]*)"', content)


def _parse_numbers_literal(body: str) -> list[int] | None:
    """The one record (f_block's ``Fr_to_Lr``) that gives atomic numbers
    directly via ``num(nat)`` instead of element symbols."""
    match = re.search(r"num\(nat\)\s*=\s*\[", body)
    if match is None:
        return None
    content = _find_balanced_bracket(body, match.start())
    continuation_free = content.replace("&", " ")
    return [int(value) for value in re.findall(r"-?\d+", continuation_free)]


def _parse_positions(body: str, nat: int) -> list[tuple[float, float, float]]:
    match = re.search(r"xyz\(3,\s*nat\)\s*=\s*reshape\(\s*\[", body)
    if match is None:
        raise ValueError("xyz(...) = reshape([...]) not found")
    content = _find_balanced_bracket(body, match.start())
    values = _parse_floats(content)
    if len(values) != 3 * nat:
        raise ValueError(f"xyz has {len(values)} floats, expected {3 * nat}")
    return [
        (values[i], values[i + 1], values[i + 2])
        for i in range(0, len(values), 3)
    ]


def _parse_lattice(body: str) -> list[tuple[float, float, float]] | None:
    match = re.search(r"lattice\(3,\s*3\)\s*=\s*reshape\(\s*\[", body)
    if match is None:
        return None
    content = _find_balanced_bracket(body, match.start())
    values = _parse_floats(content)
    if len(values) != 9:
        raise ValueError(f"lattice has {len(values)} floats, expected 9")
    # mstore's reshape into a (3, 3) array fills column-by-column, and its
    # columns are the lattice vectors, so each consecutive triple of the
    # flat literal *is* one lattice vector already -- no transpose needed
    # to reach tad-mctc's row convention. Verified against the `anthracene`
    # record, already mirrored and independently checked in a prior session.
    return [(values[i], values[i + 1], values[i + 2]) for i in range(0, 9, 3)]


def _parse_charge(body: str) -> float | None:
    match = re.search(r"parameter\s*::\s*charge\s*=\s*([-+]?[\d.]+)", body)
    return None if match is None else float(match.group(1))


def _parse_uhf(body: str) -> int | None:
    match = re.search(r"parameter\s*::\s*uhf\s*=\s*(-?\d+)", body)
    return None if match is None else int(match.group(1))


def parse_dataset(path: Path, getter_name: str) -> list[ParsedRecord]:
    text = path.read_text()
    records = []
    for record_id, subroutine_name in _registration_order(text, getter_name):
        body = _subroutine_body(text, subroutine_name)
        nat = _parse_nat(body)
        symbols = _parse_symbols(body)
        numbers_literal = _parse_numbers_literal(body)

        if (symbols is None) == (numbers_literal is None):
            raise ValueError(
                f"{path.name}:{record_id}: expected exactly one of "
                f"sym(nat)/num(nat), found "
                f"sym={symbols is not None} num={numbers_literal is not None}"
            )

        if symbols is not None:
            if len(symbols) != nat:
                raise ValueError(
                    f"{path.name}:{record_id}: sym has {len(symbols)} "
                    f"entries, expected {nat}"
                )
            unknown = [s for s in symbols if s.title() not in S2Z]
            if unknown:
                raise ValueError(
                    f"{path.name}:{record_id}: unknown element symbol(s) "
                    f"{unknown!r}"
                )
            numbers = [S2Z[s.title()] for s in symbols]
        else:
            assert numbers_literal is not None
            if len(numbers_literal) != nat:
                raise ValueError(
                    f"{path.name}:{record_id}: num has "
                    f"{len(numbers_literal)} entries, expected {nat}"
                )
            numbers = numbers_literal

        records.append(
            ParsedRecord(
                id=record_id,
                symbols=symbols,
                numbers=numbers,
                positions=_parse_positions(body, nat),
                lattice=_parse_lattice(body),
                charge=_parse_charge(body),
                uhf=_parse_uhf(body),
            )
        )
    return records


# --------------------------------------------------------------------------
# Code generation
# --------------------------------------------------------------------------


def _format_aligned_floats(values: list[float]) -> list[str]:
    """
    Render floats with an explicit sign and a common integer/fractional
    width, so a block of them lines up column-by-column in a monospaced
    font -- the decimal points included, not just the sign character.

    Each value is first rendered with ``repr``, the shortest decimal that
    round-trips back to the exact same double, then expanded to plain
    fixed-point notation via :class:`decimal.Decimal` -- required because
    ``repr`` switches to scientific notation for anything with magnitude
    below ``1e-4`` (some mstore coordinates sit that close to zero), and
    naively splitting a string like ``"2.0988889e-07"`` on ``"."`` would
    silently corrupt it (the ``"e-07"`` ends up glued onto the fractional
    digits as if it were part of the mantissa). ``Decimal``'s ``"f"``
    format only repositions the decimal point, it never rounds, so this
    stays exact. Alignment itself then only ever *adds* characters (a
    leading zero on the integer part, a trailing zero on the fractional
    part, a ``+`` where ``repr`` would omit it), never rounds or truncates
    a significant digit, so every returned string parses back to the exact
    input float.

    The integer part is padded with leading *zeros*, not spaces, even
    though the visual effect is the same: this converter's own pipeline
    (see module docstring) pipes its output through ``black``, which
    strips insignificant whitespace inside a list literal -- confirmed
    directly by probing it -- and would silently undo space-padded
    alignment on the very next step. A leading zero is a literal digit of
    the numeral and survives ``black`` unchanged.
    """
    parsed: list[tuple[str, str, str]] = []
    for value in values:
        text = format(Decimal(repr(float(value))), "f")
        sign, digits = ("-", text[1:]) if text.startswith("-") else ("+", text)
        integer_part, _, fractional_part = digits.partition(".")
        parsed.append((sign, integer_part, fractional_part))

    integer_width = max(len(integer_part) for _, integer_part, _ in parsed)
    fractional_width = max(
        len(fractional_part) for _, _, fractional_part in parsed
    )

    return [
        f"{sign}{integer_part.zfill(integer_width)}."
        f"{fractional_part.ljust(fractional_width, '0')}"
        for sign, integer_part, fractional_part in parsed
    ]


def _render_float_matrix(rows: Sequence[tuple[float, ...]]) -> str:
    row_length = len(rows[0])
    flat_values = [value for row in rows for value in row]
    formatted_values = _format_aligned_floats(flat_values)
    formatted_rows = [
        formatted_values[start : start + row_length]
        for start in range(0, len(formatted_values), row_length)
    ]

    row_text = ",\n".join(
        "        [" + ", ".join(row) + "]" for row in formatted_rows
    )
    return "[\n" + row_text + ",\n    ]"


def _render_numbers(record: ParsedRecord) -> str:
    if record.symbols is not None:
        # A single space-joined string, split at call time, matches the
        # style already established in the hand-written mb16_43.py entries:
        # one flowing line per record's element sequence, rather than one
        # array entry per line as black would lay out an explicit list.
        symbol_line = " ".join(record.symbols)
        return f'symbol_to_number("{symbol_line}".split())'
    return f"torch.tensor([{', '.join(str(z) for z in record.numbers)}])"


def _render_record(record: ParsedRecord) -> str:
    lines = [f'    "{record.id}": {{']
    lines.append(f'        "numbers": {_render_numbers(record)},')
    lines.append(
        '        "positions": torch.tensor(\n'
        f"            {_render_float_matrix(record.positions)},\n"
        "            dtype=torch.double,\n"
        "        ),"
    )
    if record.lattice is not None:
        lines.append(
            '        "lattice": torch.tensor(\n'
            f"            {_render_float_matrix(record.lattice)},\n"
            "            dtype=torch.double,\n"
            "        ),"
        )
        lines.append(
            '        "periodic": torch.tensor(\n'
            "            [True, True, True], dtype=torch.bool\n"
            "        ),"
        )
    if record.charge is not None and record.charge != 0.0:
        lines.append(f'        "charge": torch.tensor({int(record.charge)}),')
    if record.uhf is not None and record.uhf != 0:
        lines.append(f'        "uhf": torch.tensor({record.uhf}),')
    lines.append("    },")
    return "\n".join(lines)


def render_module(
    name: str, source_file: str, commit: str, records: list[ParsedRecord]
) -> str:
    n_periodic = sum(1 for r in records if r.lattice is not None)
    n_charged = sum(1 for r in records if r.charge not in (None, 0.0))
    n_open_shell = sum(1 for r in records if r.uhf not in (None, 0))

    title = f"Data: mstore - {name}"
    provenance = (
        f"Generated by ``tools/mstore/convert.py`` from mstore's "
        f"``{source_file}`` at commit ``{commit}``.\n"
        f"See https://github.com/grimme-lab/mstore."
    )
    counts = textwrap.fill(
        f"{len(records)} records, {n_periodic} periodic, {n_charged} "
        f"charged, {n_open_shell} open-shell. Do not edit by hand; rerun "
        f"the converter instead.",
        width=79,
    )
    summary = f"{provenance}\n\n{counts}"
    header = f'''# This file is part of tad-mctc.
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
{title}
{"=" * len(title)}

{summary}
"""

from __future__ import annotations

import torch

from ....convert import symbol_to_number

__all__ = ["{name}"]


{name}: dict[str, dict[str, torch.Tensor]] = {{
'''
    body = "\n".join(_render_record(r) for r in records)
    return header + body + "\n}\n"


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkout", type=Path, help="path to an mstore checkout"
    )
    args = parser.parse_args()

    commit = subprocess.run(
        ["git", "-C", str(args.checkout), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    written = []
    for name, (source_file, getter_name) in DATASETS.items():
        source_path = args.checkout / "src" / "mstore" / source_file
        records = parse_dataset(source_path, getter_name)
        module_text = render_module(name, source_file, commit, records)
        output_path = OUTPUT_DIR / f"{name}.py"
        output_path.write_text(module_text)
        written.append(output_path)
        print(f"{name}: {len(records)} records -> {output_path}")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "isort",
            "--profile",
            "black",
            "--line-length",
            "80",
        ]
        + [str(p) for p in written],
        check=True,
    )
    subprocess.run(
        [sys.executable, "-m", "black", "--line-length", "80"]
        + [str(p) for p in written],
        check=True,
    )


if __name__ == "__main__":
    main()
