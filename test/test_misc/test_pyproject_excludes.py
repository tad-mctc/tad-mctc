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
Consistency test guarding against drift between the `[tool.mypy]`/
`[tool.pyright]` `exclude` lists in `pyproject.toml` and the
`[tool.coverage.run]` `omit` list, whose contents are pinned below.

If this test fails, `pyproject.toml`'s mypy/pyright excludes have drifted
from the coverage exclusion set -- update the exclude lists to match.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

try:  # Python >=3.11
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    try:  # optional dependency, not declared for this project
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError:  # pragma: no cover
        tomllib = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parents[2]


def _load_pyproject() -> dict[str, Any]:
    if tomllib is None:  # pragma: no cover
        pytest.skip("neither tomllib nor tomli is available")
    with open(ROOT / "pyproject.toml", "rb") as fh:
        return tomllib.load(fh)


def _sample_path(omit_entry: str) -> str:
    """
    Turn a `[tool.coverage.run].omit` glob entry into a representative
    relative path (no leading "./", no glob) that the mypy/pyright exclude
    settings can be checked against.
    """
    path = omit_entry.removeprefix("./")
    if path.endswith("/*"):
        path = path[: -len("/*")] + "/__dummy__.py"
    return path


def test_coverage_omit_matches_documented_set() -> None:
    """
    Ground truth check: `[tool.coverage.run].omit` must still be exactly
    `autograd/gradcheck.py`, `data/structures/*`, `typing/*`
    and `units/*`. The mypy/pyright tests below compare against this
    list, so pin it down explicitly.
    """
    data = _load_pyproject()
    omit = set(data["tool"]["coverage"]["run"]["omit"])

    documented = {
        "./src/tad_mctc/autograd/gradcheck.py",
        "./src/tad_mctc/data/structures/*",
        "./src/tad_mctc/typing/*",
        "./src/tad_mctc/units/*",
    }
    assert omit == documented


def test_mypy_exclude_covers_documented_paths() -> None:
    """
    `[tool.mypy].exclude` must exclude (at least) everything that
    `[tool.coverage.run].omit` excludes.
    """
    data = _load_pyproject()
    omit = data["tool"]["coverage"]["run"]["omit"]
    pattern = re.compile(data["tool"]["mypy"]["exclude"])

    for entry in omit:
        sample = _sample_path(entry)
        assert pattern.match(sample), (
            f"[tool.mypy].exclude does not cover {sample!r} "
            f"(derived from coverage omit entry {entry!r})"
        )


def test_pyright_exclude_covers_documented_paths() -> None:
    """
    `[tool.pyright].exclude` must exclude (at least) everything that
    `[tool.coverage.run].omit` excludes.
    """
    data = _load_pyproject()
    omit = data["tool"]["coverage"]["run"]["omit"]
    exclude = data["tool"]["pyright"]["exclude"]

    for entry in omit:
        sample = _sample_path(entry)
        assert any(
            sample == ex or sample.startswith(ex.rstrip("/") + "/")
            for ex in exclude
        ), (
            f"[tool.pyright].exclude does not cover {sample!r} "
            f"(derived from coverage omit entry {entry!r})"
        )
