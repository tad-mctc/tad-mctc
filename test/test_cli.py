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
Test the command line tool (`tad_mctc.cli`): the native extension's build
info and timing step, and the timed neighbour-list build.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tad_mctc.cli import main
from tad_mctc.cli._main import _build_neighborlist
from tad_mctc.cli._timing import Timings
from tad_mctc.io.structure import Structure
from tad_mctc.neighbor.list import build_neighborlist
from tad_mctc.typing import DD

from .conftest import DEVICE
from .utils import load_structure

_WATER = """3
water
O  0.000  0.000  0.000
H  0.757  0.586  0.000
H -0.757  0.586  0.000
"""


@pytest.mark.native
def test_sparse_run_times_and_reports_the_native_extension(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Loading the extension, which may compile it, is its own timed step,
    and the run reports how the extension was obtained."""
    structure = tmp_path / "water.xyz"
    structure.write_text(_WATER)

    assert main(["--timing", "--nlist-only", str(structure)]) == 0
    out = capsys.readouterr().out

    timing_table = out.split("Timing")[-1]
    assert "native extension" in timing_table
    assert "Native extension" in out
    assert "library" in out


def test_dense_run_does_not_load_the_native_extension(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The dense path builds no neighbour list, so it neither loads the
    extension nor reports it."""
    structure = tmp_path / "water.xyz"
    structure.write_text(_WATER)

    assert main(["--timing", "--neighbor", "dense", str(structure)]) == 0
    out = capsys.readouterr().out

    assert "native extension" not in out
    assert "Native extension" not in out


def test_coldfusion_check_of_a_small_molecule_does_not_load_the_native_extension(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A small molecule is checked by comparing all its pairs, so a dense
    run with the check still leaves the extension alone. The check is its
    own timed step."""
    structure = tmp_path / "water.xyz"
    structure.write_text(_WATER)

    argv = ["--timing", "--neighbor", "dense", "--coldfusion-check"]
    assert main([*argv, str(structure)]) == 0
    out = capsys.readouterr().out

    assert "cold-fusion check" in out.split("Timing")[-1]
    assert "native extension" not in out
    assert "Native extension" not in out


def _assert_cli_list_matches_library(
    structure: Structure, capsys: pytest.CaptureFixture[str]
) -> None:
    """The list the timed build returns is the library's own list, and
    every step of the build was timed."""
    cutoff = 8.0
    timings = Timings(enabled=True)

    got = _build_neighborlist(structure, cutoff, timings)
    want = build_neighborlist(structure, cutoff)

    assert torch.equal(got.idx_i, want.idx_i)
    assert torch.equal(got.idx_j, want.idx_j)
    assert torch.equal(got.shift, want.shift)
    assert torch.equal(got.mask, want.mask)

    out = capsys.readouterr().out
    assert "nlist: tiles" in out
    assert "nlist: pair filter" in out


def test_timed_build_matches_the_library_for_a_molecule(
    capsys: pytest.CaptureFixture[str],
) -> None:
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    structure = load_structure("mb16_43", "01", dd)
    _assert_cli_list_matches_library(structure, capsys)


def test_timed_build_matches_the_library_for_a_cell(
    capsys: pytest.CaptureFixture[str],
) -> None:
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    structure = load_structure("other", "periodic_triclinic", dd)
    _assert_cli_list_matches_library(structure, capsys)


_WATER_AND_HYDROGEN = """3
water
O  0.000  0.000  0.000
H  0.757  0.586  0.000
H -0.757  0.586  0.000
2
hydrogen
H  0.000  0.000  0.000
H  0.000  0.000  0.740
"""


def test_multi_frame_file_reports_real_atoms_of_all_frames(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A multi-frame file is read as a padded batch. The summary counts
    and names only real atoms, and the statistics leave out the zero CN
    of the hydrogen frame's padding atom."""
    structure = tmp_path / "frames.xyz"
    structure.write_text(_WATER_AND_HYDROGEN)

    assert main(["--neighbor", "dense", str(structure)]) == 0
    out = capsys.readouterr().out

    assert "frames    2" in out
    assert "atoms     5" in out
    assert "elements  H4O" in out
    assert "(frame 1, atom 1 O)" in out.split("max")[-1]
    min_cn = float(out.split("min")[-1].split()[0])
    assert min_cn > 0.5


def test_multi_frame_file_with_the_neighbour_list_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--nlist-only` prints the same summary of a batch."""
    structure = tmp_path / "frames.xyz"
    structure.write_text(_WATER_AND_HYDROGEN)

    assert main(["--nlist-only", str(structure)]) == 0
    out = capsys.readouterr().out

    assert "frames    2" in out
    assert "atoms     5" in out
    assert "Neighbour list" in out


def test_omp_setting_is_restored_after_the_run(tmp_path: Path) -> None:
    """`--omp` applies to the run only, so calling `main` from Python
    leaves the caller's thread count alone."""
    structure = tmp_path / "water.xyz"
    structure.write_text(_WATER)
    before = torch.get_num_threads()
    during = str(before + 1)

    assert main(["--omp", during, "--neighbor", "dense", str(structure)]) == 0

    assert torch.get_num_threads() == before


def test_single_atom_has_a_finite_std(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CN statistics of a one-atom structure report a zero spread,
    not the `nan` of a sample std over one value."""
    structure = tmp_path / "he.xyz"
    structure.write_text("1\nhelium\nHe  0.000  0.000  0.000\n")

    assert main(["--neighbor", "dense", str(structure)]) == 0
    out = capsys.readouterr().out

    assert float(out.split("std")[-1].split()[0]) == 0.0


@pytest.mark.parametrize("n_threads", ["0", "-1"])
def test_non_positive_omp_is_a_usage_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], n_threads: str
) -> None:
    """A thread count below one is rejected by the parser, before torch
    ever sees it."""
    structure = tmp_path / "water.xyz"
    structure.write_text(_WATER)

    with pytest.raises(SystemExit) as exc:
        main(["--omp", n_threads, str(structure)])

    assert exc.value.code == 2
    assert "--omp: must be at least 1" in capsys.readouterr().err
