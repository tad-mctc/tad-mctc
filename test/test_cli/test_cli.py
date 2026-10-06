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

import argparse
from pathlib import Path

import pytest
import torch

from tad_mctc.cli import _timing, main
from tad_mctc.cli._args import CN_MODELS
from tad_mctc.cli._main import _build_neighborlist, _coordination_number
from tad_mctc.cli._output import print_native_build, print_system_info
from tad_mctc.cli._timing import Timings
from tad_mctc.exceptions import StructureWarning
from tad_mctc.io.structure import Structure
from tad_mctc.neighbor import _native
from tad_mctc.neighbor.list import build_neighborlist
from tad_mctc.typing import DD

from ..conftest import DEVICE
from ..utils import load_structure

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

    # the lone atom sits in the origin, which the padding check flags
    with pytest.warns(StructureWarning, match="padding value"):
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


def test_non_integer_omp_is_a_usage_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    structure = tmp_path / "water.xyz"
    structure.write_text(_WATER)

    with pytest.raises(SystemExit) as exc:
        main(["--omp", "many", str(structure)])

    assert exc.value.code == 2
    assert "invalid int value: 'many'" in capsys.readouterr().err


def test_nlist_only_cannot_be_combined_with_a_dense_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    structure = tmp_path / "water.xyz"
    structure.write_text(_WATER)

    with pytest.raises(SystemExit) as exc:
        main(["--nlist-only", "--neighbor", "dense", str(structure)])

    assert exc.value.code == 2
    assert "cannot be combined" in capsys.readouterr().err


def test_cuda_without_a_cuda_device_is_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    structure = tmp_path / "water.xyz"
    structure.write_text(_WATER)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(SystemExit, match="no CUDA device"):
        main(["--cuda", str(structure)])


def test_cuda_run_moves_the_structure_to_the_device(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The move is its own timed step. There is no device here, so the
    move is recorded and left undone."""
    structure = tmp_path / "water.xyz"
    structure.write_text(_WATER)
    moved_to: list[torch.device] = []

    def record_move(self: Structure, device: torch.device) -> Structure:
        moved_to.append(device)
        return self

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(Structure, "to", record_move)

    argv = ["--timing", "--cuda", "--neighbor", "dense", str(structure)]
    assert main(argv) == 0

    assert moved_to == [torch.device("cuda")]
    assert "move to device" in capsys.readouterr().out.split("Timing")[-1]


@pytest.mark.parametrize("nlist_only", [False, True])
def test_sparse_run_without_the_native_extension_reports_it(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    nlist_only: bool,
) -> None:
    """With the extension disabled, the sparse run goes through the
    pure-Python search and says so."""
    structure = tmp_path / "water.xyz"
    structure.write_text(_WATER)
    monkeypatch.setenv("TAD_MCTC_DISABLE_NATIVE", "1")
    _native._load_with_info.cache_clear()

    argv = ["--nlist-only"] if nlist_only else []
    try:
        assert main([*argv, str(structure)]) == 0
    finally:
        _native._load_with_info.cache_clear()
    out = capsys.readouterr().out

    assert "disabled by TAD_MCTC_DISABLE_NATIVE" in out
    assert ("Neighbour list" in out) == nlist_only
    assert ("Results" in out) != nlist_only


def test_dense_run_of_a_cell_builds_its_periodic_shifts(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A periodic structure is summed over its images through shifts built
    for the model's cutoff, as its own timed step."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    structure = load_structure("other", "periodic_triclinic", dd)
    args = argparse.Namespace(
        cn="d3", neighbor="dense", mode="graph", compile=False
    )
    cn = _coordination_number(
        args, CN_MODELS["d3"], structure, Timings(enabled=True)
    )

    assert "build periodic shifts" in capsys.readouterr().out
    assert cn.shape == structure.numbers.shape
    assert bool(torch.isfinite(cn).all())


def test_system_info_leaves_out_charge_and_uhf_if_unset(
    capsys: pytest.CaptureFixture[str],
) -> None:
    numbers = torch.tensor([2])
    structure = Structure(numbers=numbers, positions=torch.zeros(1, 3))

    print_system_info("he.xyz", structure)
    out = capsys.readouterr().out

    assert "charge" not in out
    assert "uhf" not in out
    assert "periodic  no" in out


def test_system_info_lists_charge_and_uhf_of_every_frame(
    capsys: pytest.CaptureFixture[str],
) -> None:
    structure = Structure(
        numbers=torch.tensor([[2, 0], [1, 1]]),
        positions=torch.zeros(2, 2, 3),
        charge=torch.tensor([0.0, 1.0]),
        uhf=torch.tensor([0.0, 2.0]),
    )

    print_system_info("frames.xyz", structure)
    out = capsys.readouterr().out

    assert "charge    0 1" in out
    assert "uhf       0 2" in out


@pytest.mark.parametrize(
    ("info", "expected", "absent"),
    [
        (
            _native.BuildInfo("precompiled", "/lib/ext.so"),
            ["precompiled (TAD_MCTC_BUILD_NATIVE=1 install)", "/lib/ext.so"],
            ["compiler", "flags", "ninja", "error"],
        ),
        (
            _native.BuildInfo("disabled"),
            ["disabled by TAD_MCTC_DISABLE_NATIVE, pure Python"],
            ["library", "compiler", "ninja"],
        ),
        (
            _native.BuildInfo("unavailable", error="no compiler"),
            ["failed to load or compile, pure Python", "error     no compiler"],
            ["library", "ninja"],
        ),
        (
            _native.BuildInfo(
                "jit",
                "/lib/ext.so",
                compiled_now=True,
                ninja="/bin/ninja (1.11)",
                cflags=("-O3", "-fno-fast-math"),
                compiler="/bin/c++ (gcc 13)",
            ),
            [
                "compiled on first use (JIT), compiled in this run",
                "compiler  /bin/c++ (gcc 13)",
                "flags     -O3 -fno-fast-math",
                "ninja     /bin/ninja (1.11)",
            ],
            ["error"],
        ),
        (
            _native.BuildInfo("jit", "/lib/ext.so"),
            ["compiled on first use (JIT), cached build", "not found"],
            ["compiler", "flags"],
        ),
    ],
    ids=[
        "precompiled",
        "disabled",
        "unavailable",
        "jit-compiled",
        "jit-cached",
    ],
)
def test_native_build_report(
    capsys: pytest.CaptureFixture[str],
    info: _native.BuildInfo,
    expected: list[str],
    absent: list[str],
) -> None:
    print_native_build(info)
    out = capsys.readouterr().out

    for text in expected:
        assert text in out
    for text in absent:
        assert text not in out.replace("Native extension", "")


def test_timings_wait_for_queued_cuda_kernels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Kernels launch asynchronously, so each step is closed only after
    the device is done, if CUDA is in use."""
    calls: list[str] = []
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: calls.append("sync"))

    _timing._synchronize_cuda()

    assert calls == ["sync"]


def test_timings_skip_synchronize_without_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: calls.append("sync"))

    _timing._synchronize_cuda()

    assert calls == []


def test_cuda_neighbour_list_does_not_report_the_native_extension(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The search on a CUDA device does not run through the CPU
    extension, so there is nothing to report about it."""
    structure = tmp_path / "water.xyz"
    structure.write_text(_WATER)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(Structure, "to", lambda self, device: self)

    assert main(["--cuda", "--nlist-only", str(structure)]) == 0
    out = capsys.readouterr().out

    assert "Neighbour list" in out
    assert "Native extension" not in out


def test_compiled_run_matches_the_eager_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--compile`` times the compile as its own step and reports the
    same coordination numbers as the eager run."""
    structure = tmp_path / "water.xyz"
    structure.write_text(_WATER)

    assert main(["--neighbor", "dense", str(structure)]) == 0
    eager = capsys.readouterr().out.split("Results")[-1]

    argv = ["--timing", "--compile", "--neighbor", "dense", str(structure)]
    assert main(argv) == 0
    out = capsys.readouterr().out

    assert "torch.compile (cn_d3 (dense))" in out
    assert "cn_d3 (dense), compiled" in out
    assert out.split("Results")[-1].split("Timing")[0].strip() == (
        eager.strip()
    )


def test_compile_cannot_be_combined_with_recompute_mode(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    structure = tmp_path / "water.xyz"
    structure.write_text(_WATER)

    with pytest.raises(SystemExit) as exc:
        main(["--compile", "--mode", "recompute", str(structure)])

    assert exc.value.code == 2
    assert "--compile needs '--mode graph'" in capsys.readouterr().err
