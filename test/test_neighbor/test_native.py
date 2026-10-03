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
Test the optional native (OpenMP, CPU-only) neighbour-list filter
(`tad_mctc.neighbor._native`) against the pure-Python path it stands in
for, and that `build_neighborlist` picks it up transparently on CPU where
it is available.

See `tad_mctc.neighbor._native`'s module docstring for the exact-output
contract these tests check: same pairs, same order, for every threshold,
including the truncation path a too-small `capacity` exercises.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import os
import types
import warnings
from typing import Any

import pytest
import torch

from tad_mctc.neighbor import _native, _native_flags
from tad_mctc.neighbor._tiles import Tiles, tile_pairs
from tad_mctc.neighbor.list import (
    _atom_pairs_within_thresholds,
    _pad_to_capacity,
    _Padding,
    build_neighborlist,
)
from tad_mctc.typing import DD

from ..utils import hydrogens


def test_is_available_type() -> None:
    """Whatever the outcome (compiler present or not), `is_available`
    must return a plain `bool`, never raise."""
    assert isinstance(_native.is_available(), bool)


def test_added_flags_are_split_like_a_shell_command_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TAD_MCTC_NATIVE_CFLAGS", "-mavx2  -mfma")
    assert _native_flags.extra_cflags() == ("-mavx2", "-mfma")

    monkeypatch.delenv("TAD_MCTC_NATIVE_CFLAGS")
    assert _native_flags.extra_cflags() == ()


def test_added_flags_cannot_turn_fast_math_or_contraction_back_on() -> None:
    """The kernel relies on `-fno-fast-math -ffp-contract=off`, so they
    must come after the added flags, where the compiler's last-one-wins
    rule keeps them."""
    cflags = _native_flags.cflags(("-march=native", "-ffast-math"))
    assert cflags == (
        "-O3",
        "-march=native",
        "-ffast-math",
        "-fno-fast-math",
        "-ffp-contract=off",
        "-fopenmp",
    )


def test_every_build_configuration_gets_its_own_jit_build() -> None:
    """Separate cache directories, so that processes with a different
    compiler, flags or `ninja` do not recompile over each other. Two
    `ninja` versions do not recognise each other's builds."""
    gcc = "/usr/bin/c++ (c++ 13.3.0)"
    icpx = "/opt/intel/bin/icpx (Intel(R) oneAPI DPC++/C++ Compiler)"
    ninja_1_11 = "/usr/bin/ninja (1.11.1)"
    ninja_1_13 = "/opt/conda/bin/ninja (1.13.0)"
    default = _native_flags.cflags(())
    avx2 = _native_flags.cflags(("-mavx2",))

    names = {
        _native._jit_name(gcc, ninja_1_13, default),
        _native._jit_name(gcc, ninja_1_11, default),
        _native._jit_name(gcc, ninja_1_13, avx2),
        _native._jit_name(icpx, ninja_1_13, default),
        _native._jit_name(None, None, default),
    }

    assert len(names) == 5
    assert _native._jit_name(gcc, ninja_1_13, default) == (
        _native._jit_name(gcc, ninja_1_13, default)
    )


def test_compiler_is_the_one_cxx_names(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CXX", raising=False)
    assert _native._compiler() is None

    monkeypatch.setenv("CXX", "/no/such/compiler")
    assert _native._compiler() == "/no/such/compiler"


def test_only_the_wrong_compiler_warning_is_silenced() -> None:
    """PyTorch's warning about a compiler other than its own is dropped
    while the extension loads, whether PyTorch logs it or warns it; other
    warnings still come through."""
    from torch.utils import cpp_extension

    logger = logging.getLogger(cpp_extension.__name__)

    def log_record(message: str) -> logging.LogRecord:
        return logging.LogRecord(
            logger.name, logging.WARNING, __file__, 0, message, None, None
        )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with _native._without_wrong_compiler_warning():
            warnings.warn("Your compiler (icpx) is not compatible with g++")
            warnings.warn("something else")
            assert logger.filter(log_record("something else"))
            if hasattr(cpp_extension, "WRONG_COMPILER_WARNING"):
                wrong_compiler = cpp_extension.WRONG_COMPILER_WARNING
                assert not logger.filter(log_record(wrong_compiler))

    assert [str(w.message) for w in caught] == ["something else"]


def test_error_summary_prefers_the_compiler_error() -> None:
    error = RuntimeError(
        "Error building extension 'x': [1/2] c++ -mfancy -c x.cpp\n"
        "c++: error: unrecognized command-line option '-mfancy'\n"
        "ninja: build stopped: subcommand failed."
    )
    assert _native._error_summary(error) == (
        "c++: error: unrecognized command-line option '-mfancy'"
    )


@pytest.mark.native
def test_build_info_names_the_loaded_library() -> None:
    """A loaded extension reports where it came from and its library; a
    JIT build also names the `ninja` that decided whether to rebuild."""
    info = _native.build_info()

    assert info.origin in ("precompiled", "jit")
    assert info.library is not None and os.path.isfile(info.library)
    if info.origin == "jit":
        assert info.ninja is not None
        assert info.compiler is not None


@pytest.mark.native
def test_a_precompiled_module_that_does_not_load_is_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pre-built module that exists but does not load, such as one built
    with `icpx` without Intel's runtime libraries on the library path, is
    replaced by the JIT build, with a warning and the reason in the build
    info."""
    find_spec = importlib.util.find_spec
    import_module = importlib.import_module

    def find_a_precompiled_module(name: str, *args: Any) -> Any:
        if name == _native._PRECOMPILED:
            return object()
        return find_spec(name, *args)

    def fail_to_load_it(name: str, *args: Any) -> Any:
        if name == _native._PRECOMPILED:
            raise ImportError("libiomp5.so: cannot open shared object file")
        return import_module(name, *args)

    monkeypatch.setattr(importlib.util, "find_spec", find_a_precompiled_module)
    monkeypatch.setattr(importlib, "import_module", fail_to_load_it)
    _native._load_with_info.cache_clear()
    try:
        with pytest.warns(UserWarning, match="pre-built"):
            info = _native.build_info()
    finally:
        _native._load_with_info.cache_clear()

    assert info.origin == "jit"
    assert info.error == (
        "pre-built module: libiomp5.so: cannot open shared object file"
    )


@pytest.mark.native
def test_a_precompiled_module_of_another_source_is_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pre-built module built from another source than the installed
    one, such as that of an editable install before the source was edited,
    is replaced by the JIT build, with a warning and the reason in the
    build info."""
    find_spec = importlib.util.find_spec
    import_module = importlib.import_module
    outdated = types.SimpleNamespace(
        source_digest="0" * 16, __file__="/outdated/_native_pairs_ext.so"
    )

    def find_a_precompiled_module(name: str, *args: Any) -> Any:
        if name == _native._PRECOMPILED:
            return object()
        return find_spec(name, *args)

    def load_the_outdated_module(name: str, *args: Any) -> Any:
        if name == _native._PRECOMPILED:
            return outdated
        return import_module(name, *args)

    monkeypatch.setattr(importlib.util, "find_spec", find_a_precompiled_module)
    monkeypatch.setattr(importlib, "import_module", load_the_outdated_module)
    _native._load_with_info.cache_clear()
    try:
        with pytest.warns(UserWarning, match="built from another"):
            info = _native.build_info()
    finally:
        _native._load_with_info.cache_clear()

    assert info.origin == "jit"
    assert info.error == (
        "pre-built module: built from another _native_pairs.cpp"
    )


@pytest.mark.native
def test_a_precompiled_module_of_the_installed_source_is_used(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pre-built module built from the installed source is used as is,
    without a JIT build."""
    find_spec = importlib.util.find_spec
    import_module = importlib.import_module
    current = types.SimpleNamespace(
        source_digest=_native._source_digest(),
        __file__="/current/_native_pairs_ext.so",
    )

    def find_a_precompiled_module(name: str, *args: Any) -> Any:
        if name == _native._PRECOMPILED:
            return object()
        return find_spec(name, *args)

    def load_the_current_module(name: str, *args: Any) -> Any:
        if name == _native._PRECOMPILED:
            return current
        return import_module(name, *args)

    monkeypatch.setattr(importlib.util, "find_spec", find_a_precompiled_module)
    monkeypatch.setattr(importlib, "import_module", load_the_current_module)
    _native._load_with_info.cache_clear()
    try:
        module, info = _native._load_with_info()
    finally:
        _native._load_with_info.cache_clear()

    assert module is current
    assert info == _native.BuildInfo(
        "precompiled", "/current/_native_pairs_ext.so"
    )


@pytest.mark.native
def test_the_loaded_module_reports_its_source_digest() -> None:
    """A pre-built module carries the digest of the installed source (it
    would not have been used otherwise); a JIT build, which `setup.py`
    does not configure, carries an empty one."""
    module = _native._load()
    info = _native.build_info()

    if info.origin == "precompiled":
        assert module.source_digest == _native._source_digest()
    else:
        assert module.source_digest == ""


@pytest.mark.native
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("tile", [4, 32, 40, 96])
def test_matches_python_path(dtype: torch.dtype, tile: int) -> None:
    """The native path must return exactly the same `idx_i`/`idx_j`, in
    the same order, as the pure-Python path -- for both floating dtypes
    and both a tile width smaller and larger than a typical system, so
    the same-tile strict-upper-triangle case is exercised either way.
    Rows of tiles wider than 32 atoms span several words of the native
    path's hit masks.

    Compared against the ``"broadcast"`` kernel specifically, not the
    CPU-automatic ``"baddbmm"`` one: ``_baddbmm_distance_squared``'s
    ``|a|^2 + |b|^2 - 2 a.b`` reformulation rounds differently from the
    direct subtraction. Native matches ``"broadcast"`` -- the
    direct-subtraction, dtype- and device-agnostic reference formula --
    exactly; it is not expected to match ``"baddbmm"`` bit-for-bit for a
    pair within rounding of the threshold."""
    torch.manual_seed(0)
    positions = torch.randn(600, 3, dtype=dtype, device="cpu") * 15.0
    cutoff = 4.0

    tiles = Tiles(positions, tile=tile)
    tile_a, tile_b = tile_pairs(tiles, cutoff)

    ((native_i, native_j, _),) = _atom_pairs_within_thresholds(
        tiles, tile_a, tile_b, positions, (cutoff,), pair_filter="native"
    )
    ((python_i, python_j, _),) = _atom_pairs_within_thresholds(
        tiles,
        tile_a,
        tile_b,
        positions,
        (cutoff,),
        distance_kernel="broadcast",
        pair_filter="python",
    )

    assert native_i.shape[0] > 0, "test needs a system with real pairs"
    assert torch.equal(native_i, python_i)
    assert torch.equal(native_j, python_j)


@pytest.mark.native
@pytest.mark.parametrize("tile", [1, 2])
def test_pair_right_at_a_float32_threshold_matches_python_path(
    tile: int,
) -> None:
    """A float32 pair whose squared distance equals the squared threshold
    rounded to float32 -- which is larger than the exact square -- is kept
    by both paths, since both compare in float32. With one atom per tile
    (``tile=1``) it also passes the native path's bounding-box skip, which
    must compare in the same precision as the exact test."""
    cutoff = 1.1
    distance = torch.tensor(cutoff, dtype=torch.float32, device="cpu")
    positions = torch.zeros(2, 3, dtype=torch.float32, device="cpu")
    positions[1, 0] = distance

    distance_squared = float(distance * distance)
    threshold_squared = float(
        torch.tensor(cutoff * cutoff, dtype=torch.float32, device="cpu")
    )
    assert distance_squared == threshold_squared
    assert distance_squared > cutoff * cutoff, "not a boundary case"

    tiles = Tiles(positions, tile=tile)
    tile_a, tile_b = tile_pairs(tiles, cutoff)

    ((native_i, native_j, _),) = _atom_pairs_within_thresholds(
        tiles, tile_a, tile_b, positions, (cutoff,), pair_filter="native"
    )
    ((python_i, python_j, _),) = _atom_pairs_within_thresholds(
        tiles,
        tile_a,
        tile_b,
        positions,
        (cutoff,),
        distance_kernel="broadcast",
        pair_filter="python",
    )

    assert python_i.shape[0] == 1
    assert torch.equal(native_i, python_i)
    assert torch.equal(native_j, python_j)


@pytest.mark.native
def test_matches_python_path_multiple_thresholds() -> None:
    """Two thresholds sharing one candidate scan (as D3's CN and two-body
    cutoffs do via `build_neighborlists`) must each independently match
    the Python ``"broadcast"`` path, not just the largest one."""
    torch.manual_seed(1)
    positions = torch.randn(400, 3, device="cpu") * 12.0
    thresholds = (2.5, 5.0)

    tiles = Tiles(positions, tile=16)
    tile_a, tile_b = tile_pairs(tiles, max(thresholds))

    native_results = _atom_pairs_within_thresholds(
        tiles, tile_a, tile_b, positions, thresholds, pair_filter="native"
    )
    python_results = _atom_pairs_within_thresholds(
        tiles,
        tile_a,
        tile_b,
        positions,
        thresholds,
        distance_kernel="broadcast",
        pair_filter="python",
    )

    for native, python in zip(native_results, python_results):
        assert torch.equal(native.idx_i, python.idx_i)
        assert torch.equal(native.idx_j, python.idx_j)


@pytest.mark.native
def test_matches_python_path_with_anchor_atoms() -> None:
    """With `anchor_atoms`, both paths must drop exactly the pairs without
    an anchor atom and keep the order of the rest."""
    torch.manual_seed(2)
    positions = torch.randn(500, 3, dtype=torch.float64, device="cpu") * 12.0
    anchor_atoms = torch.rand(500, device="cpu") < 0.1
    cutoff = 5.0

    tiles = Tiles(positions, tile=16)
    tile_a, tile_b = tile_pairs(tiles, cutoff)

    ((native_i, native_j, _),) = _atom_pairs_within_thresholds(
        tiles,
        tile_a,
        tile_b,
        positions,
        (cutoff,),
        pair_filter="native",
        anchor_atoms=anchor_atoms,
    )
    ((python_i, python_j, _),) = _atom_pairs_within_thresholds(
        tiles,
        tile_a,
        tile_b,
        positions,
        (cutoff,),
        distance_kernel="broadcast",
        pair_filter="python",
        anchor_atoms=anchor_atoms,
    )
    ((all_i, all_j, _),) = _atom_pairs_within_thresholds(
        tiles, tile_a, tile_b, positions, (cutoff,), pair_filter="native"
    )

    assert torch.equal(native_i, python_i)
    assert torch.equal(native_j, python_j)

    has_anchor = anchor_atoms[all_i] | anchor_atoms[all_j]
    assert 0 < int(has_anchor.sum()) < all_i.shape[0]
    assert torch.equal(native_i, all_i[has_anchor])
    assert torch.equal(native_j, all_j[has_anchor])


@pytest.mark.native
def test_undersized_capacity_truncation_matches() -> None:
    """`_pad_to_capacity` truncates by slicing `[:npair]` when `capacity`
    is too small, which makes pair *order* observable -- build a
    `NeighborList` both ways (`pair_filter="native"` and `"python"`) with
    a deliberately undersized `capacity` and check the truncated, padded
    tensors agree exactly, not just the full untruncated ones.

    Goes through `_atom_pairs_within_thresholds`/`_pad_to_capacity`
    directly rather than the public `build_neighborlist`: `pair_filter`
    is deliberately not exposed on the public builders (see its
    docstring), so forcing one implementation from a test means calling
    the private functions it wraps."""
    torch.manual_seed(2)
    positions = torch.randn(300, 3, device="cpu") * 12.0
    cutoff = 4.0
    nat = positions.shape[0]
    dd: DD = {"device": positions.device, "dtype": positions.dtype}

    full = build_neighborlist(hydrogens(positions), cutoff=cutoff, tile=16)
    npair = int(full.mask.sum().item())
    assert npair > 16, "test needs a system with more than 16 real pairs"

    small_capacity = npair // 2

    tiles = Tiles(positions, tile=16)
    tile_a, tile_b = tile_pairs(tiles, cutoff)

    ((native_i, native_j, _),) = _atom_pairs_within_thresholds(
        tiles, tile_a, tile_b, positions, (cutoff,), pair_filter="native"
    )
    ((python_i, python_j, _),) = _atom_pairs_within_thresholds(
        tiles,
        tile_a,
        tile_b,
        positions,
        (cutoff,),
        distance_kernel="broadcast",
        pair_filter="python",
    )

    native_nbl = _pad_to_capacity(
        [native_i], [native_j], nat, cutoff, 0.0, small_capacity, positions, dd
    )
    python_nbl = _pad_to_capacity(
        [python_i], [python_j], nat, cutoff, 0.0, small_capacity, positions, dd
    )
    assert native_nbl.overflow
    assert python_nbl.overflow

    assert torch.equal(native_nbl.idx_i, python_nbl.idx_i)
    assert torch.equal(native_nbl.idx_j, python_nbl.idx_j)


@pytest.mark.native
@pytest.mark.parametrize(
    "capacity",
    [None, 10_000, 100],  # rounded up; fixed and large; fixed and too small
)
def test_padded_output_matches_python_path(capacity: int | None) -> None:
    """With `padding`, the native path writes its pairs straight into the
    padded buffer. It must give exactly what the Python path gives: the
    same capacity, the pairs first (truncated if the capacity is too
    small), the padding value after them, and the number of pairs found.
    """
    torch.manual_seed(3)
    positions = torch.randn(300, 3, dtype=torch.float64, device="cpu") * 12.0
    thresholds = (3.0, 4.0)
    padding = _Padding(capacity=capacity, value=300)

    tiles = Tiles(positions, tile=16)
    tile_a, tile_b = tile_pairs(tiles, max(thresholds))

    native_results = _atom_pairs_within_thresholds(
        tiles,
        tile_a,
        tile_b,
        positions,
        thresholds,
        pair_filter="native",
        padding=padding,
    )
    python_results = _atom_pairs_within_thresholds(
        tiles,
        tile_a,
        tile_b,
        positions,
        thresholds,
        distance_kernel="broadcast",
        pair_filter="python",
        padding=padding,
    )
    unpadded_results = _atom_pairs_within_thresholds(
        tiles, tile_a, tile_b, positions, thresholds, pair_filter="native"
    )

    for native, python, unpadded in zip(
        native_results, python_results, unpadded_results
    ):
        assert torch.equal(native.idx_i, python.idx_i)
        assert torch.equal(native.idx_j, python.idx_j)
        assert native.n_found == python.n_found == unpadded.idx_i.shape[0]

        n_kept = min(native.n_found, native.idx_i.shape[0])
        assert torch.equal(native.idx_i[:n_kept], unpadded.idx_i[:n_kept])
        assert bool((native.idx_i[n_kept:] == 300).all())
        if capacity is None:
            assert native.idx_i.shape[0] % 4096 == 0
        else:
            assert native.idx_i.shape[0] == capacity


@pytest.mark.native
def test_padded_output_without_candidates() -> None:
    """No candidate tile pairs at all still give a padded, empty list."""
    positions = torch.tensor([[0.0, 0.0, 0.0], [50.0, 0.0, 0.0]], device="cpu")
    tiles = Tiles(positions, tile=1)
    tile_a, tile_b = tile_pairs(tiles, 2.0)
    tile_a, tile_b = tile_a[:0], tile_b[:0]

    ((idx_i, idx_j, n_found),) = _atom_pairs_within_thresholds(
        tiles,
        tile_a,
        tile_b,
        positions,
        (2.0,),
        pair_filter="native",
        padding=_Padding(capacity=None, value=2),
    )

    assert n_found == 0
    assert idx_i.shape[0] == idx_j.shape[0] == 0


@pytest.mark.native
def test_native_matches_dense_reference() -> None:
    """Cross-check against an independent, dense all-pairs computation --
    not just against the Python chunked path -- so a bug shared by both
    the native and Python tile-based implementations would still be
    caught.

    Compares *unordered* pairs: a tile pair only guarantees
    ``tile_a <= tile_b``, not ``atom_i < atom_j`` -- two different tiles'
    atom indices need not be sorted relative to each other, so either
    orientation of a real pair is valid and the Python path itself
    returns a mix of both (confirmed empirically before writing this
    test, rather than assumed)."""
    torch.manual_seed(3)
    positions = torch.randn(120, 3) * 8.0
    cutoff = 3.5

    tiles = Tiles(positions, tile=8)
    tile_a, tile_b = tile_pairs(tiles, cutoff)
    ((native_i, native_j, _),) = _atom_pairs_within_thresholds(
        tiles, tile_a, tile_b, positions, (cutoff,)
    )
    native_pairs = {
        frozenset((i, j)) for i, j in zip(native_i.tolist(), native_j.tolist())
    }

    nat = positions.shape[0]
    dense_pairs = set()
    for i in range(nat):
        for j in range(i + 1, nat):
            if torch.norm(positions[i] - positions[j]) <= cutoff:
                dense_pairs.add(frozenset((i, j)))

    assert native_pairs == dense_pairs


def test_pair_filter_native_conflicts_with_distance_kernel() -> None:
    """`pair_filter="native"` combined with an explicit `distance_kernel`
    must raise, not silently ignore one or the other: the native path
    computes distances itself and has no way to honour a requested
    formula."""
    positions = torch.randn(20, 3)
    tiles = Tiles(positions, tile=8)
    tile_a, tile_b = tile_pairs(tiles, 3.0)

    with pytest.raises(ValueError, match="pair_filter='native'"):
        _atom_pairs_within_thresholds(
            tiles,
            tile_a,
            tile_b,
            positions,
            (3.0,),
            distance_kernel="broadcast",
            pair_filter="native",
        )


@pytest.mark.cuda
def test_pair_filter_native_not_applicable_off_cpu() -> None:
    """`pair_filter="native"` is CPU-only; requesting it on another device
    must raise rather than silently falling back to the Python path."""
    device = torch.device("cuda")
    positions = torch.randn(20, 3, device=device)
    tiles = Tiles(positions, tile=8)
    tile_a, tile_b = tile_pairs(tiles, 3.0)

    with pytest.raises(ValueError, match="pair_filter='native'"):
        _atom_pairs_within_thresholds(
            tiles, tile_a, tile_b, positions, (3.0,), pair_filter="native"
        )


@pytest.mark.native
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_half_precision_uses_the_python_path(dtype: torch.dtype) -> None:
    """The native kernel is compiled for float32 and float64 only. On CPU,
    the automatic choice must use the pure-Python path for other dtypes
    instead of failing inside the native dispatch."""
    torch.manual_seed(0)
    positions = (torch.randn(60, 3) * 5.0).to(dtype)
    cutoff = 4.0

    tiles = Tiles(positions, tile=8)
    tile_a, tile_b = tile_pairs(tiles, cutoff)

    ((automatic_i, automatic_j, _),) = _atom_pairs_within_thresholds(
        tiles, tile_a, tile_b, positions, (cutoff,)
    )
    ((python_i, python_j, _),) = _atom_pairs_within_thresholds(
        tiles, tile_a, tile_b, positions, (cutoff,), pair_filter="python"
    )

    assert automatic_i.shape[0] > 0, "test needs a system with real pairs"
    assert torch.equal(automatic_i, python_i)
    assert torch.equal(automatic_j, python_j)


def test_pair_filter_native_not_applicable_for_half_precision() -> None:
    """`pair_filter="native"` with positions of a dtype the kernel is not
    compiled for must raise and name the dtype, not fail inside the
    native dispatch."""
    positions = torch.randn(20, 3, dtype=torch.float16)
    tiles = Tiles(positions, tile=8)
    tile_a, tile_b = tile_pairs(tiles, 3.0)

    with pytest.raises(ValueError, match="float32 or float64"):
        _atom_pairs_within_thresholds(
            tiles, tile_a, tile_b, positions, (3.0,), pair_filter="native"
        )
