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
Neighbour search: optional native CPU acceleration
==================================================

An OpenMP-parallel C++ implementation (:file:`_native_pairs.cpp`) of the
per-candidate "filter down to exact atom pairs" step inside
:func:`tad_mctc.neighbor.list._atom_pairs_within_thresholds`, opt-in and
compiled lazily.

Why this exists
----------------
Profiling a 1.7M-atom CPU build showed close to half of wall time in two
things that do not parallelize under PyTorch's CPU backend at all:
``torch.nonzero`` (a serial compaction, regardless of thread count) and
the Python-level loop dispatching one chunk at a time. mctc-lib's Fortran
CSR neighbour list (``mctc-lib/src/mctc/csrlist/type.f90``,
``generate_hybrid``) avoids both: there is no filter-then-compact step,
and a prefix sum over pair counts gives every writer its place in the
output. :file:`_native_pairs.cpp` applies that idea to this library's own
candidate structure (:class:`tad_mctc.neighbor._tiles.Tiles` and
:func:`tad_mctc.neighbor._tiles.tile_pairs`) rather than mctc-lib's
linked-cell grid: porting the grid too would replace a different,
already-measured front-end for no reason, so only the filter-and-compact
body is replaced. Where mctc-lib's threads append to private buffers
that are stitched together afterwards, the kernel here counts the pairs
of every candidate tile pair first and then writes them straight into
the output (see the comment at the top of :file:`_native_pairs.cpp`).

Why it is optional, not a runtime dependency
----------------------------------------------
This library has no compiled dependency at runtime, and this module keeps
that true for anyone who never triggers it: by default, the C++ source is compiled just-in-time, on first use, via
:func:`torch.utils.cpp_extension.load`, and any failure at all -- no C++
compiler, no OpenMP support, a build-toolchain quirk, whatever -- is
caught in :func:`_load` and simply turns native support off;
:func:`tad_mctc.neighbor.list._atom_pairs_within_thresholds` falls back to
the pure-Python path. The same path handles positions of a dtype the kernel
is not compiled for (see :data:`SUPPORTED_DTYPES`). Installing or importing
``tad_mctc`` never requires a compiler. Set the ``TAD_MCTC_DISABLE_NATIVE``
environment variable (any non-empty value) to force that fallback even when
the extension would otherwise load, e.g. to reproduce a result independent
of whether native support happened to be available.

That JIT compile happens once per environment, not once per process: the
first real invocation after a fresh install, a wiped cache, or a new
container pays a one-off compile-and-link cost (measured at ~13.5s) before
:func:`torch.utils.cpp_extension.load` finds its own on-disk build cache
and every subsequent process just loads the cached ``.so``. On a
long-lived machine that is a one-time tax; on something rebuilt often
(CI, an ephemeral container image), it recurs. For that case,
``TAD_MCTC_BUILD_NATIVE=1 pip install --no-build-isolation .`` builds
``tad_mctc.neighbor._native_pairs_ext`` ahead of time, against the
installed torch, as a normal compiled extension module installed
alongside the package, via
``setup.py``'s ``ext_modules``/``cmdclass`` (gated on that same
environment variable, so a plain ``pip install .`` still needs no
compiler). :func:`_load` tries importing that pre-built module first and
only falls back to the JIT path if it is not there, does not load, or was
built from another source than the installed one (as after editing
:file:`_native_pairs.cpp` in an editable install) -- so a
``TAD_MCTC_BUILD_NATIVE=1`` install never pays the first-run tax at all,
while the default install needs no compiler and compiles on first use.

Whether the JIT build is reused is decided by ``ninja``, the first one on
``PATH``. Two ``ninja`` versions with different build-log formats (1.11
and 1.13, say) do not recognise each other's builds, which happens when a
conda environment's activation hides the ``ninja`` of the base
environment. So every build configuration -- the source file, the
compiler, the flags and the ``ninja`` -- gets its own cache directory
(see :func:`_jit_name`), and each compiles once. :func:`build_info`
reports which way this process got the extension, whether it compiled it,
and which compiler and ``ninja`` it used.

On Linux, the kernel asks for transparent huge pages (``madvise``) for its
own large buffers, the hit masks and the output, which spares most of the
page faults of writing them. It changes no setting of the process, so the
memory of a program that embeds this library is unaffected; where huge
pages are off, nothing changes.

Machine-specific compiler flags
---------------------------------
Both builds use flags that run on any x86-64 CPU. On a known machine,
``TAD_MCTC_NATIVE_CFLAGS`` adds flags, split like a shell command line,
for example ``TAD_MCTC_NATIVE_CFLAGS="-mavx2"``. It applies to the JIT
build and to a ``TAD_MCTC_BUILD_NATIVE=1`` install; an already
installed pre-built module is used as it is. Each set of flags gets its
own JIT build, so runs with and without them do not recompile over each
other. The added flags come before ``-fno-fast-math -ffp-contract=off``,
which the kernel relies on (see :file:`_native_pairs.cpp`) and which they
therefore cannot undo. A library built with ``-march=native`` crashes on
an older CPU, and the JIT cache in ``~/.cache/torch_extensions`` may be
shared between machines, for example through a cluster's home directory.
If a build with added flags fails, a warning says why and the pure-Python
path is used.

Other compilers
-----------------
Both builds use the compiler named by ``CXX``, or ``c++`` if it is not
set. Each compiler gets its own JIT build, like each set of added flags.
Intel's ``icpx`` (``module load intel``) builds the kernel, with the
same output as GCC's, but three things differ:

- PyTorch warns that the compiler is not the one it was built with. This
  module silences that warning, since ``icpx`` uses GCC's C++ standard
  library, and :func:`build_info` names the compiler instead.
- The library needs Intel's runtime libraries (``libiomp5``, ``libsvml``,
  ...), which loading the compiler's module puts on the library path. A
  pre-built module that cannot find them is reported (see
  :func:`build_info`) and replaced by a JIT build with the compiler at
  hand.
- ``icpx`` links Intel's OpenMP runtime next to PyTorch's own. Its threads
  would keep spinning after the search while PyTorch's threads run; the
  kernel turns that off for the duration of a search.

What the native op must match, exactly
-----------------------------------------
The native op returns the same pairs, in the same order, for every
threshold, as the Python path with the ``"broadcast"`` distance kernel.
A list truncated to a too-small fixed ``capacity`` keeps the first
pairs, so this order is observable, not an internal detail:

- A distance is ``dx * dx + dy * dy + dz * dz`` of the direct coordinate
  differences, in the positions' dtype, and it is compared against the
  squared threshold rounded once to that dtype. A pair right at a
  threshold therefore lands on the same side in both paths. The Python
  path's automatic CPU kernel, ``"baddbmm"``, computes the distance by
  another formula, so it can disagree with both for a pair within
  rounding of a threshold; it only runs when the native extension is
  unavailable.

- Global order matches ``torch.nonzero``'s row-major ``(block, row,
  col)`` walk over ``(tile_a, tile_b)``'s candidates, exactly as the
  Python path's ``collected_i[k].append(...)`` sequence produces it. The
  native op counts the pairs of every candidate first and writes each
  candidate's pairs into its own range of the output, so the order does
  not depend on how the candidates were shared among threads.
- A same-tile candidate (``tile_a[c] == tile_b[c]``) keeps only its
  strict upper triangle (``row < col``); a cross-tile candidate keeps
  every ``(row, col)``. Both sides of a pair must come from a real
  (``valid``) slot.
- Every threshold is compared against the same computed distance, so a
  pair within a smaller threshold is within every larger one too.
- With ``anchor_atoms``, a pair needs at least one anchor atom.
- With ``padding``, the output has the capacity of ``_capacity_for`` in
  :mod:`tad_mctc.neighbor.list`, the pairs first and the padding value
  after them.

``test/test_neighbor/test_native.py`` checks this directly: build a
mid-size system both ways and assert ``torch.equal`` on every
``idx_i``/``idx_j``, plus an explicit undersized-``capacity`` case so the
truncation path is exercised.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import importlib
import importlib.util
import logging
import os
import shutil
import subprocess
import time
import warnings
from collections.abc import Generator
from typing import Any, Literal, NamedTuple

import torch

from ..typing import Tensor
from . import _native_flags

__all__ = [
    "BuildInfo",
    "SUPPORTED_DTYPES",
    "atom_pairs_within_thresholds_native",
    "build_info",
    "is_available",
]

_SOURCE_NAME = "_native_pairs.cpp"
_SOURCE = os.path.join(os.path.dirname(__file__), _SOURCE_NAME)
_PRECOMPILED = f"{__package__}._native_pairs_ext"

# The positions' dtypes the kernel is compiled for, by its
# `AT_DISPATCH_FLOATING_TYPES` in `_native_pairs.cpp`.
SUPPORTED_DTYPES = (torch.float32, torch.float64)


class BuildInfo(NamedTuple):
    """How this process obtained the native extension."""

    origin: Literal["precompiled", "jit", "disabled", "unavailable"]
    """
    ``"precompiled"``: the ``TAD_MCTC_BUILD_NATIVE=1`` install's module.
    ``"jit"``: compiled from source by :func:`torch.utils.cpp_extension.load`.
    ``"disabled"``: ``TAD_MCTC_DISABLE_NATIVE`` is set.
    ``"unavailable"``: loading or compiling failed. The last two use the
    pure-Python path.
    """

    library: str | None = None
    """Path of the loaded shared library."""

    compiled_now: bool = False
    """JIT only: whether this process compiled the library, instead of
    reusing the cached build."""

    ninja: str | None = None
    """JIT only: the ``ninja`` that decided whether to recompile, with its
    version."""

    cflags: tuple[str, ...] = ()
    """JIT only: the compiler flags of the build, including those added by
    ``TAD_MCTC_NATIVE_CFLAGS``."""

    error: str | None = None
    """Why loading or compiling failed, if it did. For a JIT build, why the
    pre-built module could not be used, if one was installed."""

    compiler: str | None = None
    """JIT only: the C++ compiler of the build, with its version."""


def _compiler() -> str | None:
    """
    The C++ compiler set by ``CXX``, which the JIT build runs instead of
    ``c++``, as a full path if it is on ``PATH``, or ``None`` if ``CXX`` is
    not set.
    """
    compiler = os.environ.get("CXX")
    if not compiler:
        return None
    return shutil.which(compiler) or compiler


def _jit_name(
    compiler: str | None, ninja: str | None, cflags: tuple[str, ...]
) -> str:
    """
    Name of the JIT build of the source by ``compiler`` with ``cflags``,
    run by ``ninja`` (both with their versions, as in :class:`BuildInfo`),
    which is also the name of its cache directory. Every configuration gets
    its own, so that processes with different ones do not recompile over
    each other.
    """
    key = "\n".join((_SOURCE, compiler or "", ninja or "", *cflags))
    digest = hashlib.sha256(key.encode()).hexdigest()[:8]
    return f"tad_mctc_native_pairs_{digest}"


def _source_digest() -> str | None:
    """Digest of the installed source (see
    :func:`._native_flags.source_digest`), or ``None`` if there is none to
    read."""
    try:
        with open(_SOURCE, "rb") as f:
            return _native_flags.source_digest(f.read())
    except OSError:
        return None


@functools.lru_cache(maxsize=1)
def _load_with_info() -> tuple[Any | None, BuildInfo]:
    """
    Load the native extension, once per process.

    Tries the ahead-of-time-compiled ``_native_pairs_ext`` module first
    (present only when the package was installed with
    ``TAD_MCTC_BUILD_NATIVE=1``, see the module docstring) so that install
    never pays any JIT compile cost; otherwise falls back to compiling the
    same source just-in-time via :func:`torch.utils.cpp_extension.load`.
    A pre-built module that is installed but does not import, or that was
    built from another source than the installed one (its
    ``source_digest``), is replaced by the JIT build as well, with a
    warning, and :attr:`BuildInfo.error` says why.

    Returns the loaded module, or ``None`` on any failure (missing
    compiler, missing OpenMP, a disabled-by-request environment variable,
    or any other compile/load error), and how it was obtained. Never
    raises, since a failure here must always mean "fall back to the
    pure-Python path", not a crash for a user who has no C++ toolchain at
    all.
    """
    if os.environ.get("TAD_MCTC_DISABLE_NATIVE"):
        return None, BuildInfo("disabled")

    # Without a pre-built module, the JIT build is the ordinary way. A
    # pre-built module that does not load, such as one built with `icpx`
    # whose runtime libraries are not on the library path, is reported, and
    # so is one built from another source, such as that of an editable
    # install before the source was edited.
    precompiled_error = None
    if importlib.util.find_spec(_PRECOMPILED) is not None:
        try:
            module = importlib.import_module(_PRECOMPILED)
        except ImportError as error:
            precompiled_error = f"pre-built module: {_error_summary(error)}"
            warnings.warn(
                f"The pre-built native neighbour search did not load, so "
                f"it is compiled just-in-time instead:\n{error}",
                stacklevel=2,
            )
        else:
            # Without the source to compare with, the module cannot be
            # checked, but there is no JIT build to fall back to either.
            digest = _source_digest()
            if digest is None or digest == getattr(
                module, "source_digest", None
            ):
                return module, BuildInfo("precompiled", module.__file__)
            precompiled_error = (
                f"pre-built module: built from another {_SOURCE_NAME}"
            )
            warnings.warn(
                f"The pre-built native neighbour search "
                f"({module.__file__}) was built from another "
                f"{_SOURCE_NAME} than the installed one, so it is compiled "
                f"just-in-time instead. Reinstall with "
                f"TAD_MCTC_BUILD_NATIVE=1 to update it.",
                stacklevel=2,
            )

    compiler = _with_version(_compiler() or shutil.which("c++"))
    ninja = _with_version(shutil.which("ninja"))
    extra = _native_flags.extra_cflags()
    cflags = _native_flags.cflags(extra)
    try:
        from torch.utils.cpp_extension import load

        started = time.time()
        with _without_wrong_compiler_warning():
            module: Any = load(
                name=_jit_name(compiler, ninja, cflags),
                sources=[_SOURCE],
                extra_cflags=list(cflags),
                extra_ldflags=["-fopenmp"],
                verbose=False,
            )
    except Exception as error:
        # Without added flags, a failure is the ordinary case of a machine
        # without a C++ toolchain. With them, it is most likely a flag the
        # compiler rejects, which the user should hear about.
        if extra:
            warnings.warn(
                f"The native neighbour search did not build with "
                f"TAD_MCTC_NATIVE_CFLAGS={' '.join(extra)!r}, so the "
                f"pure-Python path is used instead:\n{error}",
                stacklevel=2,
            )
        return None, BuildInfo(
            "unavailable", cflags=cflags, error=_error_summary(error)
        )

    # A library written after the load started was compiled by it.
    compiled_now = os.path.getmtime(module.__file__) >= started
    info = BuildInfo(
        "jit",
        library=module.__file__,
        compiled_now=compiled_now,
        ninja=ninja,
        cflags=cflags,
        error=precompiled_error,
        compiler=compiler,
    )
    return module, info


@contextlib.contextmanager
def _without_wrong_compiler_warning() -> Generator[None, None, None]:
    """
    Silence PyTorch's warning that the compiler is not the one PyTorch was
    built with. It comes on every load of a build by a compiler such as
    Intel's ``icpx``, which is compatible (see the module docstring), and
    recent PyTorch versions garble it into a logging traceback.
    :func:`build_info` names the compiler instead. Older PyTorch versions
    raise the warning with :func:`warnings.warn`, newer ones log it.
    """
    from torch.utils import cpp_extension

    logged_warning = getattr(cpp_extension, "WRONG_COMPILER_WARNING", None)

    def is_not_the_warning(record: logging.LogRecord) -> bool:
        return record.msg is not logged_warning

    logger = logging.getLogger(cpp_extension.__name__)
    logger.addFilter(is_not_the_warning)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=r"(?s).*Your compiler .* is not compatible"
            )
            yield
    finally:
        logger.removeFilter(is_not_the_warning)


def _error_summary(error: Exception) -> str:
    """
    One line saying why a build failed: the compiler's first ``error:``
    line if there is one, since a failed build's message starts with the
    whole compile command, otherwise the message's first line.
    """
    lines = str(error).splitlines() or [repr(error)]
    compiler_errors = [line for line in lines if "error:" in line]
    return (compiler_errors or lines)[0].strip()


def _with_version(program: str | None) -> str | None:
    """``program`` with the first line of its ``--version`` output, or
    ``None`` if there is no program."""
    if program is None:
        return None
    try:
        output = subprocess.run(
            [program, "--version"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return program
    lines = output.strip().splitlines()
    return f"{program} ({lines[0]})" if lines else program


def _load() -> Any | None:
    """The native extension, or ``None`` (see :func:`_load_with_info`)."""
    module, _ = _load_with_info()
    return module


def is_available() -> bool:
    """Whether the native CPU extension compiled and loaded successfully."""
    return _load() is not None


def build_info() -> BuildInfo:
    """How this process obtained the native extension. Loads it (and may
    compile it) if that has not happened yet."""
    _, info = _load_with_info()
    return info


def atom_pairs_within_thresholds_native(
    index: Tensor,
    valid: Tensor,
    tile_a: Tensor,
    tile_b: Tensor,
    positions: Tensor,
    thresholds: tuple[float, ...],
    anchor_atoms: Tensor | None = None,
    padding: tuple[int | None, int] | None = None,
    capacity_bucket: int = 1,
) -> list[tuple[Tensor, Tensor, int]] | None:
    """
    Native (OpenMP, CPU-only) equivalent of the exact-pair filter inside
    :func:`tad_mctc.neighbor.list._atom_pairs_within_thresholds`.

    Parameters mirror that function's own ``tiles.index``, ``tiles.valid``,
    ``tile_a``, ``tile_b``, ``reference_positions``, ``thresholds`` and
    ``anchor_atoms``. ``padding`` is ``(capacity, value)``, as in that
    function's ``_Padding``, and ``capacity_bucket`` is the multiple a
    ``None`` capacity is rounded up to.

    Returns
    -------
    list[tuple[Tensor, Tensor, int]] | None
        ``(idx_i, idx_j, n_found)`` per threshold, in the same order as
        ``thresholds``: 1D ``torch.int32`` tensors (the stored index
        dtype of :class:`.NeighborList`), padded to their capacity
        when ``padding`` is given, and the number of pairs found -- or
        ``None`` if the native extension is unavailable (see the module
        docstring); the caller then falls back to the pure-Python path.
    """
    module = _load()
    if module is None:
        return None

    capacity, pad_value = (None, None) if padding is None else padding
    thresholds_sq = [float(t) * float(t) for t in thresholds]
    result = module.atom_pairs_within_thresholds_cpu(
        index,
        valid,
        tile_a,
        tile_b,
        positions,
        thresholds_sq,
        anchor=anchor_atoms,
        pad_value=pad_value,
        capacity=capacity,
        capacity_bucket=capacity_bucket,
    )
    return list(result)
