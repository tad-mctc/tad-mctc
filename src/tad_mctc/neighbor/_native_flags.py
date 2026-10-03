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
Neighbour search: compiler flags of the native extension
========================================================

The compiler flags of the native CPU extension (:mod:`._native`), and the
digest of its source, in one place for its two builds: the just-in-time
build on first use and the ahead-of-time build of a
``TAD_MCTC_BUILD_NATIVE=1`` install. The
latter runs in ``setup.py`` before the package is installed, which loads
this file on its own, so it imports nothing but the standard library.
"""

from __future__ import annotations

import hashlib
import os
import shlex

__all__ = ["cflags", "extra_cflags", "source_digest"]


def extra_cflags() -> tuple[str, ...]:
    """The compiler flags added by ``TAD_MCTC_NATIVE_CFLAGS``, split like a
    shell command line."""
    return tuple(shlex.split(os.environ.get("TAD_MCTC_NATIVE_CFLAGS", "")))


def cflags(extra: tuple[str, ...]) -> tuple[str, ...]:
    """
    The compiler flags of a build with the ``extra`` flags added. They come
    before ``-fno-fast-math -ffp-contract=off``, so that they cannot turn
    fast math or floating-point contraction back on (see
    ``_native_pairs.cpp`` for why both are off).
    """
    return (
        "-O3",
        *extra,
        "-fno-fast-math",
        "-ffp-contract=off",
        "-fopenmp",
    )


def source_digest(source: bytes) -> str:
    """
    Digest of the extension's source code ``source``. The ahead-of-time
    build compiles it into the module (as ``TAD_MCTC_SOURCE_DIGEST``), and
    :mod:`._native` compares it with the digest of the installed source.
    Hexadecimal digits only, so that it is a single preprocessor token.
    """
    return hashlib.sha256(source).hexdigest()[:16]
