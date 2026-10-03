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
from __future__ import annotations

import os
import runpy

from setuptools import setup

# Building the optional native CPU neighbour-list extension ahead of time
# (rather than letting it JIT-compile on first use, see
# `tad_mctc.neighbor._native`) is opt-in via this environment variable, not
# a `pip install` extra: `extras_require` only changes which *dependencies*
# get pulled in, it cannot change what a build step does, and this changes
# `ext_modules`/`cmdclass` -- so it has to be an env var, checked before
# `setup()` runs. Left off by default so that `pip install .` alone never
# needs a C++ compiler: the library has no compiled dependency at install
# or import time.
ext_modules = []
cmdclass: dict[str, type] = {}

if os.environ.get("TAD_MCTC_BUILD_NATIVE"):
    # The extension must be built against the torch it will run with, so
    # it is not a build requirement (pip would install a fresh, possibly
    # different torch into an isolated build environment). The build has
    # to see the installed one instead.
    try:
        from torch.utils.cpp_extension import BuildExtension, CppExtension
    except ImportError as e:
        raise RuntimeError(
            "TAD_MCTC_BUILD_NATIVE needs torch at build time. Install torch "
            "first, then run `pip install --no-build-isolation .`."
        ) from e

    # The same flags as the just-in-time build. The package is not
    # installed yet, so its flag module is run as a plain file.
    native_flags = runpy.run_path(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "src",
            "tad_mctc",
            "neighbor",
            "_native_flags.py",
        )
    )
    compile_args = native_flags["cflags"](native_flags["extra_cflags"]())

    # The module records the digest of the source it was built from, so
    # that `_native` can tell when the source has changed since, as after
    # editing it in an editable install, and compile it just-in-time.
    source = "src/tad_mctc/neighbor/_native_pairs.cpp"
    with open(source, "rb") as f:
        digest = native_flags["source_digest"](f.read())

    ext_modules = [
        CppExtension(
            name="tad_mctc.neighbor._native_pairs_ext",
            sources=[source],
            define_macros=[("TAD_MCTC_SOURCE_DIGEST", digest)],
            extra_compile_args=list(compile_args),
            extra_link_args=["-fopenmp"],
        ),
    ]
    cmdclass = {"build_ext": BuildExtension}

if __name__ == "__main__":
    setup(ext_modules=ext_modules, cmdclass=cmdclass)
