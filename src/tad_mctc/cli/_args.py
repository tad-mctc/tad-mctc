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
Command line interface: arguments
=================================

The argument parser and the choices it offers.
"""

from __future__ import annotations

import argparse

import torch

from ..ncoord import cn_d3, cn_d4, cn_eeq, cn_eeqbc, cn_gfn2
from ..ncoord.common import CNModel

__all__ = ["CN_MODELS", "DTYPES", "build_parser"]

CN_MODELS: dict[str, CNModel] = {
    "d3": cn_d3,
    "d4": cn_d4,
    "eeq": cn_eeq,
    "eeqbc": cn_eeqbc,
    "gfn2": cn_gfn2,
}
"""Available ``--cn`` choices, keyed by the name used on the command line."""

DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float64": torch.float64,
}
"""Available ``--dtype`` choices, keyed by the name used on the command
line."""

_EPILOG = """\
The native CPU neighbour search compiles with extra flags from the
TAD_MCTC_NATIVE_CFLAGS environment variable, e.g.
TAD_MCTC_NATIVE_CFLAGS=-mavx2 tad_mctc structure.xyz. A build with them
runs only on CPUs that support them. The "Native extension" section of
the output lists the flags of the build in use.
"""


def _positive_int(value: str) -> int:
    """An ``argparse`` type for a count that must be at least one, so a
    bad value is reported as a usage error instead of a traceback from
    wherever it is first used."""
    try:
        number = int(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"invalid int value: '{value}'") from e
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {number}")
    return number


def build_parser() -> argparse.ArgumentParser:
    """The parser of the ``tad_mctc`` command line."""
    parser = argparse.ArgumentParser(
        prog="tad_mctc",
        description="Calculate the coordination number of a structure.",
        epilog=_EPILOG,
    )
    parser.add_argument(
        "structure",
        type=str,
        help="Path to the structure file (xyz, Turbomole coord, ...).",
    )
    parser.add_argument(
        "--cn",
        dest="cn",
        choices=sorted(CN_MODELS),
        default="d3",
        help="Coordination number model to evaluate. Defaults to 'd3'.",
    )
    parser.add_argument(
        "--neighbor",
        dest="neighbor",
        choices=("dense", "sparse"),
        default="sparse",
        help=(
            "Neighbour search used for the coordination number: 'dense' "
            "evaluates all atom pairs, masked at the model's cutoff; "
            "'sparse' builds a padded neighbour list first. Defaults to "
            "'sparse'."
        ),
    )
    parser.add_argument(
        "--nlist-only",
        dest="nlist_only",
        action="store_true",
        help=(
            "Only build the neighbour list and print a summary of it, "
            "skipping the coordination number. '--cn' still selects the "
            "cutoff the list is built for. Incompatible with "
            "'--neighbor dense', which builds no list."
        ),
    )
    parser.add_argument(
        "--mode",
        dest="mode",
        choices=("graph", "recompute"),
        default="graph",
        help=(
            "How the sparse path evaluates the neighbour list: 'graph' "
            "keeps every pair's intermediates for the backward pass "
            "(memory O(n_pairs), supports vmap); 'recompute' checkpoints "
            "it in chunks (memory O(chunk), no vmap) -- use this for a "
            "system too large to fit the whole neighbour list's "
            "intermediates in memory. Ignored with '--neighbor dense'. "
            "Defaults to 'graph'."
        ),
    )
    parser.add_argument(
        "--coldfusion-check",
        dest="coldfusion_check",
        action="store_true",
        help=(
            "Run the interatomic-distance sanity check while reading. "
            "That check builds its own neighbour list on the CPU, before "
            "'--cuda' takes effect, and can dominate read time for a "
            "large structure -- opt in for an untrusted geometry. "
            "Disabled by default."
        ),
    )
    parser.add_argument(
        "--timing",
        action="store_true",
        help=(
            "Print the wall time of each computation step, including "
            "each step of the neighbour-list build."
        ),
    )
    parser.add_argument(
        "--dtype",
        dest="dtype",
        choices=sorted(DTYPES),
        default="float64",
        help=(
            "Floating point precision of the positions and lattice. "
            "Defaults to 'float64'."
        ),
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="Run on the first CUDA device instead of the CPU.",
    )
    parser.add_argument(
        "--omp",
        type=_positive_int,
        default=None,
        metavar="N",
        help=(
            "Number of threads torch uses for intra-op CPU parallelism "
            "during this run."
        ),
    )
    return parser
