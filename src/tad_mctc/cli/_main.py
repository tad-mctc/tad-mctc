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
Command line interface: the run
===============================

Reads the structure, builds what the chosen path needs, evaluates the
coordination number and prints the results.
"""

from __future__ import annotations

import argparse
import contextlib
from collections.abc import Callable, Generator
from typing import Any

import torch

from ..io.checks import coldfusion_check
from ..io.checks.structure import _coldfusion_uses_neighborlist
from ..io.read import read_structure
from ..io.structure import Structure
from ..ncoord.common import CNModel
from ..neighbor import _native
from ..neighbor.images import build_periodic_shifts
from ..neighbor.list import NeighborList, build_neighborlists
from ..typing import Tensor
from ._args import CN_MODELS, DTYPES, build_parser
from ._output import (
    print_native_build,
    print_neighborlist,
    print_results,
    print_system_info,
)
from ._timing import Timings

__all__ = ["main"]


def _build_neighborlist(
    structure: Structure, cutoff: float, timings: Timings
) -> NeighborList:
    """Build the neighbour list, timing each step of the build."""

    def nlist_stage(label: str) -> contextlib.AbstractContextManager[None]:
        return timings.stage(f"nlist: {label}")

    (nbl,) = build_neighborlists(
        structure,
        (cutoff,),  # pyright: ignore[reportCallIssue]
        stage=nlist_stage,
    )
    return nbl


def _coordination_number(
    args: argparse.Namespace,
    model: CNModel,
    structure: Structure,
    timings: Timings,
) -> Tensor:
    """Evaluate ``model`` on ``structure`` through the path the arguments
    select, timing each step."""
    label = f"cn_{args.cn}"

    if args.neighbor == "sparse":
        nbl = _build_neighborlist(structure, model.cutoff, timings)
        return _evaluate(
            args,
            lambda s, p: model(s, pairs=p, mode=args.mode),
            structure,
            nbl,
            f"{label} (sparse, {args.mode})",
            timings,
        )

    if structure.lattice is None:
        return _evaluate(
            args,
            lambda s, p: model(s),
            structure,
            None,
            f"{label} (dense)",
            timings,
        )

    assert structure.periodic is not None
    with timings.stage("build periodic shifts"):
        shifts = build_periodic_shifts(
            structure.lattice, structure.periodic, model.cutoff
        )
    return _evaluate(
        args,
        lambda s, p: model(s, pairs=p),
        structure,
        shifts,
        f"{label} (dense)",
        timings,
    )


def _evaluate(
    args: argparse.Namespace,
    func: Callable[[Structure, Any], Tensor],
    structure: Structure,
    pairs: Any,
    label: str,
    timings: Timings,
) -> Tensor:
    """Call ``func(structure, pairs)`` as the step ``label``. With
    ``--compile``, ``func`` goes through ``torch.compile(fullgraph=True)``
    and the first call, which traces and compiles, is its own step, so the
    step ``label`` times the compiled function alone."""
    if not args.compile:
        with timings.stage(label):
            return func(structure, pairs)

    compiled = torch.compile(func, fullgraph=True)
    with timings.stage(f"torch.compile ({label})"):
        compiled(structure, pairs)
    with timings.stage(f"{label}, compiled"):
        return compiled(structure, pairs)


@contextlib.contextmanager
def _torch_threads(n_threads: int | None) -> Generator[None, None, None]:
    """Run the ``with`` block with ``n_threads`` intra-op threads, then
    restore the previous count, so that calling :func:`main` from Python
    leaves the caller's setting alone. ``None`` keeps the current count."""
    if n_threads is None:
        yield
        return

    previous = torch.get_num_threads()
    torch.set_num_threads(n_threads)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``tad_mctc`` command line tool."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.nlist_only and args.neighbor == "dense":
        parser.error(
            "--nlist-only builds a neighbour list, so it cannot be "
            "combined with '--neighbor dense'."
        )

    if args.compile and args.mode == "recompute" and args.neighbor == "sparse":
        parser.error(
            "--compile needs '--mode graph': 'recompute' checkpoints the "
            "pair loop, which is not meant for a compiled function."
        )

    if args.cuda and not torch.cuda.is_available():
        raise SystemExit("--cuda given but no CUDA device is available.")

    with _torch_threads(args.omp):
        _run(args)
    return 0


def _run(args: argparse.Namespace) -> None:
    """Carry out the run the parsed arguments describe."""
    model = CN_MODELS[args.cn]
    timings = Timings(args.timing)

    with timings.stage("read structure"):
        structure = read_structure(args.structure, dtype=DTYPES[args.dtype])

    # A neighbour search on the CPU runs through the native extension, and
    # so does the optional check on the structure just read, unless it is a
    # small molecule compared densely. Loading the extension can mean
    # compiling it, so it is its own step rather than part of whichever
    # search happens to come first.
    checks_with_native = args.coldfusion_check and (
        _coldfusion_uses_neighborlist(structure)
    )
    uses_native = checks_with_native or (
        not args.cuda and args.neighbor == "sparse"
    )
    if uses_native:
        with timings.stage("native extension"):
            _native.is_available()

    if args.coldfusion_check:
        with timings.stage("cold-fusion check"):
            coldfusion_check(structure)
    if args.cuda:
        with timings.stage("move to device"):
            structure = structure.to(torch.device("cuda"))

    if args.nlist_only:
        nbl = _build_neighborlist(structure, model.cutoff, timings)
        print_system_info(args.structure, structure)
        if uses_native:
            print_native_build(_native.build_info())
        print_neighborlist(nbl)
        timings.report()
        return

    cn = _coordination_number(args, model, structure, timings)

    print_system_info(args.structure, structure)
    if uses_native:
        print_native_build(_native.build_info())
    print_results(f"cn_{args.cn}", cn, structure)
    timings.report()
