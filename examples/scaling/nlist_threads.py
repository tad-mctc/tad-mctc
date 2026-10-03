# SPDX-Identifier: CC0-1.0
"""
CPU thread-count / dtype sweep for `build_neighborlist` construction alone
(no `cn_eeq`/CN consumption -- see `glu_ala.py` for that combined picture).
Characterises the current default, `_native.py`'s OpenMP-parallel filter,
across 1/2/4 threads and both dtypes, up to real multi-million-atom
structures.

Thread count has to be fixed before PyTorch's OpenMP runtime spins up, so
this script is meant to be invoked once per `(structure, threads, dtype)`
combination as a fresh process, with `OMP_NUM_THREADS`/`MKL_NUM_THREADS`
set in that process's environment by the caller (a shell loop or
`run_matrix.py`) -- setting them from inside this script via `os.environ`
after `torch` is already imported would be too late for the OpenMP
backend PyTorch picked at import time. `torch.set_num_threads` is also
called explicitly below, redundantly, so a stray unset environment
variable still gets the right thread count for the ATen/BLAS side; the
native extension's own `#pragma omp` loops share the process-wide OpenMP
runtime PyTorch's OpenMP backend configures, so the same call reaches
both.

Prints one JSON line to stdout per invocation -- easy to collect across
many subprocesses without parsing formatted text.
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import torch

from tad_mctc.io import read
from tad_mctc.neighbor._native import is_available as native_available
from tad_mctc.neighbor.list import build_neighborlist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Structure file (.xyz).")
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument(
        "--dtype", choices=("float32", "float64"), required=True
    )
    parser.add_argument("--cutoff", type=float, default=25.0)
    parser.add_argument("--tile", type=int, default=32)
    parser.add_argument("--reps", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=1)
    return parser.parse_args()


def choose_reps(nat: int) -> int:
    """Same shape as `glu_ala.py`'s: fewer repeats as structures grow, a
    multi-million-atom build already takes real wall time once."""
    if nat < 10_000:
        return 7
    if nat < 200_000:
        return 5
    if nat < 1_000_000:
        return 3
    return 1


def main() -> None:
    args = parse_args()

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)

    dtype = getattr(torch, args.dtype)
    structure = read.read_xyz(
        args.path,
        device=torch.device("cpu"),
        dtype=dtype,
    )
    nat = int(structure.numbers.shape[0])

    reps = args.reps if args.reps is not None else choose_reps(nat)

    nbl = build_neighborlist(structure, cutoff=args.cutoff, tile=args.tile)
    for _ in range(args.warmup - 1):
        nbl = build_neighborlist(structure, cutoff=args.cutoff, tile=args.tile)
    del nbl

    times: list[float] = []
    n_pairs = 0
    capacity = 0
    overflow = False
    for _ in range(reps):
        start = time.perf_counter()
        nbl = build_neighborlist(structure, cutoff=args.cutoff, tile=args.tile)
        times.append(time.perf_counter() - start)
        n_pairs = int(nbl.mask.sum())
        capacity = int(nbl.idx_i.shape[0])
        overflow = bool(nbl.overflow)
        del nbl

    times.sort()
    median = times[len(times) // 2]
    peak_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    print(
        json.dumps(
            {
                "path": str(args.path),
                "nat": nat,
                "threads": args.threads,
                "torch_threads": torch.get_num_threads(),
                "dtype": args.dtype,
                "cutoff": args.cutoff,
                "tile": args.tile,
                "reps": reps,
                "times_s": times,
                "median_s": median,
                "min_s": times[0],
                "n_pairs": n_pairs,
                "capacity": capacity,
                "overflow": overflow,
                "native_available": native_available(),
                "pairs_per_atom": n_pairs / nat if nat else 0.0,
                "atoms_per_s": nat / median if median else 0.0,
                "peak_rss_kb": peak_rss_kb,
            }
        )
    )


if __name__ == "__main__":
    main()
