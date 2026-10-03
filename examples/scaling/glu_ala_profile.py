# SPDX-Identifier: CC0-1.0
"""
`examples/scaling/glu_ala.py` only ever times two stages, `build_neighborlist`
and `cn_eeq`, and its own `--small` run showed `build` dominating: 2512 ms vs.
913 ms at 53,250 atoms, about 73% of the 3426 ms total. This script asks
which part of `build` that 73% actually is.

The steps of `build_neighborlist` are timed through the builder's private
`stage` hook (`tad_mctc.neighbor.list._build_neighborlists`), the same one
`tad_mctc --timing` uses: it wraps the tile build ("tiles"), the tile-pair
screen ("tile pairs"), the exact distance filter ("pair filter") and the
padding into the final list ("finalize"). Timing the steps of the real
build, rather than re-assembling it here, keeps this script in step with
the library. This is timed across every record of the packaged
`glu_ala_a` fixture (`tad_mctc.data.structures.glu_ala`, 28 to 53,250
atoms, no download needed -- see `glu_ala.py` for the full, downloaded
ladder up to 1.7 million atoms).

`cn_eeq`'s sparse path has no such hook: its per-pair steps are private
to `tad_mctc.ncoord.common` and run inside `CNModel`. For that half,
`torch.profiler` is used instead, at one `--profile-label` record (default `"0512"`, 13,314 atoms) picked well below
the ladder top -- `with_stack=True` and a multi-million-pair trace both make
the profiler impractical at the full 53,250-atom size, and a mid-sized
record already gives a representative op mix.
"""

import argparse
import contextlib
import statistics
import time
from collections.abc import Callable, Generator
from typing import TypeVar

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from tad_mctc.data.structures import get_structure, list_records
from tad_mctc.ncoord import cn_eeq
from tad_mctc.neighbor.list import _build_neighborlists, build_neighborlist

T = TypeVar("T")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--cpu", action="store_true", help="Run on CPU (default)."
    )
    group.add_argument(
        "--gpu",
        "--cuda",
        dest="gpu",
        action="store_true",
        help="Run on CUDA.",
    )
    parser.add_argument(
        "--profile-label",
        default="0512",
        choices=list_records("glu_ala"),
        help=(
            "Which packaged glu_ala_a record to run the torch.profiler "
            "breakdown on (default: %(default)s, 13,314 atoms). Keep this "
            "well below the 53,250-atom top of the ladder -- profiling "
            "overhead and trace size both grow with the pair count."
        ),
    )
    return parser.parse_args()


args = parse_args()
device = torch.device("cuda" if args.gpu else "cpu")
if args.gpu and not torch.cuda.is_available():
    raise SystemExit(
        "--gpu/--cuda given, but torch.cuda.is_available() is False"
    )

torch.set_printoptions(precision=6)
print(f"device: {device}\n")


def timed_median(
    fn: Callable[..., T],
    *args: object,
    reps: int,
    **kwargs: object,
) -> tuple[T, float]:
    """Median wall-clock time of `reps` (>= 1) calls `fn(*args, **kwargs)`,
    synchronizing CUDA around each one (a no-op on CPU) so async kernel
    launches are not mistaken for a fast call."""
    assert reps >= 1

    result: T | None = None
    times: list[float] = []
    for _ in range(reps):
        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        result = fn(*args, **kwargs)

        if device.type == "cuda":
            torch.cuda.synchronize()

        times.append(time.perf_counter() - start)

    assert result is not None
    return result, sorted(times)[len(times) // 2]


def choose_reps(nat: int) -> int:
    """Same tiering as `glu_ala.py`: fewer repetitions as structures grow."""
    if nat < 10_000:
        return 5
    if nat < 200_000:
        return 3
    return 1


class StageTimes:
    """Wall time of each step of one neighbour-list build, recorded
    through the builder's `stage` hook. CUDA is synchronized around each
    step (a no-op on CPU), so async kernel launches are not mistaken for a
    fast step."""

    def __init__(self) -> None:
        self.seconds: dict[str, float] = {}

    @contextlib.contextmanager
    def __call__(self, label: str) -> Generator[None, None, None]:
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()

        yield

        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        self.seconds[label] = self.seconds.get(label, 0.0) + elapsed


# ---------------------------------------------------------------------------
# Stage 1: break `build_neighborlist`'s own time into its steps
# ---------------------------------------------------------------------------
records = list_records("glu_ala")
steps = ("tiles", "tile pairs", "pair filter", "finalize")

columns = ("record", "nat", *(f"{step} (ms)" for step in steps), "cn (ms)")
widths = (7, 7, 10, 15, 16, 13, 9)
header = "  ".join(f"{c:>{w}s}" for c, w in zip(columns, widths))
print(header)
print("-" * len(header))

atom_counts: list[int] = []
step_shares: dict[str, list[float]] = {step: [] for step in steps}

for record in records:
    structure = get_structure(
        "glu_ala", record, device=device, dtype=torch.double
    )
    nat = int(structure.numbers.shape[0])
    cutoff = cn_eeq.cutoff

    reps = choose_reps(nat)

    # One untimed warm-up: allocator/thread-pool setup has nothing to do
    # with the breakdown below.
    nbl = build_neighborlist(structure, cutoff=cutoff)
    cn_eeq(structure, pairs=nbl)

    # Median over `reps` builds, per step.
    per_rep_times: list[StageTimes] = []
    for _ in range(reps):
        times = StageTimes()
        (nbl,) = _build_neighborlists(structure, (cutoff,), stage=times)
        per_rep_times.append(times)
    step_ms = {
        step: 1e3 * statistics.median(t.seconds[step] for t in per_rep_times)
        for step in steps
    }
    build_ms = sum(step_ms.values())

    cn_result, cn_time = timed_median(cn_eeq, structure, pairs=nbl, reps=reps)

    atom_counts.append(nat)
    for step in steps:
        step_shares[step].append(step_ms[step] / build_ms)

    cells = [f"{step_ms[step]:{w}.4f}" for step, w in zip(steps, widths[2:])]
    print(f"{record:>7s}  {nat:7d}  {'  '.join(cells)}  {cn_time * 1e3:9.4f}")

print(
    f"\nbuild-time share of each step, smallest -> largest record "
    f"({atom_counts[0]} -> {atom_counts[-1]} atoms):"
)
for step in steps:
    shares = step_shares[step]
    print(f"  {step:<12s}{shares[0] * 100:5.1f}% -> {shares[-1] * 100:5.1f}%")

# ---------------------------------------------------------------------------
# Stage 2: `cn_eeq`'s sparse pair-summation has no stage hook (its per-pair
# steps are private to `tad_mctc.ncoord.common`), so a
# torch.profiler op-level breakdown stands in for stage timing here, at one
# deliberately mid-sized record.
# ---------------------------------------------------------------------------
profile_structure = get_structure(
    "glu_ala", args.profile_label, device=device, dtype=torch.double
)
profile_nat = int(profile_structure.numbers.shape[0])

profile_nbl = build_neighborlist(profile_structure, cutoff=cn_eeq.cutoff)
cn_eeq(profile_structure, pairs=profile_nbl)  # warm-up

activities = [ProfilerActivity.CPU]
if device.type == "cuda":
    activities.append(ProfilerActivity.CUDA)
    torch.cuda.synchronize()

print(
    f"\ntorch.profiler breakdown of cn_eeq's sparse sum, record "
    f"{args.profile_label} ({profile_nat} atoms), 5 reps:"
)
with profile(activities=activities) as prof:
    for _ in range(5):
        with record_function("cn_eeq(sparse)"):
            cn_eeq(profile_structure, pairs=profile_nbl)

if device.type == "cuda":
    torch.cuda.synchronize()

sort_by = (
    "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
)
print(prof.key_averages().table(sort_by=sort_by, row_limit=15))
