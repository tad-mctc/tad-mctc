# SPDX-Identifier: CC0-1.0
"""
`examples/scaling/polyalanine.py` proves dense and sparse `cn_eeq` agree and
shows the O(nat^2)-vs-O(pairs) gap widening up to 1003 atoms. This example
takes the same sparse, neighbour-list path to real HPC scale: an extended
glutamine-alanine peptide chain from 28 up to 1.7 million atoms, from the
two size ladders at https://www.ergoscf.org/xyz/gluala.php.

`--small` (the default) downloads and unpacks just ``glu_ala_a_0001_to_2048/``
(28 to 53,250 atoms) next to this script on first run. `--large` also fetches
``glu_ala_b_512_to_65536/`` (13,314 to 1,703,938 atoms) and runs the whole
combined ladder up to 1.7 million atoms; expect that to take several minutes
and, extrapolating from measurements up to 212,994 atoms, on the order of
25-30 GB of *system* RAM. `glu_ala_a`, plus `glu_ala_b`'s own 106,498- and
212,994-atom structures, are also available with no download at all, as a
compressed package fixture, via `tad_mctc.data.structures.glu_ala`.

On a 12 GB-class GPU, `--large --cuda` is expected to fit up to roughly
`glu_ala_b`'s `"8192"` label (212,994 atoms, ~3.5 GB peak) but not much
further: `"16384"` (425,986 atoms) alone peaks around 6.7 GB in
`mode="graph"`, most of a 12 GB card's ~11 GB actually free. These numbers
are CPU `torch.double` `ru_maxrss` (this environment has no CUDA device to
measure against directly) and are a proxy, not a guarantee, for CUDA peak
allocation -- the caching allocator, block fragmentation, and kernel
workspaces can move actual GPU peak in either direction. `--mode recompute`
(below) cuts the CPU number to ~4.7 GB by checkpointing the CN sum, but
`build_neighborlist` itself already accounts for that remaining ~4.7 GB and
`--mode` does nothing to shrink list construction itself -- so `"32768"` and
`"65536"` (851,970 and 1,703,938 atoms) are expected to stay out of reach of
a 12 GB card in either mode.

The warm-up/timed-call and cross-iteration `nbl`s are dropped (plus
`torch.cuda.empty_cache()` on CUDA) as soon as they are no longer needed: on
CUDA, a retained list occupies VRAM the caching allocator cannot hand to
the next, concurrent build. CPU `ru_maxrss` A/B testing at 106,498 and
425,986 atoms showed no change either way from this -- but that is not
evidence of no effect: `ru_maxrss` is a resident-page high-water mark, not
a concurrent-liveness counter, and cannot see what these `del`s change. The
CUDA-side check that would (`torch.cuda.max_memory_allocated` around the
same transition) needs a device this environment does not have. `--mode
recompute`'s ~32% reduction (below), by contrast, is directly CPU-measured.

Dense-vs-sparse agreement is still checked once, on the smallest structure:
running the dense, all-pairs path anywhere near the large end would need a
`nat x nat` distance matrix (millions x millions of entries, TB of memory),
which is exactly the case sparse construction exists to avoid. Every size
after that is sparse-only, timing both list construction and the CN sum it
feeds, to show wall-clock time tracking `nat`, not `nat^2`.

`--mode recompute` switches `cn_eeq`'s neighbor-list consumption from the
default `"graph"` (chunked, but every chunk's intermediates kept for the
backward pass, `O(n_pairs)` memory) to `CNModel`'s checkpointed pair sum
(`O(chunk)` memory) -- see the GPU VRAM note above for what it does and
does not buy.

Per-structure timings are also written to
``glu_ala_scaling_<small|large>_<cpu|cuda>[_recompute].txt`` next to this
script, one file per ladder/device/mode combination (the `_recompute` suffix
is only appended for `--mode recompute`, so the default `"graph"` run's
filename is unchanged) -- see `examples/scaling/glu_ala_plot.py` to plot
them (across CPU and CUDA together, if both were run).
"""

import argparse
import tarfile
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import torch

from tad_mctc.io import read
from tad_mctc.ncoord import cn_eeq
from tad_mctc.neighbor.list import build_neighborlist

T = TypeVar("T")

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_URL = "https://www.ergoscf.org/files/molecules/"
LADDERS = {
    "a": (SCRIPT_DIR / "glu_ala_a_0001_to_2048", "glu_ala_a_0001_to_2048"),
    "b": (SCRIPT_DIR / "glu_ala_b_512_to_65536", "glu_ala_b_512_to_65536"),
}
DEFAULT_MAX_NAT = 53_250  # glu_ala_a's largest structure
LADDER_A_MAX_LABEL = 2048  # glu_ala_a's largest size label


def _safe_extractall(tar: tarfile.TarFile, directory: Path) -> None:
    """Safely `tar.extractall(directory)`.

    Only regular files and directories are allowed. Rejecting links
    up front also blocks the known symlink/hardlink bypasses of the
    `data` filter (CVE-2025-4517 et al., fixed in 3.10.18 / 3.11.13 /
    3.12.11 / 3.13.4) on interpreters that predate those fixes.
    Extraction then uses `filter="data"` (PEP 706; 3.10.12+ / 3.11.4+),
    which also blocks path traversal, device files, and unsafe
    permission bits. Older interpreters are refused rather than given
    a weaker hand-rolled check.
    """
    if not hasattr(tarfile, "data_filter"):
        raise RuntimeError(
            "Safe tar extraction requires Python 3.10.12+ / 3.11.4+"
        )

    members = tar.getmembers()
    for member in members:
        if not (member.isfile() or member.isdir()):
            raise RuntimeError(
                f"Unsupported member type in archive: {member.name}"
            )

    tar.extractall(directory, members=members, filter="data")


def ensure_ladder(directory: Path, archive_stem: str) -> None:
    """Download and unpack `<archive_stem>.tar.gz` into `directory` if it
    is not already there. Ladder `a`'s archive has its members at its root;
    ladder `b`'s wraps them in a `<archive_stem>/` subdirectory -- either
    way, `directory` ends up holding the `.xyz` files directly, which is
    also what makes the existence check below (and re-runs) work."""
    if directory.is_dir() and any(directory.glob("*.xyz")):
        return

    url = f"{BASE_URL}{archive_stem}.tar.gz"
    archive_path = directory.parent / f"{archive_stem}.tar.gz"
    directory.mkdir(parents=True, exist_ok=True)

    print(f"downloading {url} -> {archive_path}")
    urllib.request.urlretrieve(url, archive_path)

    print(f"extracting {archive_path} -> {directory}/")
    with tarfile.open(archive_path) as tar:
        _safe_extractall(tar, directory)
    archive_path.unlink()

    if not any(directory.glob("*.xyz")):
        wrapper = directory / archive_stem
        if wrapper.is_dir():
            for child in wrapper.iterdir():
                child.rename(directory / child.name)
            wrapper.rmdir()

    print(f"{len(list(directory.glob('*.xyz')))} files ready in {directory}/")


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
    size_group = parser.add_mutually_exclusive_group()
    size_group.add_argument(
        "--small",
        action="store_true",
        help=f"Run only glu_ala_a (28 to {DEFAULT_MAX_NAT} atoms). Default.",
    )
    size_group.add_argument(
        "--large",
        action="store_true",
        help=(
            "Also download glu_ala_b and run the whole combined ladder, "
            "up to 1.7 million atoms. Slow and memory-hungry."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("graph", "recompute"),
        default="graph",
        help=(
            "cn_eeq neighbor-list consumption mode (see module docstring). "
            "'recompute' trades time for memory via checkpointed chunks; "
            "it does not reduce build_neighborlist's own footprint, which "
            "dominates peak memory at the largest sizes. Default: 'graph'."
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


def discover_structures(dirs: tuple[Path, ...]) -> list[tuple[str, Path]]:
    """One `.xyz` path per size label, first directory in `dirs` wins on a
    label both directories carry (they are identical there anyway)."""
    by_label: dict[str, Path] = {}
    for d in dirs:
        if not d.is_dir():
            continue
        for path in sorted(d.glob("*.xyz")):
            by_label.setdefault(path.stem, path)
    return sorted(by_label.items(), key=lambda kv: int(kv[0]))


def peek_natoms(path: Path) -> int:
    """Atom count from an xyz file's first line, without parsing the rest --
    cheap enough to decide whether a multi-hundred-MB file is worth reading
    at all before actually reading it."""
    with open(path, encoding="utf-8") as fh:
        return int(fh.readline().strip())


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

        # Drop the previous rep's result before building the next one --
        # otherwise, for reps >= 2, two full results are transiently live
        # at once (the old one pending rebinding, the new one mid-build),
        # roughly doubling peak memory right where reps are highest for
        # small structures and non-trivial for mid-sized ones.
        result = None

        start = time.perf_counter()
        result = fn(*args, **kwargs)

        if device.type == "cuda":
            torch.cuda.synchronize()

        times.append(time.perf_counter() - start)

    assert result is not None
    return result, sorted(times)[len(times) // 2]


def choose_reps(nat: int) -> int:
    """Fewer repetitions as structures get bigger -- a 1.7M-atom build
    already takes real time on its own; timing it 5x over would not."""
    if nat < 10_000:
        return 5
    if nat < 200_000:
        return 3
    return 1


ladder_name = "large" if args.large else "small"

ensure_ladder(*LADDERS["a"])
dirs: tuple[Path, ...] = (LADDERS["a"][0],)
if args.large:
    ensure_ladder(*LADDERS["b"])
    dirs = (LADDERS["a"][0], LADDERS["b"][0])

structures = discover_structures(dirs)

if args.large:
    largest_label = max(int(label) for label, _ in structures)
    if largest_label <= LADDER_A_MAX_LABEL:
        raise SystemExit(
            f"--large requested, but no glu_ala_b structures were found "
            f"(largest label discovered was {largest_label!r}, at or below "
            f"glu_ala_a's own maximum of {LADDER_A_MAX_LABEL}). Check that "
            f"{LADDERS['b'][0]}/ actually contains .xyz files -- "
            "ensure_ladder may have extracted the archive to the wrong "
            "layout."
        )

n_first = peek_natoms(structures[0][1])
n_last = peek_natoms(structures[-1][1])
print(f"{len(structures)} structures, {n_first} to {n_last} atoms\n")

# ---------------------------------------------------------------------------
# Dense-vs-sparse sanity check, once, on the smallest structure only
# ---------------------------------------------------------------------------
small_label, small_path = structures[0]
structure0 = read.read_xyz(small_path, device=device, dtype=torch.double)
dense0 = cn_eeq(structure0)
nbl0 = build_neighborlist(structure0, cutoff=cn_eeq.cutoff)
sparse0 = cn_eeq(structure0, pairs=nbl0, mode=args.mode)
max_abs_diff0 = (dense0 - sparse0).abs().max()
assert torch.allclose(
    dense0, sparse0, atol=1e-9
), f"{small_label}: dense and sparse CN disagree, max |Δ| = {max_abs_diff0}"
print(
    f"sanity check ({int(structure0.numbers.shape[0])} atoms): dense and sparse CN "
    f"agree, max |Δ| = {float(max_abs_diff0):.2e}\n"
)

# ---------------------------------------------------------------------------
# Sparse-only scaling across the whole ladder: dense is infeasible up here
# ---------------------------------------------------------------------------
columns = ("label", "nat", "pairs", "build (ms)", "cn (ms)", "total (ms)")
widths = (7, 9, 9, 11, 10, 11)
header = "  ".join(f"{c:>{w}s}" for c, w in zip(columns, widths))
print(header)
print("-" * len(header))

atom_counts: list[int] = []
total_times: list[float] = []
rows: list[tuple[str, int, int, float, float, float]] = []

for label, path in structures:
    structure = read.read_xyz(path, device=device, dtype=torch.double)
    nat = int(structure.numbers.shape[0])

    reps = choose_reps(nat)

    # One untimed warm-up per path, same reasoning as `polyalanine.py`:
    # first-call allocator/thread-pool setup has nothing to do with the
    # scaling story.
    nbl = build_neighborlist(structure, cutoff=cn_eeq.cutoff)
    cn_eeq(structure, pairs=nbl, mode=args.mode)

    # Drop the warm-up's neighbor list before the timed call builds a new,
    # equally large one -- otherwise both are transiently live at once,
    # close to doubling peak memory right at the sizes where it matters
    # most. `empty_cache()` only belongs here, between calls, never inside
    # `timed_median`'s own loop, where it would contaminate the timings.
    del nbl
    if device.type == "cuda":
        torch.cuda.empty_cache()

    nbl, build_time = timed_median(
        build_neighborlist, structure, cutoff=cn_eeq.cutoff, reps=reps
    )
    n_pairs = int(nbl.mask.sum())
    _, cn_time = timed_median(
        cn_eeq,
        structure,
        pairs=nbl,
        mode=args.mode,
        reps=reps,
    )
    total_time = build_time + cn_time

    atom_counts.append(nat)
    total_times.append(total_time)
    rows.append((label, nat, n_pairs, build_time, cn_time, total_time))

    print(
        f"{label:>7s}  {nat:9d}  {n_pairs:9d}  {build_time * 1e3:11.4f}  "
        f"{cn_time * 1e3:10.4f}  {total_time * 1e3:11.4f}"
    )

    # Drop this iteration's neighbor list and geometry before the next
    # iteration's `read_xyz`/`build_neighborlist` rebind them -- otherwise
    # the previous (smaller but still large) structure's tensors stay live
    # across the loop boundary, on top of whatever peak the next, bigger
    # structure reaches on its own.
    del nbl, structure
    if device.type == "cuda":
        torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# Persist timings for examples/scaling/glu_ala_plot.py, keyed by ladder and
# device so a CPU and a CUDA run of the same ladder land in separate files
# and can later be plotted together.
# ---------------------------------------------------------------------------
mode_suffix = "_recompute" if args.mode == "recompute" else ""
data_path = (
    SCRIPT_DIR / f"glu_ala_scaling_{ladder_name}_{device.type}{mode_suffix}.txt"
)
with open(data_path, "w", encoding="utf-8") as fh:
    fh.write("# label nat pairs build_ms cn_ms total_ms\n")
    for row_label, row_nat, row_pairs, row_build, row_cn, row_total in rows:
        fh.write(
            f"{row_label} {row_nat} {row_pairs} {row_build * 1e3:.6f} "
            f"{row_cn * 1e3:.6f} {row_total * 1e3:.6f}\n"
        )
print(f"\ntiming data written to {data_path}")

# ---------------------------------------------------------------------------
# The scaling itself: total time tracks nat, not nat^2
# ---------------------------------------------------------------------------
n_growth = atom_counts[-1] / atom_counts[0]
time_growth = total_times[-1] / total_times[0]

print(
    f"\nfrom {atom_counts[0]} to {atom_counts[-1]} atoms ({n_growth:.1f}x more):"
)
print(
    f"  sparse total time grew {time_growth:8.1f}x  "
    f"(nat growth was {n_growth:.1f}x; nat^2 growth would have been "
    f"{n_growth**2:.2e}x)"
)
