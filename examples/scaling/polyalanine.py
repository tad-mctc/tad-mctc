# SPDX-Identifier: CC0-1.0
"""
`CNModel` (`examples/cn/molecular_dense_single.py`) has two code paths for a
molecule: dense, all-pairs (`pairs=None`) and sparse, `NeighborList`-based
(`pairs=nbl`, see `examples/neighbor/list.py` for what that list
is). The dense path is `O(nat^2)` -- every atom pair is formed, then masked at
`cutoff` -- while the sparse path only ever touches the pairs the list
already found to be within `cutoff`, so its cost tracks the *pair count*,
not `nat^2`.

`mstore`'s `polyalanine` collection is a ready-made size ladder for seeing
that difference: 25 extended-chain conformers from 43 to 1003 atoms, each
about as long as the last is short, so `cutoff`'s Bohr radius reaches a
roughly constant, non-`nat^2`-growing number of neighbours per atom
regardless of chain length. This example runs both paths over the whole
ladder, checks they agree, and times both to show the resulting gap widen
with `nat`. For the same story at real HPC scale (up to 1.7 million
atoms), see `examples/scaling/glu_ala.py`.
"""

import argparse
import time
from collections.abc import Callable

import torch

from tad_mctc.data.structures import get_structure, list_records
from tad_mctc.ncoord import cn_eeq
from tad_mctc.neighbor.list import build_neighborlist


def parse_device() -> torch.device:
    """`--cpu` (default) or `--gpu`/`--cuda` from the command line."""
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
    args = parser.parse_args()

    if args.gpu:
        if not torch.cuda.is_available():
            parser.error(
                "--gpu/--cuda given, but torch.cuda.is_available() is False"
            )
        return torch.device("cuda")
    return torch.device("cpu")


device = parse_device()

torch.set_printoptions(precision=6)
print(f"device: {device}\n")


def timed_median(
    fn: Callable[..., torch.Tensor],
    *args: object,
    reps: int = 5,
    **kwargs: object,
) -> tuple[torch.Tensor, float]:
    """Median wall-clock time of `reps` (>= 1) calls `fn(*args, **kwargs)`,
    synchronizing CUDA around each one (a no-op on CPU) so async kernel
    launches are not mistaken for a fast call. Takes `fn`'s own arguments
    rather than a zero-argument closure, so no loop-local closure ever has
    to capture (and risk staying bound to) a per-iteration variable."""
    assert reps >= 1

    result: torch.Tensor | None = None
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


# ---------------------------------------------------------------------------
# Dense vs. sparse, one size at a time: same result, growing speed gap
# ---------------------------------------------------------------------------
columns = (
    "record",
    "nat",
    "pairs",
    "dense (ms)",
    "sparse (ms)",
    "speedup",
    "max |Δ|",
)
widths = (12, 5, 8, 11, 12, 8, 9)
header = "  ".join(f"{c:>{w}s}" for c, w in zip(columns, widths))
print(header)
print("-" * len(header))

records = list_records("polyalanine")
dense_times: list[float] = []
sparse_times: list[float] = []
atom_counts: list[int] = []

for record in records:
    structure = get_structure(
        "polyalanine", record, device=device, dtype=torch.double
    )
    nat = int(structure.numbers.shape[0])

    # `nbl` only ever needs rebuilding when the geometry (or cutoff) changes,
    # never per call -- see `examples/neighbor/list.py` for that reuse story.
    # `cn_eeq.cutoff` keeps this in sync with the model actually evaluated.
    nbl = build_neighborlist(structure, cutoff=cn_eeq.cutoff)
    n_pairs = int(nbl.mask.sum())

    # One untimed warm-up call per path: first-call allocator/thread-pool
    # setup has nothing to do with the O(nat^2)-vs-O(pairs) story below.
    cn_eeq(structure)
    cn_eeq(structure, pairs=nbl)

    dense_value, dense_time = timed_median(cn_eeq, structure)
    sparse_value, sparse_time = timed_median(cn_eeq, structure, pairs=nbl)

    # The actual point: two independently coded paths -- an O(nat^2) sum
    # over all pairs and an O(pairs) sum over a pre-filtered index list --
    # must agree to (near) floating-point precision, not just "look similar".
    max_abs_diff = (dense_value - sparse_value).abs().max()
    assert torch.allclose(
        dense_value, sparse_value, atol=1e-9
    ), f"{record}: dense and sparse CN disagree, max |Δ| = {max_abs_diff}"

    atom_counts.append(nat)
    dense_times.append(dense_time)
    sparse_times.append(sparse_time)

    print(
        f"{record:>12s}  {nat:5d}  {n_pairs:8d}  {dense_time * 1e3:11.4f}  "
        f"{sparse_time * 1e3:12.4f}  {dense_time / sparse_time:7.2f}x  "
        f"{float(max_abs_diff):9.2e}"
    )

# ---------------------------------------------------------------------------
# The scaling itself: dense grows with nat^2, sparse tracks the pair count
# ---------------------------------------------------------------------------
n_growth = atom_counts[-1] / atom_counts[0]
dense_growth = dense_times[-1] / dense_times[0]
sparse_growth = sparse_times[-1] / sparse_times[0]

print(
    f"\nfrom {atom_counts[0]} to {atom_counts[-1]} atoms ({n_growth:.1f}x more):"
)
print(
    f"  dense  time grew {dense_growth:6.1f}x  (nat^2 growth would be {n_growth**2:.1f}x)"
)
print(
    f"  sparse time grew {sparse_growth:6.1f}x  (nat   growth would be {n_growth:.1f}x)"
)
print(
    "\nall dense and sparse results agreed to within 1e-9 across every "
    f"one of the {len(records)} polyalanine records above"
)
