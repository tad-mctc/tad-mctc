# SPDX-Identifier: CC0-1.0
"""
`examples/cn/molecular_dense_single.py`'s EEQ-CN calculation is
`torch.compile(fullgraph=True)`-clean on the dense (no neighbour list) path
used here. This example compiles it, checks the compiled result against eager, and
times the first compiled call against later ones -- the whole point of
`torch.compile` is that only the first call pays tracing/compilation cost.

See `examples/neighbor/list.py` for the sparse, neighbour-list-based path and what a
`NeighborList` buys on top of the dense evaluation used below.
"""

import argparse
import time
from typing import Callable

import torch

import tad_mctc as mctc
from tad_mctc.io.structure import Structure


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

numbers = mctc.convert.symbol_to_number("C C C C N C S H H H H H".split())

# coordinates in Bohr (same molecule as `examples/cn/molecular_dense_single.py`)
positions = torch.tensor(
    [
        [-2.56745685564671, -0.02509985979910, 0.00000000000000],
        [-1.39177582455797, +2.27696188880014, 0.00000000000000],
        [+1.27784995624894, +2.45107479759386, 0.00000000000000],
        [+2.62801937615793, +0.25927727028120, 0.00000000000000],
        [+1.41097033661123, -1.99890996077412, 0.00000000000000],
        [-1.17186102298849, -2.34220576284180, 0.00000000000000],
        [-2.39505990368378, -5.22635838332362, 0.00000000000000],
        [+2.41961980455457, -3.62158019253045, 0.00000000000000],
        [-2.51744374846065, +3.98181713686746, 0.00000000000000],
        [+2.24269048384775, +4.24389473203647, 0.00000000000000],
        [+4.66488984573956, +0.17907568006409, 0.00000000000000],
        [-4.60044244782237, -0.17794734637413, 0.00000000000000],
    ],
    dtype=torch.double,
)

structure = Structure(numbers=numbers, positions=positions).to(device)
positions_dev = structure.positions


def cn_call(p: torch.Tensor) -> torch.Tensor:
    return mctc.ncoord.cn_eeq(structure.replace(positions=p))


def timed_call(
    fn: Callable[[torch.Tensor], torch.Tensor], p: torch.Tensor
) -> tuple[torch.Tensor, float]:
    """Time one call, synchronizing CUDA first (launches are async, so an
    unsynchronized timing would make even an uncompiled call look
    artificially fast; a no-op on CPU)."""
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    result = fn(p)

    if device.type == "cuda":
        torch.cuda.synchronize()

    end = time.perf_counter()
    return result, end - start


# One untimed warm-up call: on CUDA, the very first call in a process also
# pays one-off context/kernel-load cost that has nothing to do with eager
# vs. compiled execution, and would otherwise dwarf every timing below.
cn_call(positions_dev)

# Reset Dynamo's cache so an earlier import- or test-time compilation of
# `cn_eeq` cannot masquerade as the "already compiled" case below. No public
# equivalent exists.
torch._dynamo.reset()  # pylint: disable=protected-access
compiled_cn_call = torch.compile(cn_call, fullgraph=True, dynamic=False)

eager_value, eager_time = timed_call(cn_call, positions_dev)

# The first compiled call pays Dynamo's tracing/compilation cost on top of
# actually running the graph, so it is expected to be the slowest of the
# timings printed below -- that cost, not a faster eager call, is the
# reason it looks worse than "eager".
first_compiled_value, first_compiled_time = timed_call(
    compiled_cn_call, positions_dev
)
assert torch.allclose(first_compiled_value, eager_value)

print(f"eager call:          {eager_time:.6f} s")
print(
    f"first compiled call: {first_compiled_time:.6f} s "
    "(includes tracing + compilation)"
)

# Later calls reuse the already-compiled graph as long as the input shape
# stays the same (`dynamic=False` requires this) -- perturbing positions
# without changing their shape is exactly that, and should be markedly
# faster than the first compiled call above.
for step in range(1, 4):
    jittered_positions = positions_dev + 0.01 * torch.randn_like(positions_dev)
    _, reused_time = timed_call(compiled_cn_call, jittered_positions)
    print(f"compiled call #{step}:  {reused_time:.6f} s (reused graph)")
