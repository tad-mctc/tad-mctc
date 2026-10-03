# SPDX-Identifier: CC0-1.0
"""
`examples/cn/molecular_dense_single.py`'s EEQ-CN calculation batches over several
geometries in a single `torch.func.vmap` call, contrasted with the explicit
Python loop batching would otherwise need.
"""

import torch

import tad_mctc as mctc
from tad_mctc.io.structure import Structure

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

structure = Structure(numbers=numbers, positions=positions)

cn = mctc.ncoord.cn_eeq(structure)
print("CN (EEQ):")
print(cn)


# ---------------------------------------------------------------------------
# vmap: batch several jittered copies of the same molecule
# ---------------------------------------------------------------------------
torch.manual_seed(0)
NBATCH = 4
jitter = 0.05 * torch.randn(NBATCH, *positions.shape, dtype=positions.dtype)
positions_batch = positions.unsqueeze(0) + jitter


def cn_of_positions(p: torch.Tensor) -> torch.Tensor:
    return mctc.ncoord.cn_eeq(structure.replace(positions=p))


# Without `vmap`, batching over geometries needs an explicit Python loop:
# one `cn_eeq` call per geometry, with no shared tracing/dispatch.
looped = torch.stack([cn_of_positions(p) for p in positions_batch])

# `vmap` performs the same batching in a single call. Only `positions` is
# mapped over -- `numbers` (the atomic species) lives on the shared
# `structure`, which is the same for every copy in the batch.
batched = torch.vmap(cn_of_positions)(positions_batch)
print("\nvmap over 4 jittered geometries matches the Python loop:")
print(torch.allclose(batched, looped))
