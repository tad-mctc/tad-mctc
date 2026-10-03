# SPDX-Identifier: CC0-1.0
"""
`NeighborList` is the padded neighbour list `CNModel` takes as `pairs` (and any
other pairwise sum built on `idx_i`/`idx_j`) consumes instead of the dense,
all-pairs evaluation. This example is independent of `examples/cn/*.py`: it
covers what the list itself looks like and the one concrete benefit of
building it with `skin` -- reusing it across several steps instead of
rebuilding on every one.
"""

import torch

from tad_mctc.data.structures import get_structure
from tad_mctc.ncoord import cn_eeq
from tad_mctc.neighbor.list import build_neighborlist

structure = get_structure("other", "vancoh2")
positions = structure.positions
print(f"vancoh2: {positions.shape[0]} atoms")


# ---------------------------------------------------------------------------
# 1. What a `NeighborList` is: a padded, fixed-capacity neighbour list
# ---------------------------------------------------------------------------
# Construction is data-dependent (which pairs exist depends on the actual
# geometry: `nonzero`, boolean-mask indexing, a Python-level padding
# decision) and always runs under `torch.no_grad()`. Every downstream
# *consumption* of the resulting list, in contrast, is fixed-shape tensor
# algebra with a static shape and is fully differentiable -- see
# `tad_mctc.ncoord.common`'s sparse path for that half.
cutoff = 8.0
nbl = build_neighborlist(structure, cutoff=cutoff)

capacity = nbl.idx_i.shape[0]
n_real_pairs = int(nbl.mask.sum())
print("\n--- construction ---")
print(f"idx_i/idx_j shape (capacity):  {tuple(nbl.idx_i.shape)}")
print(f"real pairs within cutoff:      {n_real_pairs}")
print(f"padding slots:                 {capacity - n_real_pairs}")
print(f"cutoff this list was built for: {nbl.cutoff}")


# ---------------------------------------------------------------------------
# 2./3. The reuse benefit: build once with `skin`, reuse across MD-like steps
# ---------------------------------------------------------------------------
# `skin` searches a little further than `cutoff` when the list is built, so
# a small drift in `positions` does not immediately invalidate it: as long
# as no atom has moved more than `skin / 2` since the last build (checked by
# `nbl.stale(structure)`), the existing list still holds every pair within
# `cutoff`, just with a few extra (harmless) ones a bit beyond it. Only
# rebuild when `stale()` says so; naively rebuilding on every single step
# would pay the full `O(nat)`-ish construction cost every time instead of
# only on the occasional step that actually needs it.
skin = 0.5
torch.manual_seed(0)

nbl = build_neighborlist(structure, cutoff=cutoff, skin=skin)
current_positions = positions.clone()

# `cn_eeq.cutoff` (25.0 Bohr, EEQ's own reference cutoff) is larger than
# `nbl.cutoff` (8.0), and `CNModel` rejects a `pairs` list smaller
# than its own cutoff. `CNModel.replace` gives a short-cutoff EEQ-CN that
# actually matches this list -- illustrating list reuse, not the reference
# EEQ-CN value (that one is `examples/cn/molecular_dense_single.py`'s job).
cn_model = cn_eeq.replace(cutoff=cutoff)

n_steps = 12
step_sigma = 0.05  # Bohr, per atom, per step -- small compared to skin / 2
n_rebuilds = 0

print("\n--- MD-like steps, skin = 0.5 Bohr ---")
for step in range(1, n_steps + 1):
    current_positions = current_positions + step_sigma * torch.randn_like(
        current_positions
    )

    current = structure.replace(positions=current_positions)

    if bool(nbl.stale(current)):
        nbl = build_neighborlist(current, cutoff=cutoff, skin=skin)
        n_rebuilds += 1
        status = "rebuilt"
    else:
        status = "reused"

    # The actual point: every step evaluates CN from the (possibly reused)
    # list, via `CNModel`'s neighbour-list path.
    cn = cn_model(current, pairs=nbl)
    print(f"step {step:2d}: {status:7s}  mean CN = {cn.mean().item():.6f}")

print(f"\nbuilt {n_rebuilds} times over {n_steps} steps")
print(
    f"(naively rebuilding every step would have built {n_steps} times "
    f"instead of {n_rebuilds})"
)
