# SPDX-Identifier: CC0-1.0
"""
A 2D, visual walk through how `triples_from_neighborlist` turns a neighbour list
into three-body candidates, one stage at a time -- the three-body follow-up
to `examples/neighbor/pairs_2d.py`, which builds the neighbour list this script
starts from. Uses `get_structure("other", "C4H5NCS")` (the same real, planar
molecule -- pyridine-2(1H)-thione -- hardcoded in
`examples/cn/molecular_dense_single.py`), whose atoms all have `z = 0`, so its
2D layout below is the exact geometry, not a projection.

For one chosen centre atom `j`:

1. Its neighbours come straight from the neighbour list -- no new geometry.
   Each stored pair is oriented from its lower atom to its higher one, so
   `j` keeps only the neighbours above it; the ones below it see `j` as
   *their* upward neighbour instead. This is what makes every triple come
   out exactly once, under its lowest atom.
2. Every unordered pair of those upward neighbours `(i, k)` is a
   *candidate* triple. This is pure combinatorics: nothing about `i-k` has
   been checked yet.
3. A candidate is kept only if its third side `i-k`, never implied by the
   neighbour list at all, also falls inside `cutoff`.

All three panels are centred on the same atom (atom 3, C) so the picture
reads as one progression. Saved to `triples_2d.png` next to this script.
"""

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Polygon

import tad_mctc as mctc
from tad_mctc.data.structures import get_structure
from tad_mctc.neighbor.list import build_neighborlist
from tad_mctc.neighbor.triples import triples_from_neighborlist

# Categorical slots 1/2 (blue/orange) from the shared data-viz palette,
# plus its muted ink/grid roles -- the same fixed assignment
# `examples/scaling/glu_ala_plot.py` uses, not picked per plot.
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
KEPT_COLOR = "#2a78d6"
REJECTED_COLOR = "#eb6834"
ELEMENT_COLORS = {"C": "#2a78d6", "N": "#eb6834", "S": "#1baf7a", "H": MUTED}

structure = get_structure("other", "C4H5NCS")
numbers = structure.numbers
positions = structure.positions
symbols = mctc.convert.number_to_symbol(numbers)
xy = positions[:, :2]

# The molecule is planar (every `z == 0`), so this really is the raw
# geometry, not a projection chosen to make the picture work.
assert bool((positions[:, 2] == 0).all())

cutoff = 4.65
nbl = build_neighborlist(structure, cutoff=cutoff, tile=3)

real_i = nbl.idx_i[nbl.mask]
real_j = nbl.idx_j[nbl.mask]
edges = {tuple(sorted(pair)) for pair in zip(real_i.tolist(), real_j.tolist())}

# `triples_from_neighborlist` is a chunked generator by contract (see its
# docstring): a caller must concatenate every chunk, never just the first.
# A molecule has no image shifts, so `chunk.shift_i`/`shift_k` are all zero.
kept_by_centre: dict[int, set[tuple[int, int]]] = {}
for chunk in triples_from_neighborlist(nbl, structure, cutoff):
    for i, j, k in zip(
        chunk.idx_i.tolist(), chunk.idx_j.tolist(), chunk.idx_k.tolist()
    ):
        kept_by_centre.setdefault(j, set()).add(tuple(sorted((i, k))))

# `j` the centre atom, with a middling upward neighbour count (not so few
# that step 2 is a single line, not so many that it is unreadable).
centre_atom = 3
neighbours_of_centre = sorted(
    {b for a, b in edges if a == centre_atom}
    | {a for a, b in edges if b == centre_atom}
)
# `edges` are sorted pairs, so `a < b`: the pair is oriented from `a` up to
# `b`, and only `b` is an upward neighbour of `a`.
upward_of_centre = sorted(b for a, b in edges if a == centre_atom)
downward_of_centre = sorted(a for a, b in edges if b == centre_atom)
candidate_pairs = sorted(
    tuple(sorted(p)) for p in combinations(upward_of_centre, 2)
)
kept_triples = sorted(kept_by_centre.get(centre_atom, set()))
rejected_triples = sorted(set(candidate_pairs) - set(kept_triples))

n_atoms = positions.shape[0]
n_unique_triples = sum(len(v) for v in kept_by_centre.values())
print(f"{n_atoms} atoms, cutoff = {cutoff} Bohr")
print(f"pairs (real, unique):           {len(edges)}")
print(f"triples (unique, all centres):  {n_unique_triples}")
print(
    f"1. neighbours of atom {centre_atom} ({symbols[centre_atom]}): "
    f"{len(neighbours_of_centre)}, {len(upward_of_centre)} of them above it"
)
print(f"2. candidate (i, k) pairs:      {len(candidate_pairs)}")
print(
    f"3. kept vs rejected:            {len(kept_triples)} kept, "
    f"{len(rejected_triples)} rejected (third side i-k outside cutoff)"
)


# ---------------------------------------------------------------------------
# Shared drawing helpers
# ---------------------------------------------------------------------------
def draw_atoms(ax: Axes) -> None:
    for idx, (x, y) in enumerate(xy.tolist()):
        color = ELEMENT_COLORS[symbols[idx]]
        ax.scatter(
            x, y, s=220, color=color, edgecolor=INK, linewidth=0.8, zorder=3
        )
        ax.annotate(
            symbols[idx],
            (x, y),
            color="white" if color != MUTED else INK,
            ha="center",
            va="center",
            fontsize=7,
            fontweight="bold",
            zorder=4,
        )


def draw_pair_list(ax: Axes) -> None:
    for a, b in edges:
        xa, ya = xy[a].tolist()
        xb, yb = xy[b].tolist()
        ax.plot([xa, xb], [ya, yb], color=GRID, linewidth=1.2, zorder=1)


def mark_centre_and_neighbours(ax: Axes) -> None:
    for atom in upward_of_centre:
        ax.scatter(
            *xy[atom].tolist(),
            s=340,
            facecolor="none",
            edgecolor=KEPT_COLOR,
            linewidth=1.6,
            zorder=4,
        )
    for atom in downward_of_centre:
        ax.scatter(
            *xy[atom].tolist(),
            s=340,
            facecolor="none",
            edgecolor=MUTED,
            linestyle="--",
            linewidth=1.2,
            zorder=4,
        )
    ax.scatter(
        *xy[centre_atom].tolist(),
        s=420,
        facecolor="none",
        edgecolor=INK,
        linewidth=2,
        zorder=5,
    )


margin = 2.0
caption_pad = 1.6
xlim = (xy[:, 0].min().item() - margin, xy[:, 0].max().item() + margin)
ylim = (
    xy[:, 1].min().item() - margin - caption_pad,
    xy[:, 1].max().item() + margin,
)
caption_y = ylim[0] + 0.5
caption_kwargs = {
    "color": MUTED,
    "ha": "center",
    "fontsize": 8,
    "bbox": {"facecolor": "#fcfcfb", "edgecolor": "none", "pad": 2},
}

fig, (ax_neigh, ax_cand, ax_trip) = plt.subplots(
    1, 3, figsize=(16, 6), facecolor="#fcfcfb"
)


# --- panel 1: j's neighbours, straight from the neighbour list ------------
ax_neigh.set_facecolor("#fcfcfb")
draw_pair_list(ax_neigh)
draw_atoms(ax_neigh)
mark_centre_and_neighbours(ax_neigh)
ax_neigh.set_title("1. neighbours of j (from the neighbour list)", color=INK)
ax_neigh.text(
    sum(xlim) / 2,
    caption_y,
    f"black ring = centre atom j = {centre_atom} ({symbols[centre_atom]}); "
    f"blue rings = its {len(upward_of_centre)} neighbours above it;\n"
    f"dashed grey = the {len(downward_of_centre)} below it, whose triples "
    f"with j have a lower centre",
    **caption_kwargs,
)


# --- panel 2: every candidate (i, k) pair, unfiltered ----------------------
ax_cand.set_facecolor("#fcfcfb")
draw_pair_list(ax_cand)
for i, k in candidate_pairs:
    xi, yi = xy[i].tolist()
    xk, yk = xy[k].tolist()
    ax_cand.plot(
        [xi, xk], [yi, yk], color=MUTED, linestyle="--", linewidth=1.0, zorder=2
    )
draw_atoms(ax_cand)
mark_centre_and_neighbours(ax_cand)
ax_cand.set_title("2. every candidate pair (i, k)", color=INK)
ax_cand.text(
    sum(xlim) / 2,
    caption_y,
    f"dashed lines = all C({len(upward_of_centre)}, 2) = "
    f"{len(candidate_pairs)} unordered upward neighbour pairs;\n"
    f"pure combinatorics -- the i-k distance is not checked yet",
    **caption_kwargs,
)


# --- panel 3: keep only if i-k is also inside cutoff -----------------------
ax_trip.set_facecolor("#fcfcfb")
draw_pair_list(ax_trip)

for i, k in kept_triples:
    triangle = [xy[i].tolist(), xy[centre_atom].tolist(), xy[k].tolist()]
    ax_trip.add_patch(
        Polygon(
            triangle,
            closed=True,
            facecolor=KEPT_COLOR,
            edgecolor=KEPT_COLOR,
            alpha=0.18,
            linewidth=0.8,
            zorder=2,
        )
    )

# One rejected candidate, drawn unfilled and dashed, with its too-long
# `i-k` side labelled -- the check the neighbour list alone never makes.
rej_i, rej_k = rejected_triples[0]
rej_triangle = [
    xy[rej_i].tolist(),
    xy[centre_atom].tolist(),
    xy[rej_k].tolist(),
]
ax_trip.add_patch(
    Polygon(
        rej_triangle,
        closed=True,
        fill=False,
        edgecolor=REJECTED_COLOR,
        linestyle="--",
        linewidth=1.2,
        zorder=2,
    )
)
rix, riy = xy[rej_i].tolist()
rkx, rky = xy[rej_k].tolist()
d_ik = float((xy[rej_i] - xy[rej_k]).norm())
# Two short lines, offset above the side: one long line is wider than the
# side itself and would run under the atoms at its ends.
ax_trip.annotate(
    f"i-k = {d_ik:.2f} > {cutoff}\nrejected",
    ((rix + rkx) / 2, (riy + rky) / 2),
    xytext=(0, 8),
    textcoords="offset points",
    color=REJECTED_COLOR,
    fontsize=7,
    ha="center",
    va="bottom",
    bbox={"facecolor": "#fcfcfb", "edgecolor": "none", "pad": 1},
    zorder=6,
)

draw_atoms(ax_trip)
mark_centre_and_neighbours(ax_trip)
ax_trip.set_title("3. keep only if i-k ≤ cutoff → triples", color=INK)
ax_trip.text(
    sum(xlim) / 2,
    caption_y,
    f"{len(kept_triples)} filled triangles = kept triples; dashed outline = "
    f"1 of {len(rejected_triples)} rejected\n"
    f"candidates, whose third side falls outside the cutoff",
    **caption_kwargs,
)

for ax in (ax_neigh, ax_cand, ax_trip):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(GRID)

fig.suptitle(
    "tad_mctc.neighbor.triples_from_neighborlist: neighbours → candidate "
    "pairs → triples",
    color=INK,
    y=0.99,
)
fig.tight_layout(rect=(0, 0, 1, 0.94))

out_path = Path(__file__).resolve().parent / "triples_2d.png"
fig.savefig(out_path, dpi=150)
print(f"\nplot written to {out_path}")
