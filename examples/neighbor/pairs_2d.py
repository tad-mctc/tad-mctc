# SPDX-Identifier: CC0-1.0
"""
A 2D, visual walk through how `build_neighborlist` finds pairs, one stage
at a time (see `examples/neighbor/triples_2d.py` for the three-body
follow-up). Uses `get_structure("other", "C4H5NCS")` (the same real, planar
molecule -- pyridine-2(1H)-thione -- hardcoded in
`examples/cn/molecular_dense_single.py`), whose atoms all have `z = 0`, so
its 2D layout below is the exact geometry, not a projection.

This example shows internals: `Tiles` and `tile_pairs` live in the private
module `tad_mctc.neighbor._tiles` and may change without notice. Only
`build_neighborlist` is public API. As described in `tad_mctc.neighbor.list`,
a `NeighborList` is never built by testing every atom pair directly:

1. Atoms are grouped into spatially bounded tiles (`Tiles`, `tile=3` atoms
   here).
2. Tile *pairs* are screened by an exact bounding-box test (`tile_pairs`)
   -- cheap, and rules out most of the system before any atom-atom
   distance is computed at all.
3. Only atom pairs whose tiles survived that screen are checked exactly,
   giving the final, padded neighbour list.

The three panels below are exactly those three stages, all centred on the
same tile (the one holding atom 4, N) so the picture reads as one
progression rather than three unrelated views. Saved to `pairs_2d.png`
next to this script.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Circle, Rectangle

import tad_mctc as mctc
from tad_mctc.data.structures import get_structure
from tad_mctc.neighbor._tiles import Tiles, tile_pairs
from tad_mctc.neighbor.list import build_neighborlist

# Categorical slots 1/3/8 (blue/aqua/red) from the shared data-viz palette,
# plus its muted ink/grid roles -- the same fixed assignment
# `examples/scaling/glu_ala_plot.py` uses, not picked per plot.
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
PASS_COLOR = "#2a78d6"
FAIL_COLOR = "#e34948"
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
tile_size = 3

nbl = build_neighborlist(structure, cutoff=cutoff, tile=tile_size)
tiles = Tiles(positions, tile=tile_size)
tile_a, tile_b = tile_pairs(tiles, cutoff)
candidate_tile_pairs = {
    pair for pair in zip(tile_a.tolist(), tile_b.tolist())
} | {(b, a) for a, b in zip(tile_a.tolist(), tile_b.tolist())}

centre_atom = 4
reference_tile = next(
    t
    for t in range(tiles.ntile)
    if centre_atom in tiles.index[t][tiles.valid[t]].tolist()
)

real_i = nbl.idx_i[nbl.mask]
real_j = nbl.idx_j[nbl.mask]
edges = {tuple(sorted(pair)) for pair in zip(real_i.tolist(), real_j.tolist())}
neighbours_of_centre = {b for a, b in edges if a == centre_atom} | {
    a for a, b in edges if b == centre_atom
}

n_atoms = positions.shape[0]
n_possible_tile_pairs = tiles.ntile * (tiles.ntile + 1) // 2
n_candidate_tile_pairs = len(tile_a)
print(f"{n_atoms} atoms, cutoff = {cutoff} Bohr")
print(
    f"1. tiling:            {tiles.ntile} tiles (at most {tile_size} atoms each)"
)
print(
    f"2. tile-pair screen:  {n_candidate_tile_pairs} of {n_possible_tile_pairs} "
    f"tile pairs survive the bounding-box test"
)
print(f"3. exact check:       {len(edges)} real atom pairs")


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


def draw_tile_boxes(ax: Axes, edgecolors: dict[int, str] | None = None) -> None:
    for t in range(tiles.ntile):
        lo_x, lo_y = tiles.lo[t, 0].item(), tiles.lo[t, 1].item()
        hi_x, hi_y = tiles.hi[t, 0].item(), tiles.hi[t, 1].item()
        color = MUTED if edgecolors is None else edgecolors.get(t, MUTED)
        linewidth = 1.4 if edgecolors else 1.0

        if lo_x == hi_x and lo_y == hi_y:
            # The bounding box of a one-atom tile is a point, so mark the
            # atom with a ring in the tile's colour instead.
            ax.scatter(
                lo_x,
                lo_y,
                s=340,
                facecolor="none",
                edgecolor=color,
                linestyle=":",
                linewidth=linewidth,
                zorder=0,
            )
            continue

        ax.add_patch(
            Rectangle(
                (lo_x, lo_y),
                hi_x - lo_x,
                hi_y - lo_y,
                fill=False,
                linestyle=":",
                edgecolor=color,
                linewidth=linewidth,
                zorder=0,
            )
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

fig, (ax_tiles, ax_screen, ax_pairs) = plt.subplots(
    1, 3, figsize=(16, 6), facecolor="#fcfcfb"
)


# --- panel 1: tiling ------------------------------------------------------
ax_tiles.set_facecolor("#fcfcfb")
draw_tile_boxes(ax_tiles)
draw_atoms(ax_tiles)
ax_tiles.set_title(
    f"1. group atoms into tiles (≤{tile_size} atoms each)", color=INK
)
ax_tiles.text(
    sum(xlim) / 2,
    caption_y,
    f"{tiles.ntile} tiles from {n_atoms} atoms -- no distances computed yet",
    **caption_kwargs,
)


# --- panel 2: tile-pair screening around one tile -------------------------
ax_screen.set_facecolor("#fcfcfb")
tile_status = {
    t: (
        INK
        if t == reference_tile
        else (
            PASS_COLOR
            if (reference_tile, t) in candidate_tile_pairs
            else FAIL_COLOR
        )
    )
    for t in range(tiles.ntile)
}
draw_tile_boxes(ax_screen, edgecolors=tile_status)

centers = ((tiles.lo + tiles.hi) / 2)[:, :2]
ref_x, ref_y = centers[reference_tile].tolist()
for t in range(tiles.ntile):
    if t == reference_tile:
        continue
    tx, ty = centers[t].tolist()
    passed = (reference_tile, t) in candidate_tile_pairs
    ax_screen.plot(
        [ref_x, tx],
        [ref_y, ty],
        color=PASS_COLOR if passed else FAIL_COLOR,
        linestyle="-" if passed else "--",
        linewidth=1.4,
        zorder=1,
    )
    if not passed:
        lo_r, hi_r = tiles.lo[reference_tile], tiles.hi[reference_tile]
        lo_t, hi_t = tiles.lo[t], tiles.hi[t]
        gap = (lo_r - hi_t).clamp_min(0.0) + (lo_t - hi_r).clamp_min(0.0)
        ax_screen.annotate(
            f"gap {float(gap.norm()):.2f} > {cutoff}",
            ((ref_x + tx) / 2, (ref_y + ty) / 2),
            color=FAIL_COLOR,
            fontsize=7,
            ha="center",
            va="bottom",
            bbox={"facecolor": "#fcfcfb", "edgecolor": "none", "pad": 1},
        )
draw_atoms(ax_screen)
ax_screen.set_title("2. screen tile pairs by bounding-box gap", color=INK)
ax_screen.text(
    sum(xlim) / 2,
    caption_y,
    "blue = tile pair kept for the exact check; red dashed = ruled out\n"
    "by bounding-box distance alone, no atom pair ever tested",
    **caption_kwargs,
)


# --- panel 3: exact check -> final neighbour list --------------------------
ax_pairs.set_facecolor("#fcfcfb")
for a, b in edges:
    xa, ya = xy[a].tolist()
    xb, yb = xy[b].tolist()
    ax_pairs.plot([xa, xb], [ya, yb], color=GRID, linewidth=1.6, zorder=1)

cx, cy = xy[centre_atom].tolist()
ax_pairs.add_patch(
    Circle(
        (cx, cy),
        cutoff,
        fill=False,
        linestyle="--",
        edgecolor=MUTED,
        linewidth=1.2,
        zorder=2,
    )
)
draw_atoms(ax_pairs)
for atom in neighbours_of_centre:
    ax_pairs.scatter(
        *xy[atom].tolist(),
        s=340,
        facecolor="none",
        edgecolor=PASS_COLOR,
        linewidth=1.6,
        zorder=4,
    )
ax_pairs.scatter(
    [cx], [cy], s=420, facecolor="none", edgecolor=INK, linewidth=2, zorder=5
)
ax_pairs.set_title("3. exact per-atom check → neighbour list", color=INK)
ax_pairs.text(
    sum(xlim) / 2,
    caption_y,
    f"dashed circle = cutoff ({cutoff} Bohr) around atom {centre_atom} "
    f"({symbols[centre_atom]}); blue rings = atoms it is\n"
    f"actually paired with -- {len(neighbours_of_centre)} kept, the rest "
    f"outside cutoff despite nearby tiles",
    **caption_kwargs,
)

for ax in (ax_tiles, ax_screen, ax_pairs):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(GRID)

fig.suptitle(
    "tad_mctc.neighbor.build_neighborlist: tiles → tile-pair screen "
    "→ neighbour list",
    color=INK,
    y=0.99,
)
fig.tight_layout(rect=(0, 0, 1, 0.94))

out_path = Path(__file__).resolve().parent / "pairs_2d.png"
fig.savefig(out_path, dpi=150)
print(f"\nplot written to {out_path}")
