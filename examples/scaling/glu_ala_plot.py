# SPDX-Identifier: CC0-1.0
"""
Plots the timing data `examples/scaling/glu_ala.py` writes to
``glu_ala_scaling_<small|large>_<cpu|cuda>.txt``. Run that script first --
once for `--cpu` and, if you have a GPU, once more for `--gpu`, both with the
same `--small`/`--large` choice you pass here -- then run this one to plot
total sparse CN time against atom count on log-log axes. If both a CPU and a
CUDA file exist for the chosen ladder, both are drawn on the same axes; if
only one exists, that one is drawn alone.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent

# Categorical slots 1 (blue) and 2 (orange) from the shared data-viz palette:
# a fixed, colorblind-safe assignment, not picked per plot.
COLORS = {"cpu": "#2a78d6", "cuda": "#eb6834"}
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--small",
        action="store_true",
        help="Plot the glu_ala_a-only ladder (default).",
    )
    group.add_argument(
        "--large",
        action="store_true",
        help="Plot the combined glu_ala_a + glu_ala_b ladder.",
    )
    return parser.parse_args()


def load(path: Path) -> tuple[list[int], list[float]]:
    """`(nat, total_ms)` pairs from a `glu_ala.py`-written data file, in
    file order (already ascending by atom count)."""
    nat: list[int] = []
    total_ms: list[float] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        nat.append(int(fields[1]))
        total_ms.append(float(fields[5]))
    return nat, total_ms


args = parse_args()
ladder_name = "large" if args.large else "small"

data_paths = {
    device: SCRIPT_DIR / f"glu_ala_scaling_{ladder_name}_{device}.txt"
    for device in ("cpu", "cuda")
}
available = {
    device: path for device, path in data_paths.items() if path.is_file()
}
if not available:
    checked = ", ".join(str(p) for p in data_paths.values())
    raise SystemExit(
        f"No timing data found for --{ladder_name} (checked: {checked}).\n"
        f"Run `python examples/scaling/glu_ala.py --{ladder_name}` "
        "(add --gpu/--cuda for a second, CUDA file) first."
    )

fig, ax = plt.subplots(figsize=(6, 4.5), facecolor="#fcfcfb")
ax.set_facecolor("#fcfcfb")

for device, path in available.items():
    nat, total_ms = load(path)
    ax.plot(
        nat,
        total_ms,
        marker="o",
        markersize=5,
        linewidth=2,
        color=COLORS[device],
        label=device,
    )

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("atoms", color=INK)
ax.set_ylabel("sparse CN total time (ms)", color=INK)
ax.set_title(f"glu_ala scaling ({ladder_name})", color=INK)
ax.tick_params(colors=MUTED)
for spine in ax.spines.values():
    spine.set_color(GRID)
ax.grid(True, which="both", color=GRID, linewidth=0.8)

if len(available) > 1:
    ax.legend(frameon=False, labelcolor=INK)

fig.tight_layout()
out_path = SCRIPT_DIR / f"glu_ala_scaling_{ladder_name}.png"
fig.savefig(out_path, dpi=150)
print(f"plot written to {out_path}")
