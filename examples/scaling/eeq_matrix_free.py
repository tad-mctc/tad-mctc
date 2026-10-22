# SPDX-Identifier: CC0-1.0
"""
Cost of the stages of a matrix-free charge equilibration (EEQ) solve.

Water droplets of increasing size; the operator is the float64 fused pair
product, refined from float32 products (``fast_matvec``), with the balancing
preconditioner on 12 bohr cells (not bounded with ``max_cells`` here, so
the setup memory grows like ``n^2``). Each stage is timed separately; on
GPU, its memory is the peak above what was allocated when it started:

- ``compile``: the first fused products. A kernel compiles once per dtype
  for all system sizes, so this is only a compilation for the first size
  (Inductor caches it on disk, across runs); later, it is one product.
- ``setup``: the cells, the cell sums A Z and the preconditioner.
- ``charges``: the refined constrained solve.
- ``forces``: the gradient of the energy with respect to the positions
  (an adjoint solve and the derivatives of the pair products).
- ``energy params``: the gradient of the energy with respect to the
  parameters (radii, hardnesses, electronegativities).
- ``charge params``: the gradient of a function of the charges,
  ``sum(q^2)``, with respect to the parameters (the charge response).
- ``force params``: the parameter gradient of a function of the forces,
  the mixed derivative of a fit to forces. Its graph is that of all pair
  tiles, ``O(n^2)`` memory, so it runs only up to ``MIXED_MAX`` atoms.

The derivatives evaluate the pair products eagerly in float64 (see
:mod:`tad_mctc.storch.pairwise`), so they are slower than the solve.

Usage: ``python eeq_matrix_free.py [sizes...]``; the default sizes are 1k,
5k and 10k atoms on GPU, 300 and 1k on CPU.
"""

from __future__ import annotations

import math
import sys
import time

import torch

from tad_mctc.storch.linsolve import balancing, cell_index, constrained_solve
from tad_mctc.storch.pairwise import pair_cell_sums, pair_product

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dd = {"device": device, "dtype": torch.double}
fused = device.type == "cuda"

MIXED_MAX = 3000
"""Largest system for the mixed derivative (``O(n^2)`` memory)."""

# radius (bohr), hardness and electronegativity (hartree) of H and O
PARAMS = {"H": (0.84, 0.47, 1.24), "O": (1.24, 0.17, 1.54)}


def kernel(r, pi, pj):
    return torch.erf(r * torch.rsqrt(pi[0] ** 2 + pj[0] ** 2)) / r


def diagonal(alpha, eta):
    return eta + math.sqrt(2 / math.pi) / alpha


def exact(q, pos, alpha, eta):
    pairs = pair_product(kernel, pos, q, alpha, fused=fused)
    return diagonal(alpha, eta) * q + pairs


def fast(q, pos, alpha, eta):
    pairs = pair_product(kernel, pos, q, alpha, dtype=torch.float, fused=fused)
    return diagonal(alpha, eta) * q + pairs


def droplet(molecules: int, seed: int = 0):
    """Water molecules on a lattice of liquid density, in a sphere."""
    gen = torch.Generator().manual_seed(seed)
    spacing = 5.86  # bohr, about 1 g/cm^3
    side = math.ceil((6 * molecules / math.pi) ** (1 / 3)) + 2
    g = torch.arange(side, dtype=torch.double) - (side - 1) / 2
    grid = torch.stack(torch.meshgrid(g, g, g, indexing="ij"), -1).reshape(
        -1, 3
    )
    oxygens = spacing * grid[grid.norm(dim=-1).argsort()[:molecules]]

    # O-H 1.81 bohr, H-O-H 104.5 degrees, random orientations
    half = math.radians(104.5) / 2
    local = 1.81 * torch.tensor(
        [
            [math.sin(half), math.cos(half), 0.0],
            [-math.sin(half), math.cos(half), 0.0],
        ],
        dtype=torch.double,
    )
    rot = torch.linalg.qr(
        torch.randn(molecules, 3, 3, generator=gen, dtype=torch.double)
    )[0]
    hydrogens = oxygens.unsqueeze(1) + local @ rot.mT
    positions = torch.cat((oxygens.unsqueeze(1), hydrogens), 1).reshape(-1, 3)

    symbols = ["O", "H", "H"] * molecules
    alpha, eta, chi = (
        torch.tensor([PARAMS[s][k] for s in symbols], dtype=torch.double)
        for k in range(3)
    )
    return tuple(t.to(device) for t in (positions, alpha, eta, chi))


def stage(name: str, row: dict, fn):
    base = 0
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
    start = time.perf_counter()
    out = fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() - base
        row[f"{name} GB"] = f"{peak / 1e9:.2f}"
    row[f"{name} s"] = f"{time.perf_counter() - start:.2f}"
    return out


def run(atoms: int) -> dict:
    positions, alpha, eta, chi = droplet(atoms // 3)
    n = positions.shape[0]
    row: dict = {"atoms": n}
    constraint = torch.ones(1, n, **dd)
    total = torch.zeros(1, **dd)

    def compile_products():
        exact(chi, positions, alpha, eta)
        fast(chi, positions, alpha, eta)

    stage("compile", row, compile_products)

    def setup():
        cells, k = cell_index(positions, 12.0)
        products = pair_cell_sums(
            kernel, positions, cells, k, alpha, dtype=torch.float
        )
        diag = diagonal(alpha, eta)
        products = products.scatter_add(
            -1, cells.unsqueeze(-1), diag.float().unsqueeze(-1)
        )
        row["cells"] = k
        return balancing(diag, cells, products)

    prec = stage("setup", row, setup)

    pos = positions.clone().requires_grad_(True)
    params = tuple(t.clone().requires_grad_(True) for t in (alpha, eta, chi))

    def charges():
        result = constrained_solve(
            exact,
            -params[2],
            constraint,
            total,
            pos,
            params[0],
            params[1],
            preconditioner=prec,
            fast_matvec=fast,
            rtol=1e-12,
        )
        row["iterations"] = int(result.iterations)
        row["converged"] = bool(result.converged)
        row["charge error"] = f"{float(result.constraint_residual):.0e}"
        return result.solution

    q = stage("charges", row, charges)

    def energy():
        pairs = exact(q, pos, params[0], params[1])
        return (params[2] * q).sum() + 0.5 * (q * pairs).sum()

    # the graphs are kept for the later stages
    e = energy()
    stage("forces", row, lambda: torch.autograd.grad(e, pos, retain_graph=True))
    stage(
        "energy params",
        row,
        lambda: torch.autograd.grad(e, params, retain_graph=True),
    )
    stage(
        "charge params",
        row,
        lambda: torch.autograd.grad((q * q).sum(), params, retain_graph=True),
    )

    def force_params():
        (f,) = torch.autograd.grad(energy(), pos, create_graph=True)
        return torch.autograd.grad((f * f).sum(), params)

    if n <= MIXED_MAX:
        stage("force params", row, force_params)
    return row


if __name__ == "__main__":
    default = [1000, 5000, 10000] if device.type == "cuda" else [300, 1000]
    sizes = [int(a) for a in sys.argv[1:]] or default
    print(f"device: {device}")
    for atoms in sizes:
        print(run(atoms), flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()
