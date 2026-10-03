# SPDX-Identifier: CC0-1.0
"""
`examples/cn/molecular_dense_single.py`'s EEQ-CN calculation is differentiable to arbitrary
order via `torch.func.jacrev`. This example takes the full per-atom Jacobian
and checks it against a hand-rolled central-difference (numerical) Jacobian,
so the analytical result is verified against an independent method rather
than merely printed.
"""

from collections.abc import Callable

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
# Analytical gradient: jacrev with respect to positions
# ---------------------------------------------------------------------------
# Do NOT differentiate `cn(...).sum()`: the total coordination number is
# translationally invariant (shifting every atom by the same vector leaves
# every pairwise distance, hence every count, unchanged), so
# `jacrev(lambda p: cn_eeq(structure.replace(positions=p)).sum())` is identically zero -- it
# would look like a working gradient demo whether or not the underlying
# derivative is actually being computed correctly. Taking the full,
# unsummed per-atom Jacobian avoids the trap.
def cn_of_positions(p: torch.Tensor) -> torch.Tensor:
    return mctc.ncoord.cn_eeq(structure.replace(positions=p))


analytical_jacobian = mctc.autograd.jacrev(cn_of_positions)(positions)
print(
    "\nd(CN)/d(positions) via jacrev, shape (nat, nat, 3):",
    tuple(analytical_jacobian.shape),
)


# ---------------------------------------------------------------------------
# Numerical gradient: central differences, as an independent cross-check
# ---------------------------------------------------------------------------
def numerical_jacobian(
    f: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    step: float = 1.0e-6,
) -> torch.Tensor:
    """
    Central-difference Jacobian of `f` at `x`, matching `jacrev`'s output
    shape `(*f(x).shape, *x.shape)`. One column per scalar entry of `x`,
    two forward evaluations (`+step`, `-step`) per column.
    """
    baseline_shape = f(x).shape
    flat_x = x.reshape(-1)

    columns = []
    for entry in range(flat_x.numel()):
        offset = torch.zeros_like(flat_x)
        offset[entry] = step

        x_plus = (flat_x + offset).reshape(x.shape)
        x_minus = (flat_x - offset).reshape(x.shape)

        derivative = (f(x_plus) - f(x_minus)) / (2.0 * step)
        columns.append(derivative)

    return torch.stack(columns, dim=-1).reshape(*baseline_shape, *x.shape)


numerical_jac = numerical_jacobian(cn_of_positions, positions)

max_abs_diff = (analytical_jacobian - numerical_jac).abs().max().item()
print("\nmax |jacrev - central_difference| :", max_abs_diff)
print(
    "jacrev matches the numerical Jacobian:",
    torch.allclose(analytical_jacobian, numerical_jac, atol=1.0e-5),
)
