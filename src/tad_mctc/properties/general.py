# This file is part of tad-mctc.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2024 Grimme Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Properties: General
===================

Collection of functions for the calculation of molecular properties.
"""

from __future__ import annotations

import torch

from .. import storch, units
from ..batch import eye, real_pairs
from ..convert import any_to_tensor
from ..data import pse
from ..math import einsum
from ..typing import DD, Tensor

__all__ = [
    "inertia_moment",
    "center_of_mass",
    "rot_consts",
    "enn",
    "sum_formula",
]


def enn(
    numbers: Tensor,
    positions: Tensor,
    cutoff: Tensor | float | int = 25.0,
) -> Tensor:
    """
    Calculate the nuclear repulsion energy.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers for all atoms in the system of shape ``(..., nat)``.
    positions : Tensor
        Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
    cutoff : Tensor | float | int, optional
        Cutoff distance for the nuclear repulsion energy. Defaults to
        ``25.0``.

    Returns
    -------
    Tensor
        Nuclear repulsion energy.
    """
    dd: DD = {"device": positions.device, "dtype": positions.dtype}
    zero = torch.tensor(0.0, **dd)
    cutoff = any_to_tensor(cutoff, **dd)

    mask = real_pairs(numbers, mask_diagonal=True)
    distances = storch.cdist(positions)

    charges = numbers.type(positions.dtype)
    charge_products = einsum("i,j->ij", charges, charges)

    within_cutoff = mask * (distances <= cutoff)
    contributions = torch.where(
        within_cutoff, storch.divide(charge_products, distances), zero
    )

    return 0.5 * torch.sum(contributions)


def sum_formula(numbers: Tensor) -> str:
    """
    Calculate the sum formula from atomic numbers.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers for all atoms in the system of shape ``(nat,)``.

    Returns
    -------
    str
        Sum formula.
    """
    formula = ""

    unique, counts = torch.unique(numbers, return_counts=True)
    for u, c in zip(unique, counts):
        formula += f"{pse.Z2S[int(u)]}"
        if c > 1:
            formula += f"{c}"

    return formula


def center_of_mass(masses: Tensor, positions: Tensor) -> Tensor:
    """
    Calculate the center of mass from the atomic coordinates and masses.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers for all atoms in the system of shape ``(..., nat)``.
    positions : Tensor
        Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).

    Returns
    -------
    Tensor
        Cartesian coordinates of center of mass of shape ``(..., 3)``.
    """
    s = storch.reciprocal(torch.sum(masses, dim=-1))
    return einsum("...z,...zx,...->...x", masses, positions, s)


def positions_rel_com(masses: Tensor, positions: Tensor) -> Tensor:
    """
    Calculate positions relative to the center of mass.

    Parameters
    ----------
    masses : Tensor
        Atomic masses for all atoms in the system (shape: ``(..., nat)``).
    positions : Tensor
        Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).

    Returns
    -------
    Tensor
        Cartesian coordinates relative to center of mass (shape:
        ``(..., nat, 3)``).
    """
    com = center_of_mass(masses, positions)
    return positions - com.unsqueeze(-2)


def inertia_moment(
    masses: Tensor,
    positions: Tensor,
    center_pa: bool = True,
    pos_already_com: bool = False,
) -> Tensor:
    """
    Calculate the inertia tensor of the molecule.

    Parameters
    ----------
    masses : Tensor
        Atomic masses for all atoms in the system (shape: ``(..., nat)``).
    positions : Tensor
        Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
    center_pa : bool, optional
        If ``True``, the tensor is centered relative to the principal axes,
        which prepares for rotational analysis. Defaults to ``True``.
    pos_already_com : bool, optional
        If ``True``, the positions are already centered at the center of mass.
        Defaults to ``False``.

    Returns
    -------
    Tensor
        Inertia tensor of shape ``(..., 3, 3)``.
    """
    if pos_already_com is False:
        positions = positions_rel_com(masses, positions)

    im = einsum("...m,...mx,...my->...xy", masses, positions, positions)

    if center_pa is True:
        # trace = einsum("...ii->...", im)
        # einsum("...ij,...->...ij", eye(im), trace) - im)
        return einsum("...ij,...kk->...ij", eye(im.shape), im) - im
    return im


# TODO: Check against reference values
# https://github.com/psi4/psi4/blob/3c2be0144a850eaad3b428ceabc58ff38a163fde/psi4/src/psi4/libmints/molecule.cc#L1353
# https://github.com/pyscf/pyscf/blob/master/pyscf/hessian/thermo.py#L111
def rot_consts(masses: Tensor, positions: Tensor) -> Tensor:  # pragma: no cover
    r"""
    Calculate the rotational constants from the inertia tensor.

    .. math::

        B = \frac{h}{8 \pi^2 c I} = \frac{\hbar}{4 \pi c I}

    Parameters
    ----------
    masses : Tensor
        Atomic masses for all atoms in the system (shape: ``(..., nat)``).
    positions : Tensor
        Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).

    Returns
    -------
    Tensor
        Rotational constants of shape ``(..., 3)``.
    """
    im = inertia_moment(masses, positions, center_pa=True)

    # Eigendecomposition yields the principal moments of inertia (w)
    # and the principal axes of rotation (_) of a molecule.
    w, _ = storch.eighb(im)

    # rotational constant in atomic units
    c_au = units.CODATA.c * (units.METER2AU / units.SECOND2AU)
    b = storch.reciprocal(4 * torch.pi * c_au * w)  # hbar = 1

    return torch.where(w > 1e-6, b, torch.zeros_like(b))
