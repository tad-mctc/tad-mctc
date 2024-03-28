# This file is part of tad-mctc.
#
# SPDX-Identifier: LGPL-3.0
# Copyright (C) 2023 Marvin Friede
#
# tad-mctc is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# tad_mctc is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with tad-mctc. If not, see <https://www.gnu.org/licenses/>.
"""
Molecule: Property
==================

Collection of functions for the calculation of molecular properties.
"""

from __future__ import annotations

import torch

from .. import storch, units
from ..batch import eye
from ..math import einsum
from ..typing import Tensor

__all__ = ["inertia_moment", "mass_center", "rot_consts"]


def mass_center(masses: Tensor, positions: Tensor) -> Tensor:
    """
    Calculate the center of mass from the atomic coordinates and masses.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers for all atoms in the system of shape `(..., nat)`.
    positions : Tensor
        Cartesian coordinates of all atoms of shape `(..., nat, 3)`.

    Returns
    -------
    Tensor
        Cartesian coordinates of center of mass of shape `(..., 3)`.
    """
    s = storch.reciprocal(torch.sum(masses, dim=-1))
    return einsum("...z,...zx,...->...x", masses, positions, s)


def positions_rel_com(masses: Tensor, positions: Tensor) -> Tensor:
    """
    Calculate positions relative to the center of mass.

    Parameters
    ----------
    masses : Tensor
        Atomic masses for all atoms in the system (shape: `(..., nat)`).
    positions : Tensor
        Cartesian coordinates of all atoms (shape: `(..., nat, 3)`).

    Returns
    -------
    Tensor
        Cartesian coordinates relative to center of mass (shape:
        `(..., nat, 3)`).
    """
    com = mass_center(masses, positions)
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
        Atomic masses for all atoms in the system (shape: `(..., nat)`).
    positions : Tensor
        Cartesian coordinates of all atoms (shape: `(..., nat, 3)`).
    center_pa : bool, optional
        If `True`, the tensor is centered relative to the principal axes, which
        prepares for rotational analysis. Defaults to `True`.
    pos_already_com : bool, optional
        If `True`, the positions are already centered at the center of mass.
        Defaults to `False`.

    Returns
    -------
    Tensor
        Inertia tensor of shape `(..., 3, 3)`.
    """
    if pos_already_com is False:
        positions = positions_rel_com(masses, positions)

    im = einsum("...m,...mx,...my->...xy", masses, positions, positions)

    if center_pa is True:
        # trace = einsum("...ii->...", im)
        # einsum("...ij,...->...ij", eye(im), trace) - im)
        return einsum("...ij,...kk->...ij", eye(im), im) - im
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
        Atomic masses for all atoms in the system (shape: `(..., nat)`).
    positions : Tensor
        Cartesian coordinates of all atoms (shape: `(..., nat, 3)`).

    Returns
    -------
    Tensor
        Rotational constants of shape `(..., 3)`.
    """
    im = inertia_moment(masses, positions, center_pa=True)

    # Eigendecomposition yields the principal moments of inertia (w)
    # and the principal axes of rotation (_) of a molecule.
    w, _ = storch.eighb(im)

    # rotational constant in atomic units
    c_au = units.CODATA.c * (units.METER2AU / units.SECOND2AU)
    b = storch.reciprocal(4 * torch.pi * c_au * w)  # hbar = 1

    return torch.where(w > 1e-6, b, torch.zeros_like(b))
