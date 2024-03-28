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
Data: Getter Functions
======================

This module only contains some convenience functions for collecting constants
for a given set of atomic numbers.
"""

from __future__ import annotations

import torch

from ..typing import Tensor
from ..units import GMOL2AU
from . import mass, zeff

__all__ = ["get_atomic_masses", "get_ecore", "get_zvalence"]


def get_atomic_masses(
    numbers: Tensor,
    atomic_units: bool = True,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """
    Get isotope-averaged atomic masses for all `numbers`.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers for all atoms in the system.
    atomic_units : bool, optional
        Flag for unit conversion. If `True` (default), the atomic masses will
        be returned in atomic units. If `False`, the unit remains g/mol.
    device : torch.device | None, optional
        Device to store the tensor. If `None` (default), the default device is used.
    dtype : torch.dtype, optional
        Data type of the tensor. If `None` (default), the default dtype is used.

    Returns
    -------
    Tensor
        Atomic masses.
    """
    m = mass.ATOMIC.to(device=device, dtype=dtype)[numbers]
    return m * GMOL2AU if atomic_units is True else m


def get_zvalence(
    numbers: Tensor,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """
    Get charge of valence shell for all `numbers`.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers for all atoms in the system.
    device : torch.device | None, optional
        Device to store the tensor. If `None` (default), the default device is used.
    dtype : torch.dtype, optional
        Data type of the tensor. If `None` (default), the default dtype is used.

    Returns
    -------
    Tensor
        Charges of valence shell of atoms.
    """
    return zeff.ZVALENCE.to(device=device, dtype=dtype)[numbers]


def get_ecore(
    numbers: Tensor,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """
    Get number of core electrons for all `numbers`.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers for all atoms in the system.
    device : torch.device | None, optional
        Device to store the tensor. If `None` (default), the default device is used.
    dtype : torch.dtype, optional
        Data type of the tensor. If `None` (default), the default dtype is used.

    Returns
    -------
    Tensor
        Number of core electrons.
    """
    return zeff.ECORE.to(device=device, dtype=dtype)[numbers]
