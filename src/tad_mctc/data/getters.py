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
    Get isotope-averaged atomic masses for all ``numbers``.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers for all atoms in the system of shape ``(..., nat)``.
    atomic_units : bool, optional
        Flag for unit conversion. If ``True`` (default), the atomic masses
        will be returned in atomic units. If ``False``, the unit remains g/mol.
    device : :class:`torch.device` | None, optional
        Device to store the tensor. If ``None`` (default), the default device
        is used.
    dtype : :class:`torch.dtype`, optional
        Data type of the tensor. If ``None`` (default), the default dtype
        is used.

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
    Get charge of valence shell for all ``numbers``.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers for all atoms in the system of shape ``(..., nat)``.
    device : :class:`torch.device` | None, optional
        Device to store the tensor. If ``None`` (default), the default device
        is used.
    dtype : :class:`torch.dtype`, optional
        Data type of the tensor. If ``None`` (default), the default dtype
        is used.

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
    Get number of core electrons for all ``numbers``.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers for all atoms in the system of shape ``(..., nat)``.
    device : :class:`torch.device` | None, optional
        Device to store the tensor. If ``None`` (default), the default device
        is used.
    dtype : :class:`torch.dtype`, optional
        Data type of the tensor. If ``None`` (default), the default dtype
        is used.

    Returns
    -------
    Tensor
        Number of core electrons.
    """
    return zeff.ECORE.to(device=device, dtype=dtype)[numbers]
