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
Data: Radii
===========

Covalent radii.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch

from .._version import __tversion__
from ..typing import Any, Tensor
from ..units import length

__all__ = ["ATOMIC", "COV_D3", "VDW_D3", "VDW_PAIRWISE"]

# fmt: off
_ATOMIC = [
    0.00,  # dummy
    0.32,0.37,  # H,He
    1.30,0.99,0.84,0.75,0.71,0.64,0.60,0.62,  # Li-Ne
    1.60,1.40,1.24,1.14,1.09,1.04,1.00,1.01,  # Na-Ar
    2.00,1.74,  # K,Ca
    1.59,1.48,1.44,1.30,1.29,  # Sc-
    1.24,1.18,1.17,1.22,1.20,  # -Zn
    1.23,1.20,1.20,1.18,1.17,1.16,  # Ga-Kr
    2.15,1.90,  # Rb,Sr
    1.76,1.64,1.56,1.46,1.38,  # Y-
    1.36,1.34,1.30,1.36,1.40,  # -Cd
    1.42,1.40,1.40,1.37,1.36,1.36,  # In-Xe
    2.38,2.06,  # Cs,Ba
    1.94,1.84,1.90,1.88,1.86,1.85,1.83,  # La-Eu
    1.82,1.81,1.80,1.79,1.77,1.77,1.78,  # Gd-Yb
    1.74,1.64,1.58,1.50,1.41,  # Lu-
    1.36,1.32,1.30,1.30,1.32,  # -Hg
    1.44,1.45,1.50,1.42,1.48,1.46,  # Tl-Rn
    2.42,2.11,  # Fr,Ra
    2.01,1.90,1.84,1.83,1.80,1.80,1.73,  # Ac-Am
    1.68,1.68,1.68,1.65,1.67,1.73,1.76,  # Cm-No
    1.61,1.57,1.49,1.43,1.41,  # Lr-
    1.34,1.29,1.28,1.21,1.22,   # -Cn
    1.36,1.43,1.62,1.75,1.65,1.57,  # Nh-Og
]
"""Atomic radii."""
# fmt: on


@lru_cache(maxsize=None)
def ATOMIC(
    device: torch.device | None = None, dtype: torch.dtype = torch.double
) -> Tensor:
    """
    Atomic radii.

    Parameters
    ----------
    dtype : torch.dtype, optional
        Floating point precision for tensor. Defaults to `torch.double`.
    device : Optional[torch.device], optional
        Device of tensor. Defaults to None.

    Returns
    -------
    Tensor
        Atomic radii.
    """
    return length.AA2AU * torch.tensor(
        _ATOMIC, dtype=dtype, device=device, requires_grad=False
    )


##############################################################################


# fmt: off
_COV_2009 = [
    0.00,  # None
    0.32,0.46,  # H,He
    1.20,0.94,0.77,0.75,0.71,0.63,0.64,0.67,  # Li-Ne
    1.40,1.25,1.13,1.04,1.10,1.02,0.99,0.96,  # Na-Ar
    1.76,1.54,  # K,Ca
    1.33,1.22,1.21,1.10,1.07,  # Sc-
    1.04,1.00,0.99,1.01,1.09,  # -Zn
    1.12,1.09,1.15,1.10,1.14,1.17,  # Ga-Kr
    1.89,1.67,  # Rb,Sr
    1.47,1.39,1.32,1.24,1.15,  # Y-
    1.13,1.13,1.08,1.15,1.23,  # -Cd
    1.28,1.26,1.26,1.23,1.32,1.31,  # In-Xe
    2.09,1.76,  # Cs,Ba
    1.62,1.47,1.58,1.57,1.56,1.55,1.51,  # La-Eu
    1.52,1.51,1.50,1.49,1.49,1.48,1.53,  # Gd-Yb
    1.46,1.37,1.31,1.23,1.18,  # Lu-
    1.16,1.11,1.12,1.13,1.32,  # -Hg
    1.30,1.30,1.36,1.31,1.38,1.42,  # Tl-Rn
    2.01,1.81,  # Fr,Ra
    1.67,1.58,1.52,1.53,1.54,1.55,1.49,  # Ac-Am
    1.49,1.51,1.51,1.48,1.50,1.56,1.58,  # Cm-No
    1.45,1.41,1.34,1.29,1.27,  # Lr-
    1.21,1.16,1.15,1.09,1.22,  # -Cn
    1.36,1.43,1.46,1.58,1.48,1.57  # Nh-Og
]
# fmt: on
"""
Covalent radii (taken from Pyykko and Atsumi, Chem. Eur. J. 15, 2009, 188-197).
Values for metals decreased by 10 %.
"""


@lru_cache(maxsize=None)
def COV_D3(
    device: torch.device | None = None, dtype: torch.dtype = torch.double
) -> Tensor:
    """
    Covalent radii (taken from Pyykko and Atsumi, Chem. Eur. J. 15, 2009,
    188-197). Values for metals decreased by 10 %.
    """

    t = torch.tensor(_COV_2009, dtype=dtype, device=device, requires_grad=False)
    return length.AA2AU * 4.0 / 3.0 * t


##############################################################################


# fmt: off
_VDW_D3 = [
    0.00000,                            # dummy value
    1.09155, 0.86735, 1.74780, 1.54910, # H-Be
    1.60800, 1.45515, 1.31125, 1.24085, # B-O
    1.14980, 1.06870, 1.85410, 1.74195, # F-Mg
    2.00530, 1.89585, 1.75085, 1.65535, # Al-S
    1.55230, 1.45740, 2.12055, 2.05175, # Cl-Ca
    1.94515, 1.88210, 1.86055, 1.72070, # Sc-Cr
    1.77310, 1.72105, 1.71635, 1.67310, # Mn-Ni
    1.65040, 1.61545, 1.97895, 1.93095, # Cu-Ge
    1.83125, 1.76340, 1.68310, 1.60480, # As-Kr
    2.30880, 2.23820, 2.10980, 2.02985, # Rb-Zr
    1.92980, 1.87715, 1.78450, 1.73115, # Nb-Ru
    1.69875, 1.67625, 1.66540, 1.73100, # Rh-Cd
    2.13115, 2.09370, 2.00750, 1.94505, # In-Te
    1.86900, 1.79445, 2.52835, 2.59070, # I-Ba
    2.31305, 2.31005, 2.28510, 2.26355, # La-Nd
    2.24480, 2.22575, 2.21170, 2.06215, # Pm-Gd
    2.12135, 2.07705, 2.13970, 2.12250, # Tb-Er
    2.11040, 2.09930, 2.00650, 2.12250, # Tm-Hf
    2.04900, 1.99275, 1.94775, 1.87450, # Ta-Os
    1.72280, 1.67625, 1.62820, 1.67995, # Ir-Hg
    2.15635, 2.13820, 2.05875, 2.00270, # Tl-Po
    1.93220, 1.86080, 2.53980, 2.46470, # At-Ra
    2.35215, 2.21260, 2.22970, 2.19785, # Ac-U
    2.17695, 2.21705                    # Np-Pu
]
"""D3 pairwise van-der-Waals radii (only homoatomic pairs present here)"""
# fmt: on


@lru_cache(maxsize=None)
def VDW_D3(
    device: torch.device | None = None, dtype: torch.dtype = torch.double
) -> Tensor:
    """D3 pairwise van-der-Waals radii (only homoatomic pairs present here)"""
    return length.AA2AU * torch.tensor(
        _VDW_D3, dtype=dtype, device=device, requires_grad=False
    )


##############################################################################


def _load_vdw_rad_pairwise(
    device: torch.device | None = None, dtype: torch.dtype = torch.double
) -> Tensor:
    """
    Load reference VDW radii from file.

    Parameters
    ----------
    dtype : torch.dtype, optional
        Floating point precision for tensor. Defaults to `torch.double`.
    device : Optional[torch.device], optional
        Device of tensor. Defaults to None.

    Returns
    -------
    Tensor
        VDW radii.
    """
    kwargs: dict[str, Any] = {"map_location": device}
    if __tversion__ > (1, 12, 1):  # pragma: no cover
        kwargs["weights_only"] = True

    path = Path(__file__).parent / "vdw-pairwise.pt"

    tensor = torch.load(path, **kwargs)
    return tensor.to(dtype) if tensor.dtype is not dtype else tensor


@lru_cache(maxsize=None)
def VDW_PAIRWISE(
    device: torch.device | None = None, dtype: torch.dtype = torch.double
) -> Tensor:
    """Pair-wise Van-der-Waals radii."""
    return _load_vdw_rad_pairwise(dtype=dtype, device=device)
