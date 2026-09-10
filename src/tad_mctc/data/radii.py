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

from pathlib import Path
from typing import Any

import torch

from .._version import __tversion__
from ..typing import Tensor
from ..units import length

__all__ = [
    "ATOMIC_RADII",
    "COV_D3",
    "EEQBC_COV_RADII",
    "VDW_D3",
    "VDW_PAIRWISE",
]


def ATOMIC_RADII(
    device: torch.device | None = None, dtype: torch.dtype | None = torch.double
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
    if dtype is None:
        dtype = torch.double

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
    # fmt: on

    return length.AA2AU * torch.tensor(
        _ATOMIC, dtype=dtype, device=device, requires_grad=False
    )


##############################################################################


def COV_D3(
    device: torch.device | None = None, dtype: torch.dtype = torch.double
) -> Tensor:
    """
    Covalent radii (taken from Pyykko and Atsumi, Chem. Eur. J. 15, 2009,
    188-197). Values for metals decreased by 10 %.
    """

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

    t = torch.tensor(_COV_2009, dtype=dtype, device=device, requires_grad=False)
    return length.AA2AU * 4.0 / 3.0 * t


##############################################################################


def EEQBC_COV_RADII(
    device: torch.device | None = None, dtype: torch.dtype = torch.double
) -> Tensor:
    """
    Covalent radii for the coordination number of the EEQBC charge model
    (Froitzheim, Müller, Hansen, Grimme, J. Chem. Phys. 2025, 162, 214109),
    taken from ``multicharge``'s ``eeqbc2025`` parametrization
    (``multicharge_param_eeqbc2025``'s ``eeqbc_cov_radii``). Already in
    atomic units (Bohr) in the source -- unlike :func:`COV_D3`, no
    Angstrom-to-Bohr conversion is applied here.
    """

    # fmt: off
    _EEQBC_COV_RADII = [
        0.0000000000,  # dummy
        1.0873678902,0.0045628280,2.8385414023,2.2369359793,  # H-Be
        2.2631432568,2.5556464299,2.6528219471,2.5471166478,  # B-O
        2.0970520036,1.1527679853,3.9222564151,3.6628112720,  # F-Mg
        3.1200757440,3.2311571633,3.3714412240,3.4966508157,  # Al-S
        3.1641167151,1.4177781099,4.2825156987,4.0979720404,  # Cl-Ca
        3.4027683492,3.1330930644,3.2083129359,3.1971240284,  # Sc-Cr
        3.0122827243,3.0215412255,2.9286697665,2.9318983659,  # Mn-Ni
        2.9333228012,3.4211414769,3.5543149265,3.3547895521,  # Cu-Ge
        3.8566746758,4.0522579752,3.7055903624,2.1533955559,  # As-Kr
        4.8750192244,4.2251415193,3.8193395754,3.7700784196,  # Rb-Zr
        3.8026660286,3.4791250751,3.3748738252,3.3400607232,  # Nb-Ru
        3.3194948126,3.5185381046,3.6974620558,4.2120386946,  # Rh-Cd
        4.2834967376,4.0408029917,4.1029792717,4.5056357496,  # In-Te
        4.1912939737,3.1889722321,5.3761906399,4.9848540155,  # I-Ba
        4.1643020686,4.2242687055,4.0906998457,4.0483164017,  # La-Nd
        4.0130748483,3.6618368303,3.8161213688,3.6044393411,  # Pm-Gd
        3.7159335631,3.8610077243,3.8543967507,3.7804332520,  # Tb-Er
        3.6171475823,3.6614908934,3.9127576452,3.7447075110,  # Tm-Hf
        3.7737132271,3.3371881773,3.3105897209,3.3868061092,  # Ta-Os
        3.4036207674,3.5310959808,3.5697281366,4.3942379403,  # Ir-Hg
        4.6313791191,4.3952892419,4.2961541488,4.6294486870,  # Tl-Po
        4.5547581691,3.6325616087,5.0182872162,4.4284455579,  # At-Ra
        3.7478960318,2.9868700652,3.6306659286,3.8514208797,  # Ac-U
        3.4838982283,3.5107992105,3.4255592145,3.5581888894,  # Np-Cf
        3.3079867688,3.4972376288,3.4590842861,3.2327976004,  # Es-Md
        3.4619632872,3.7360296053,3.5692246969,  # No-Lr
    ]
    # fmt: on

    t = torch.tensor(
        _EEQBC_COV_RADII, dtype=dtype, device=device, requires_grad=False
    )
    return 0.5 * t


##############################################################################


def VDW_D3(
    device: torch.device | None = None, dtype: torch.dtype | None = torch.double
) -> Tensor:
    """D3 pairwise van-der-Waals radii (only homoatomic pairs present here)"""
    if dtype is None:
        dtype = torch.double

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
    # fmt: on

    return length.AA2AU * torch.tensor(
        _VDW_D3, dtype=dtype, device=device, requires_grad=False
    )


##############################################################################


def _load_vdw_rad_pairwise(
    device: torch.device | None = None, dtype: torch.dtype | None = torch.double
) -> Tensor:
    """
    Load reference VDW radii from file.

    Regenerated with the following script whenever the Angstrom source or
    `length.AA2AU` changes:

    .. code-block:: python

        import re
        import torch
        from tad_mctc.units.length import AA2AU

        source = Path("s-dftd3/src/dftd3/data/vdwrad.f90").read_text()
        start = source.index("vdwrad(max_elem*(1+max_elem)/2)")
        body = source[source.index("[", start):source.index("]", start)]
        angstrom = [float(v) for v in re.findall(r"([0-9.]+)_wp", body)]

        max_elem = 103
        table = torch.zeros(max_elem + 1, max_elem + 1, dtype=torch.float64)
        for num1 in range(1, max_elem + 1):
            for num2 in range(1, max_elem + 1):
                hi, lo = max(num1, num2), min(num1, num2)
                index = lo + hi * (hi - 1) // 2 - 1
                table[num1, num2] = angstrom[index] * AA2AU
        torch.save(table, "vdw-pairwise.pt")

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
    if dtype is None:
        dtype = torch.double

    kwargs: dict[str, Any] = {"map_location": device}
    if __tversion__ > (1, 12, 1):  # pragma: no cover
        kwargs["weights_only"] = True

    path = Path(__file__).parent / "vdw-pairwise.pt"

    tensor = torch.load(path, **kwargs)
    return tensor.to(dtype) if tensor.dtype is not dtype else tensor


def VDW_PAIRWISE(
    device: torch.device | None = None, dtype: torch.dtype | None = torch.double
) -> Tensor:
    """
    Pair-wise Van-der-Waals radii.

    These radii were previously stored explicitly in one list and then
    reshaped to the required `(MAX_ELEMENT, MAX_ELEMENT)` tensor. For the
    old version, see older commits (e.g. https://github.com/dftd3/tad-dftd3/blob/ecc50f19adb8aa8baa38a188d04228c4f26975d6/src/tad_dftd3/data/radii.py)
    """
    return _load_vdw_rad_pairwise(dtype=dtype, device=device)
