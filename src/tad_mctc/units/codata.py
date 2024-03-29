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
Units: CODATA
=============

CODATA values for various physical constants.
"""
from __future__ import annotations

from scipy.constants import physical_constants

__all__ = ["CODATA"]


def get_constant(name: str) -> float:
    """
    Get a constant from `scipy.constants.physical_constants`.

    Parameters
    ----------
    constant_name : str
        Name of the constant.

    Returns
    -------
    float
        Value of the constant.

    Raises
    ------
    KeyError
        Name of constant not found.
    """
    if name not in physical_constants:
        raise KeyError(f"Constant '{name}' not found.")
    return physical_constants[name][0]


class CODATA:
    """
    CODATA values for various physical constants.
    """

    h = get_constant("Planck constant")
    """Planck's constant"""

    c = get_constant("speed of light in vacuum")
    """Speed of light in vacuum (m/s)"""

    kb = get_constant("Boltzmann constant")
    """Boltzmann's constant"""

    na = get_constant("Avogadro constant")
    """Avogadro's number (mol^-1)"""

    e = get_constant("elementary charge")
    """Elementary charge"""

    alpha = get_constant("fine-structure constant")
    """Fine structure constant (CODATA2018)"""

    me = get_constant("electron mass")
    """Rest mass of the electron (kg)"""

    bohr = get_constant("Bohr radius")
    """Bohr radius (m)"""
