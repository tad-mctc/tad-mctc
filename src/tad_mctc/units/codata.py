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
Units: CODATA
=============

CODATA values for various physical constants.
"""
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
