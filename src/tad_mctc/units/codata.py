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

CODATA values for various physical constants using qcelemental.
"""

from __future__ import annotations

import qcelemental as qcel

__all__ = ["CODATA", "get_constant"]


CODATA = qcel.PhysicalConstantsContext("CODATA2018")
"""CODATA values for various physical constants."""


def get_constant(name: str) -> float:
    """
    Get a constant from ``qcelemental.constants`` (CODATA 2018).

    Parameters
    ----------
    name : str
        Name of the constant.

    Returns
    -------
    float
        Value of the constant.

    Raises
    ------
    KeyError
        If the constant is not found.
    """
    if name not in CODATA.pc.keys():
        raise KeyError(f"Constant '{name}' not found.")
    return float(CODATA.pc[name].data)
