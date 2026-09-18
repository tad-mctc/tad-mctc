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
Data: Structures
================

Test-fixture structures for tad-mctc, merging two sources into one public
`structures` dict:

- :mod:`tad_mctc.data.structures.other` -- bespoke structures with no
  verified upstream origin.
- :mod:`tad_mctc.data.structures.mstore` -- structures with a verified,
  coordinate-confirmed origin in https://github.com/grimme-lab/mstore,
  organized per dataset there.

`structures` holds only `other`'s bespoke entries. An mstore record is
reached exclusively through
:func:`tad_mctc.data.structures.mstore.get_structure` (e.g.
``get_structure("mb16_43", "SiH4")``) -- `structures` used to re-key 11 of
them under historical compound names (`"MB16_43_01"`, `"SiH4"`, ...), but
those aliases were removed so that mstore's own records stay the one
canonical way to reach mstore data, with no second name pointing at the
same object.

A third, sibling source, :mod:`tad_mctc.data.structures.glu_ala`, is
likewise never merged into `structures`: a 26-structure, 28-to-53,250-atom
size ladder for scaling benchmarks (see `examples/scaling/glu_ala.py`),
reached through its own `get_structure`/`list_records`.
"""

from __future__ import annotations

from typing import Any

from torch import Tensor

from ...io.structure import Structure
from .other import other

__all__ = ["structures", "merge_nested_dicts"]


structures: dict[str, Structure] = {
    name: Structure(**record) for name, record in other.items()
}


def merge_nested_dicts(
    a: dict[str, dict[str, Tensor]], b: dict[str, Any]
) -> dict[str, Any]:
    """
    Merge nested dictionaries. dictionary `a` remains unaltered, while
    the corresponding keys of it are added to `b`.

    Parameters
    ----------
    a : dict
        First dictionary (not changed).
    b : dict
        Second dictionary (changed).

    Returns
    -------
    dict
        Merged dictionary `b`.
    """
    for key in b:
        if key in a:
            b[key].update(a[key])
    return b
