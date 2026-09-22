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
Data: Structures - Other
=========================

Bespoke test structures with no verified upstream origin, split by content
shape across three sibling modules:

- :mod:`tad_mctc.data.structures.other.atoms` -- single-atom systems.
- :mod:`tad_mctc.data.structures.other.molecules` -- diatomics through the
  largest bespoke fixtures.
- :mod:`tad_mctc.data.structures.other.periodic` -- every periodic
  (lattice-bearing) record, real crystals and synthetic PBC test cells
  alike -- see that module's own docstring for why the two share a file.

Unlike :mod:`tad_mctc.data.structures.mstore` (one file per verified
upstream source, matching mstore's own Fortran file boundaries), `other`
has no per-record sources to split along -- this grouping is by shape, not
provenance, purely for file-level navigability while editing. `other`
merges all three into one flat dict, keyed by record name; callers reach
a record the same way regardless of which sibling module backs its name,
and nothing outside this package should import `atoms`, `molecules` or
`periodic` directly.

Molecules that mstore also provides (e.g. `mb16_43/CH4`, `heavy28/h2o`,
`heavy28/nh3`) are not duplicated here; get them through
:func:`tad_mctc.data.structures.get_structure`.

Structures with a verified upstream origin in mstore
(https://github.com/grimme-lab/mstore) live in
:mod:`tad_mctc.data.structures.mstore` instead, organized per dataset.
"""

from __future__ import annotations

from torch import Tensor

from .atoms import atoms
from .molecules import molecules
from .periodic import periodic

__all__ = ["other"]


other: dict[str, dict[str, Tensor]] = {**atoms, **molecules, **periodic}
