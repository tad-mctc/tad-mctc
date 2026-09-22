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
The one table naming every coordination-number variant this package ships,
each against its own :class:`~tad_mctc.typing.CNFunc`, cutoff, and the two
Fortran reference keys :mod:`samples`' ``Refs`` stores for it (the value
``cn_<variant>`` and the derivative ``dcn_<variant>dr``). ``test_reference.py``
(value correctness) and every file in ``test_grad/`` (gradient correctness)
share this same table rather than each listing the seven variants for
themselves -- two independently-typed variant lists is exactly the kind of
drift an architecture review flagged: `test_grad/`'s once listed four of
the seven and silently missed the other three.
"""

from __future__ import annotations

from typing import NamedTuple

from tad_mctc.ncoord import (
    cn_d3,
    cn_d4,
    cn_eeq,
    cn_eeq_en,
    cn_eeqbc,
    cn_eeqbc_en,
    cn_gfn2,
    defaults,
)
from tad_mctc.ncoord.common import CNModel


class Variant(NamedTuple):
    # `CNModel`, not the narrower `CNFunc` Protocol: `test_precomputed_shifts.py`
    # calls `.with_precomputed_shifts` on this, which `CNFunc` (a plain
    # `__call__`) does not declare, and every preset in `VARIANTS` below
    # is a `CNModel` instance regardless.
    call: CNModel
    ref_key: str
    dref_key: str
    cutoff: float
    abs_tol: float | None = None


# `cn_eeq`'s public preset caps at `cn_max=8`, but the stored Fortran
# reference is the uncapped construction (matching mctc-lib's own uncapped
# CN), so this uncapped model is what `VARIANTS["cn_eeq"]` calls; the cap
# itself is exercised separately (`test_model.py`, `test_reference.py::
# test_single_cnmax`).
_cn_eeq_uncapped = cn_eeq.replace(cn_max=None)

# Every variant is exercised against the full shared sample list(s): every
# stored sample carries a `cn_<variant>`/`dcn_<variant>dr` pair for every
# variant (see `samples.py`), so there is no reason for a variant to see
# only a subset of either.
VARIANTS: dict[str, Variant] = {
    "cn_d3": Variant(
        cn_d3,
        "cn_d3",
        "dcn_d3dr",
        defaults.CUTOFF_D3,
    ),
    "cn_d4": Variant(
        cn_d4,
        "cn_d4",
        "dcn_d4dr",
        defaults.CUTOFF_D4,
    ),
    "cn_gfn2": Variant(
        cn_gfn2,
        "cn_gfn2",
        "dcn_gfn2dr",
        defaults.CUTOFF_GFN2,
        abs_tol=1e-5,
    ),
    "cn_eeq": Variant(
        _cn_eeq_uncapped,
        "cn_eeq",
        "dcn_eeqdr",
        _cn_eeq_uncapped.cutoff,
    ),
    "cn_eeqbc": Variant(
        cn_eeqbc,
        "cn_eeqbc",
        "dcn_eeqbcdr",
        defaults.CUTOFF_EEQBC,
        abs_tol=1e-5,
    ),
    "cn_eeq_en": Variant(
        cn_eeq_en,
        "cn_eeq_en",
        "dcn_eeq_endr",
        _cn_eeq_uncapped.cutoff,
        abs_tol=1e-5,
    ),
    "cn_eeqbc_en": Variant(
        cn_eeqbc_en,
        "cn_eeqbc_en",
        "dcn_eeqbc_endr",
        defaults.CUTOFF_EEQBC,
        abs_tol=1e-5,
    ),
}
