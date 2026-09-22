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
Molecules for testing bond orders.
"""

from __future__ import annotations

from dataclasses import fields
from typing import TypedDict

import torch

from tad_mctc.data.structures import get_structure
from tad_mctc.io.structure import Structure
from tad_mctc.typing import Tensor

# Historical name -> (mstore collection, record), or None for a bespoke
# `tad_mctc.data.structures` entry -- see `test_ncoord/samples.py` for the
# same pattern and its rationale.
_SAMPLE_SOURCES: dict[str, tuple[str, str] | None] = {
    "PbH4-BiH3": ("heavy28", "pbh4_bih3"),
    "C6H5I-CH3SH": None,
}


def _structure(name: str, source: tuple[str, str] | None) -> Structure:
    if source is None:
        return get_structure("other", name)
    collection, record = source
    return get_structure(collection, record)


# `Structure` is a validating dataclass, not a mapping, so it cannot be
# `**`-unpacked into the merged `Record` dict below the way the old
# `Structure` `TypedDict` could. This pulls out only the *set* fields,
# mirroring `io.structure._flatten`'s own "absent optional field is
# omitted, not `None`" rule, so a closed-shell sample does not grow a
# spurious `uhf: None` entry either.
def _structure_dict(structure: Structure) -> dict[str, Tensor]:
    present = {}
    for field in fields(structure):
        value = getattr(structure, field.name)
        if value is not None:
            present[field.name] = value
    return present


class Refs(TypedDict):
    """Format for reference values."""

    bo: Tensor
    """Reference bond orders."""

    cn: Tensor
    """DFT-D3 coordination number."""


class Record(TypedDict):
    """Store for molecular information and reference values. Declares the
    `Structure` fields these samples actually carry (`numbers`/`positions`
    only -- none of them are periodic or open-shell) plus `Refs`, rather
    than inheriting from `Structure` itself, which is a dataclass and
    cannot be a `TypedDict` base."""

    numbers: Tensor
    positions: Tensor
    bo: Tensor
    cn: Tensor


refs: dict[str, Refs] = {
    "PbH4-BiH3": {
        "bo": torch.tensor(
            [
                0.6533,
                0.6533,
                0.6533,
                0.6499,
                0.6533,
                0.6533,
                0.6533,
                0.6499,
                0.5760,
                0.5760,
                0.5760,
                0.5760,
                0.5760,
                0.5760,
            ],
        ),
        "cn": torch.tensor(
            [
                3.9388208389,
                0.9832025766,
                0.9832026958,
                0.9832026958,
                0.9865897894,
                2.9714603424,
                0.9870455265,
                0.9870456457,
                0.9870455265,
            ],
        ),
    },
    "C6H5I-CH3SH": {
        "bo": torch.tensor(
            [
                0.4884,
                0.5180,
                0.4006,
                0.4884,
                0.4884,
                0.4012,
                0.4884,
                0.5181,
                0.4006,
                0.5181,
                0.5144,
                0.4453,
                0.5144,
                0.5145,
                0.4531,
                0.5180,
                0.5145,
                0.4453,
                0.4531,
                0.4012,
                0.4453,
                0.4006,
                0.4453,
                0.4006,
                0.6041,
                0.3355,
                0.6041,
                0.3355,
                0.5645,
                0.5673,
                0.5670,
                0.5645,
                0.5673,
                0.5670,
            ]
        ),
        "cn": torch.tensor(
            [
                3.1393690109,
                3.1313166618,
                3.1393768787,
                3.3153429031,
                3.1376547813,
                3.3148119450,
                1.5363609791,
                1.0035246611,
                1.0122337341,
                1.0036621094,
                1.0121959448,
                1.0036619902,
                2.1570565701,
                0.9981809855,
                3.9841127396,
                1.0146225691,
                1.0123561621,
                1.0085891485,
            ],
        ),
    },
}


samples: dict[str, Record] = {
    name: {**_structure_dict(_structure(name, source)), **refs[name]}  # type: ignore[typeddict-item]
    for name, source in _SAMPLE_SOURCES.items()
}
