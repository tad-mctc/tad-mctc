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

Named test structures, reached through one call::

    get_structure("x23", "acetic")      # an mstore record
    get_structure("other", "diamond")   # a bespoke tad-mctc record
    get_structure("glu_ala", "0064")    # one step of the glu_ala size ladder

A structure is always named by ``(collection, record)``, mirroring mstore's
``get_structure(mol, collection, record)``. The collections come from three
data sources, which only supply records and have no lookup logic of their
own:

- :mod:`tad_mctc.data.structures.mstore` -- one collection per mstore
  dataset (https://github.com/grimme-lab/mstore), with mstore's own record
  ids.
- :mod:`tad_mctc.data.structures.other` -- the ``"other"`` collection:
  bespoke structures with no mstore counterpart, including periodic cells.
- :mod:`tad_mctc.data.structures.glu_ala` -- the ``"glu_ala"`` collection:
  a peptide size ladder for scaling benchmarks, loaded lazily.
"""

from __future__ import annotations

from typing import Mapping, TypeVar

import torch
from torch import Tensor

from ...io.structure import Structure
from .glu_ala import glu_ala
from .mstore import datasets
from .other import other

__all__ = [
    "collections",
    "get_structure",
    "list_collections",
    "list_records",
]


collections: dict[str, Mapping[str, dict[str, Tensor]]] = {
    **datasets,
    "other": other,
    "glu_ala": glu_ala,
}
"""Every collection, keyed by collection name, each mapping record ids to
the fields of one :class:`~tad_mctc.io.structure.Structure`."""


V = TypeVar("V")


def _lookup_or_raise(
    mapping: Mapping[str, V], key: str, what: str, context: str = ""
) -> V:
    """
    Return ``mapping[key]``, or raise a ``KeyError`` that lists every valid
    key, so a typo in a collection or record name is easy to fix.
    """
    try:
        return mapping[key]
    except KeyError:
        where = f" in {context}" if context else ""
        raise KeyError(
            f"Unknown {what} '{key}'{where}. Available: {sorted(mapping)}"
        ) from None


def get_structure(
    collection: str,
    record: str,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Structure:
    """
    Look up one structure by collection and record id.

    Parameters
    ----------
    collection : str
        Collection name, one of :func:`list_collections` (e.g. ``"x23"``,
        ``"other"``).
    record : str
        Record id within that collection (e.g. ``"acetic"``).
    device : torch.device | None, optional
        Device to move the structure to. ``None`` keeps the default device.
        Passing a ``DD`` via ``**dd`` works, since its keys match these
        parameter names.
    dtype : torch.dtype | None, optional
        Floating dtype for the structure's floating-point fields. ``None``
        keeps the default dtype.

    Returns
    -------
    Structure
        The requested structure.

    Raises
    ------
    KeyError
        If ``collection`` or ``record`` is not found, listing the valid
        options for whichever lookup failed.
    """
    records = _lookup_or_raise(collections, collection, "collection")
    fields = _lookup_or_raise(
        records, record, "record", context=f"collection '{collection}'"
    )
    return Structure(**fields).to(device=device, dtype=dtype)


def list_collections() -> list[str]:
    """
    List every collection name, mirroring mstore's ``list_collections``.

    Returns
    -------
    list[str]
        The keys of :data:`collections`.
    """
    return list(collections)


def list_records(collection: str) -> list[str]:
    """
    List every record id in one collection, mirroring mstore's
    ``list_records``.

    Parameters
    ----------
    collection : str
        Collection name, one of :func:`list_collections`.

    Returns
    -------
    list[str]
        The record ids in that collection.

    Raises
    ------
    KeyError
        If ``collection`` is not found, listing the valid collections.
    """
    return list(_lookup_or_raise(collections, collection, "collection"))
