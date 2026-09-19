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
I/O: Structure
==============

`Structure` is the one representation of an atomic structure this library
passes around -- species, positions, and optionally charge, unpaired
electron count, a periodic unit cell, and bond connectivity. It lives
here, under `io`,
mirroring mctc-lib's own layout: `structure_type` is defined in
`mctc-lib/src/mctc/io/structure.f90`, a plain data holder with fields
only, while behaviour (distances, coordination number, ...) lives as free
procedures in sibling modules outside `io` (`properties.general`,
`ncoord.common`) that take a `Structure`'s tensors as arguments rather
than as bound methods.

`Structure` is a validating, immutable container, matching s-dftd3's own
Python binding (`dftd3.interface.Structure`): construction runs
`io.checks.structure.structure_check` and raises on a malformed field, and
the instance cannot be mutated afterwards -- `.to()`/`.type()` return a
new, re-validated instance instead.

Pytree registration and the vmap constraint it exists for
-----------------------------------------------------------
`Structure` is registered with `torch.utils._pytree` so it can be passed
directly through `torch.func.vmap`, `torch.func.jacrev` and
`torch.compile`. The flatten function omits every unset optional field
from the emitted leaves entirely, putting the *set* of present field names
into the pytree's treespec context instead -- exactly mirroring how a
plain dict's key set, not a `None` value, is what makes a key "absent".

This matters because `None` is itself a pytree leaf: a naive
implementation that always emits six children (with `None` standing in
for an unset optional field) breaks `vmap(f, in_dims=0)` the moment any
optional field is unset, since vmap cannot assign an `in_dim` to a
non-Tensor leaf. Emitting only the present fields avoids that trap.

One consequence carries over from the dict analogy: every element of a
batch passed through `vmap` must agree on which optional fields are
present, the same way stacking a batch of dicts requires the same key set
in each one. Mixing, say, one structure with `lattice` and one without in
the same batched call is not supported.

Example
-------
>>> import torch
>>> from tad_mctc.io.structure import Structure
>>> numbers = torch.tensor([8, 1, 1])
>>> positions = torch.zeros((3, 3), dtype=torch.double)
>>> structure = Structure(numbers=numbers, positions=positions)
>>> structure.charge is None
True
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch
from torch import Tensor

try:
    from torch.utils._pytree import register_pytree_node
except ImportError:  # pragma: no cover
    # PyTorch < 2.1 only has the private, otherwise identical predecessor
    # this was later renamed from.
    from torch.utils._pytree import (
        _register_pytree_node as register_pytree_node,
    )

from .checks.structure import structure_check

__all__ = ["Structure"]


# The six fields that may be absent. Order here fixes the order in which
# a present optional field appears among the pytree's leaves.
_OPTIONAL_FIELDS = (
    "charge",
    "uhf",
    "lattice",
    "periodic",
    "bonds",
    "bond_orders",
)

# `numbers`, `uhf`, `periodic` and `bonds` are integer/boolean data (atomic
# numbers, an electron count, a boundary-condition mask, atom-index pairs);
# a floating `.type()` call must never touch them, mirroring
# `NeighborList.to()`'s int/bool-vs-float distinction in
# `src/tad_mctc/neighbor/list.py`. Enforced in `.to()` by passing
# `follow_dtype=False` for exactly these four fields.


@dataclass(frozen=True, eq=False)
class Structure:
    """
    One atomic structure: species, positions, and the optional fields that
    extend them (total charge, unpaired electron count, periodic unit
    cell, bond connectivity).

    Frozen because a structure is a value, not a place to mutate in place.
    `eq=False` because the generated `__eq__` a dataclass would otherwise
    get compares the tensor fields directly, which raises (a tensor's
    `==` returns another tensor, not a `bool`) -- the same reason
    `ncoord.common.CNModel` also opts out of it.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers for all atoms in the system, shape ``(..., nat)``.
    positions : Tensor
        Cartesian coordinates of all atoms, shape ``(..., nat, 3)``.
    charge : Tensor | None, optional
        Total charge. Absent (``None``, the default) means neutral,
        mctc-lib's own default.
    uhf : Tensor | None, optional
        Number of unpaired electrons. Absent means closed-shell.
    lattice : Tensor | None, optional
        Lattice vectors as rows, shape ``(..., 3, 3)``, in Bohr. Absent
        means a non-periodic structure.
    periodic : Tensor | None, optional
        Boolean mask, shape ``(..., 3)``, marking which lattice axes are
        periodic. Only meaningful alongside ``lattice``.
    bonds : Tensor | None, optional
        Atom-index pairs describing bond connectivity, shape
        ``(..., nbond, 2)``. Absent means no connectivity information.
    bond_orders : Tensor | None, optional
        Bond order per entry in ``bonds``, shape ``(..., nbond)``. Only
        meaningful alongside ``bonds``.

    Raises
    ------
    RuntimeError
        A tensor has the wrong number of dimensions or an inconsistent
        shape (raised by `structure_check`).
    DtypeError
        ``numbers``, ``periodic`` or ``bonds`` has the wrong dtype.
    DeviceError
        The given tensors do not all live on the same device.
    """

    numbers: Tensor
    positions: Tensor
    charge: Tensor | None = None
    uhf: Tensor | None = None
    lattice: Tensor | None = None
    periodic: Tensor | None = None
    bonds: Tensor | None = None
    bond_orders: Tensor | None = None

    def __post_init__(self) -> None:
        structure_check(
            self.numbers,
            self.positions,
            charge=self.charge,
            uhf=self.uhf,
            lattice=self.lattice,
            periodic=self.periodic,
            bonds=self.bonds,
            bond_orders=self.bond_orders,
        )

    def to(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Structure:
        """
        Copy this structure to a new device and/or floating dtype.

        Every *set* field moves to ``device``, required and optional
        alike -- a partial move would leave `structure_check`'s
        device-consistency check failing the next time this structure (or
        a copy of it) is validated. Only the floating-point fields
        (``positions``, ``charge``, ``lattice``, ``bond_orders``) follow
        ``dtype``; ``numbers``, ``uhf``, ``periodic`` and ``bonds`` are
        integer/boolean and are never cast to a floating dtype.

        Parameters
        ----------
        device : torch.device | None, optional
            Device to move every set field to. ``None`` keeps the current
            device.
        dtype : torch.dtype | None, optional
            Floating dtype for the floating-point fields. ``None`` keeps
            the current dtype.

        Returns
        -------
        Structure
            A new, re-validated instance on the requested device/dtype.
        """
        if device is None and dtype is None:
            return self

        target_device = device if device is not None else self.numbers.device
        target_dtype = dtype if dtype is not None else self.positions.dtype

        def convert_optional(
            value: Tensor | None, *, follow_dtype: bool
        ) -> Tensor | None:
            """Move one optional field, skipping the ones left unset.
            `follow_dtype=False` for the integer/boolean fields (`uhf`,
            `periodic`), which must never be cast to a floating dtype."""
            if value is None:
                return None
            if follow_dtype:
                return value.to(device=target_device, dtype=target_dtype)
            return value.to(device=target_device)

        # `numbers`/`positions` are required, so they are converted
        # directly rather than through `convert_optional` -- keeping
        # their type as plain `Tensor`, not `Tensor | None`, for the
        # constructor call below.
        return Structure(
            numbers=self.numbers.to(device=target_device),
            positions=self.positions.to(
                device=target_device, dtype=target_dtype
            ),
            charge=convert_optional(self.charge, follow_dtype=True),
            uhf=convert_optional(self.uhf, follow_dtype=False),
            lattice=convert_optional(self.lattice, follow_dtype=True),
            periodic=convert_optional(self.periodic, follow_dtype=False),
            bonds=convert_optional(self.bonds, follow_dtype=False),
            bond_orders=convert_optional(self.bond_orders, follow_dtype=True),
        )

    def type(self, dtype: torch.dtype) -> Structure:
        """
        Copy this structure to a new floating dtype, keeping its device.

        See :meth:`to` for which fields ``dtype`` actually applies to.

        Parameters
        ----------
        dtype : torch.dtype
            Floating dtype for the floating-point fields.

        Returns
        -------
        Structure
            A new, re-validated instance with the requested dtype.
        """
        return self.to(dtype=dtype)


def _flatten(structure: Structure) -> tuple[list[Tensor], tuple[str, ...]]:
    """Pytree flatten function: emit only the *set* fields as leaves, in a
    fixed order, and record which fields those were as the treespec
    context -- see the module docstring for why an absent field must not
    become a `None` leaf."""
    fields: dict[str, Tensor] = {
        "numbers": structure.numbers,
        "positions": structure.positions,
    }
    for name in _OPTIONAL_FIELDS:
        value = getattr(structure, name)
        if value is not None:
            fields[name] = value

    return list(fields.values()), tuple(fields.keys())


def _unflatten(children: Iterable[Any], context: tuple[str, ...]) -> Structure:
    """Pytree unflatten function: pair the recorded field names back up
    with their (possibly transformed) leaves.

    `children` is typed `Iterable[Any]`, not `Iterable[Tensor]`, because it
    is not always one: `vmap` and friends probe a treespec by unflattening
    placeholder, non-Tensor leaves (e.g. plain ints) purely to inspect its
    shape.

    Deliberately bypasses `__init__`/`__post_init__` (via `object.__new__`
    plus direct attribute assignment) rather than calling
    `Structure(**dict(zip(context, children)))` directly: running
    `structure_check` against the placeholder leaves described above would
    raise. The leaves reaching this function during real use were already
    valid at flatten time, and the transforms this container supports
    (`vmap` slicing a batch dimension, `jacrev`/`torch.compile` wrapping a
    leaf) cannot turn a valid leaf into one `structure_check` would reject,
    so skipping re-validation here loses no safety.
    """
    structure = object.__new__(Structure)
    present = dict(zip(context, children))
    for name in ("numbers", "positions") + _OPTIONAL_FIELDS:
        object.__setattr__(structure, name, present.get(name))
    return structure


register_pytree_node(Structure, _flatten, _unflatten)
