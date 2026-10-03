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
new instance instead (conversion runs no checks).

Pytree behaviour and the vmap constraint
----------------------------------------
`Structure` is a :class:`~tad_mctc.tree.Node`, so it is registered as a
pytree and can be passed directly through `torch.func.vmap`,
`torch.func.jacrev` and `torch.compile`. The fields holding a tensor are
the leaves; an absent optional field (``None``) is part of the tree
structure, not a leaf -- exactly mirroring how a plain dict's key set, not
a `None` value, is what makes a key "absent".

One consequence carries over from the dict analogy: every element of a
batch passed through `vmap` must agree on which optional fields are
set, the same way stacking a batch of dicts requires the same key set
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

from collections.abc import Sequence

import torch
from torch import Tensor

from ..batch import pack
from ..tree import Node, child
from .checks.structure import structure_check

__all__ = ["Structure", "pack_structures"]


class Structure(Node):
    """
    One atomic structure: species, positions, and the optional fields that
    extend them (total charge, unpaired electron count, periodic unit
    cell, bond connectivity).

    Frozen because a structure is a value, not a place to mutate in place.
    Equality and hashing are by identity (see :class:`~tad_mctc.tree.Node`),
    because comparing the tensor fields directly raises (a tensor's `==`
    returns another tensor, not a `bool`).

    On construction, floating-point ``charge``, ``lattice`` and
    ``bond_orders`` tensors with a dtype different from ``positions`` are
    cast to the dtype of ``positions``, and a missing ``periodic`` mask is
    filled in for a periodic structure (see below). ``numbers``, ``uhf``,
    ``periodic`` and ``bonds`` are integer/boolean data and are never cast
    to a floating dtype by `.to()`/`.type()`. Every *set* field moves with
    ``device`` in `.to()`, so that the device-consistency check of
    `structure_check` keeps passing.

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
        Boolean mask, shape ``(3,)`` or the lattice's ``(..., 3)``, marking
        which lattice axes are periodic. Requires ``lattice``. Absent
        alongside a lattice means periodic along every axis, and is filled
        in on construction, as in mctc-lib's ``new_structure``; so a
        structure with a lattice always has a mask.
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

    numbers: Tensor = child()
    positions: Tensor = child()
    charge: Tensor | None = child(default=None)
    uhf: Tensor | None = child(default=None, keep_dtype=True)
    lattice: Tensor | None = child(default=None)
    periodic: Tensor | None = child(default=None)
    bonds: Tensor | None = child(default=None)
    bond_orders: Tensor | None = child(default=None)

    def _normalize(self) -> dict[str, Tensor]:
        updates: dict[str, Tensor] = {}

        if self.lattice is not None and self.periodic is None:
            updates["periodic"] = torch.ones(
                self.lattice.shape[:-1],
                dtype=torch.bool,
                device=self.lattice.device,
            )

        dtype = self.positions.dtype
        for name in ("charge", "lattice", "bond_orders"):
            value = getattr(self, name)
            if (
                isinstance(value, Tensor)
                and value.is_floating_point()
                and value.dtype != dtype
            ):
                updates[name] = value.to(dtype=dtype)

        return updates

    def _validate(self) -> None:
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


# Unset means neutral / closed-shell, so a structure that leaves one of
# these unset packs as zero next to one that sets it.
_ZERO_DEFAULT_FIELDS = ("charge", "uhf")

# Unset means "not periodic", which has no stackable stand-in value.
_CELL_FIELDS = ("lattice", "periodic")


def _stack_with_zero_default(values: list[Tensor | None]) -> Tensor:
    """Stack `values`, replacing each unset entry with a zero shaped like
    the set ones. At least one entry must be set."""
    template = next(value for value in values if value is not None)
    zero = torch.zeros_like(template)
    return torch.stack([zero if value is None else value for value in values])


def pack_structures(structures: Sequence[Structure]) -> Structure:
    """
    Pack several unbatched structures into one batched `Structure`.

    `numbers` and `positions` are zero-padded to the largest structure
    (see :func:`tad_mctc.batch.pack`). The per-structure fields are
    stacked: ``lattice`` and ``periodic`` as they are, ``charge`` and
    ``uhf`` with an unset value read as zero (neutral, closed-shell).
    Several molecules or several periodic cells batch fine; a molecule
    next to a periodic cell does not (see Raises).

    mctc-lib has no batched structure type, so this is a PyTorch-side
    addition. Padding is data-dependent: call this while building the
    input, outside any `vmap`/`jacrev`/`torch.compile` region.

    Parameters
    ----------
    structures : Sequence[Structure]
        Unbatched structures, all on the same device and dtype.

    Returns
    -------
    Structure
        One structure with a leading batch dimension.

    Raises
    ------
    ValueError
        `structures` is empty, one of them is already batched, or some
        set ``lattice``/``periodic`` and others do not -- a molecule next
        to a periodic cell. To batch a molecule with periodic cells, give
        it any non-singular ``lattice`` (e.g. the identity) together with
        ``periodic=[False, False, False]``.
    NotImplementedError
        A structure sets ``bonds``, for which there is no padding
        convention yet (zero-padding would invent bonds to atom 0).
    """
    if len(structures) == 0:
        raise ValueError("Cannot pack an empty sequence of structures.")
    if any(structure.numbers.ndim != 1 for structure in structures):
        raise ValueError("Only unbatched structures can be packed.")
    if any(structure.bonds is not None for structure in structures):
        raise NotImplementedError("Packing structures with bonds.")

    optional: dict[str, Tensor] = {}

    # A structure with a lattice always has a mask (see `Structure`), so
    # checking the lattice covers both cell fields.
    has_lattice = [structure.lattice is not None for structure in structures]
    if any(has_lattice) and not all(has_lattice):
        raise ValueError(
            "Cannot pack molecules with periodic structures: `lattice` is "
            "set on some structures but not on others. To batch a molecule "
            "with periodic cells, give it a non-singular lattice (e.g. the "
            "identity) and periodic=[False, False, False]."
        )
    if all(has_lattice):
        for name in _CELL_FIELDS:
            values = [getattr(structure, name) for structure in structures]
            optional[name] = torch.stack(values)

    for name in _ZERO_DEFAULT_FIELDS:
        values = [getattr(structure, name) for structure in structures]
        if any(value is not None for value in values):
            optional[name] = _stack_with_zero_default(values)

    return Structure(
        numbers=pack([structure.numbers for structure in structures]),
        positions=pack([structure.positions for structure in structures]),
        **optional,
    )
