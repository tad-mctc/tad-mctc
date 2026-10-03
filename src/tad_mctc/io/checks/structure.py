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
I/O Checks: Structure
=====================

This module contains various checkers for the structure that is read/written.

In particular,the following checks can be done:
- coldfusion_check (interatomic distances)
- content_checks (atomic numbers)
- deflatable_check (clash between padding and coordinates)

Note
----
The check `deflatable_check` attempts to catch cases, in which the padding
value (default: 0) is the same as a triple of atomic positions, which would
obscure the distinction between padding and actual atomic positions in batched
calculations. This primarily occurs for single atoms, which are usually placed
at the origin.
The behavior of this check is best controlled through keyword arguments of the
respective readers. The available keyword arguments are:
- padding_value (`float | int`, default: 0): Value for padding used in check
- raise_padding_exception (`bool`, default: False): Raise an exception (or just
a warning)
- raise_padding_warning (`bool`, default: True): Raise a warning
- shift_for_last (`bool`, default: False): Automatically shift all positions by
a constant if a clash is detected
- shift_value (`float | int`, default: 1.0): Constant for shift.

For more details and examples, check `test/test_io/test_deflatable.py`.

`coldfusion_check` takes a whole `Structure`, so that it can include
periodic images. `read_structure` runs it once on the structure it read when
called with `check_coldfusion=True`.
"""

from __future__ import annotations

from typing import IO, TYPE_CHECKING, Any, NoReturn

import torch

from ...autograd import is_functorch_tensor
from ...batch import deflate, real_pairs
from ...data import pse
from ...exceptions import (
    DeviceError,
    DtypeError,
    StructureError,
    StructureWarning,
)
from ...neighbor import (
    build_neighborlist,
    pair_distance_squared,
    split_lattice,
)
from ...typing import DD, Tensor

if TYPE_CHECKING:
    from ..structure import Structure

__all__ = [
    "coldfusion_check",
    "content_checks",
    "deflatable_check",
    "dimension_check",
    "structure_check",
]

# Below this interatomic distance (Bohr), atoms are considered fused. Real
# bonds never go this short (H-H, the shortest, is ~1.4 Bohr). It also has to
# clear the noise floor of the default (mm-based) `cdist` mode used below,
# which grows with distance from the origin -- in float32, that floor is
# still a few times below 0.5 for positions within ~300 Bohr of the origin,
# but keep this in mind for very large or far-shifted structures.
_COLDFUSION_THRESHOLD = 0.5

# Molecules are checked by comparing all pairs as long as the distance
# matrix, ``(..., nat, nat)`` over the whole batch, holds no more entries
# than that of one molecule of this many atoms: within 8 MiB (float64). A
# single molecule of up to this size, or a batch of small ones, qualifies;
# a long trajectory of them does not. The dense check takes a few
# milliseconds at most, while the neighbour list's first use on CPU loads
# (about 0.2 s) or even compiles (several seconds) the native extension,
# which a small molecule read with `check_coldfusion=True` should not pay.
# Beyond this size the neighbour list wins: 30x faster at 4000 atoms, and
# it needs memory only for the pairs within the cutoff.
_COLDFUSION_DENSE_MAX_ATOMS = 1024


def coldfusion_check(
    structure: Structure,
    *,
    threshold: Tensor | float | int | None = None,
    cutoff: float = 2.0,
) -> bool | NoReturn:
    """
    Check that no two atoms are closer than ``threshold`` (no fusion of
    atoms), periodic images included.

    A cell, a batch of cells, or a large molecule is checked through a
    neighbour list (:func:`.build_neighborlist`): only pairs within
    ``cutoff`` can be closer than ``threshold``, so this is O(nat) instead
    of O(nat^2), and a cell's list holds each atom's periodic images, so a
    contact across the cell boundary is caught too. The same holds for a
    molecule, or a batch of molecules, whose all-pairs distance matrix
    would exceed that of :data:`_COLDFUSION_DENSE_MAX_ATOMS` atoms. Anything
    smaller compares all pairs directly instead.

    Skipped inside ``torch.func.vmap``/``jacrev``, which do not allow
    data-dependent control flow.

    Parameters
    ----------
    structure : Structure
        The system(s) to check. ``numbers == 0`` marks padding, which is
        never compared.
    threshold : Tensor | float | int | None, optional
        Smallest acceptable interatomic distance (Bohr). Defaults to
        `None`, which resolves to `_COLDFUSION_THRESHOLD` (`0.5`).
    cutoff : float, optional
        Real-space cutoff (Bohr) of the neighbour search that narrows down
        the candidate pairs. Raised internally to ``threshold`` if
        smaller, so no violation is ever missed. Defaults to ``2.0``, well
        past the default ``threshold`` and below real bonding distances,
        keeping the search sparse. Unused where all pairs are compared
        directly.

    Returns
    -------
    bool
        ``True`` if no two atoms are too close.

    Raises
    ------
    StructureError
        Interatomic distances are too close.
    """
    numbers, positions = structure.numbers, structure.positions

    # vmap does not allow data-dependent control flow
    if is_functorch_tensor(numbers) or is_functorch_tensor(positions):
        return True

    dd: DD = {"device": positions.device, "dtype": positions.dtype}

    if threshold is None:
        threshold = torch.tensor(_COLDFUSION_THRESHOLD, **dd)
    elif not isinstance(threshold, Tensor):
        threshold = torch.tensor(threshold, **dd)

    if _coldfusion_uses_neighborlist(structure):
        _coldfusion_check_sparse(structure, threshold, cutoff)
        return True

    # Default (mm-based) `cdist` mode: faster, but loses precision on
    # nearby points whose coordinates are far from the origin (see the
    # comment above `_COLDFUSION_THRESHOLD` for why that is fine here).
    mask = real_pairs(numbers, mask_diagonal=True)
    pairwise = torch.cdist(positions, positions)
    distances = torch.where(mask, pairwise, torch.tensor(1e100, **dd))

    # Check if any distance below the threshold is found
    if torch.any((distances < threshold) & mask):
        raise StructureError("Too close interatomic distances found")

    return True


def _coldfusion_uses_neighborlist(structure: Structure) -> bool:
    """
    Whether :func:`coldfusion_check` searches ``structure`` with a
    neighbour list: a cell or batch of cells, or molecules whose all-pairs
    distance matrix has more than :data:`_COLDFUSION_DENSE_MAX_ATOMS` ``**
    2`` entries. Lets a caller (the command line) know whether the check
    will load the native extension.
    """
    if structure.lattice is not None:
        return True
    numbers = structure.numbers
    dense_entries = numbers.numel() * numbers.shape[-1]
    return dense_entries > _COLDFUSION_DENSE_MAX_ATOMS**2


def _coldfusion_check_sparse(
    structure: Structure, threshold: Tensor, cutoff: float
) -> None | NoReturn:
    """
    The neighbour-list form of :func:`coldfusion_check`, for a single
    structure or a batch, of cells or molecules.
    """
    effective_cutoff = max(cutoff, float(threshold))
    nbl = build_neighborlist(structure, effective_cutoff)

    idx_i, idx_j, shift = nbl.real_entries()
    shared_lattice, system_lattices = split_lattice(structure.lattice)
    distance_squared = pair_distance_squared(
        idx_i,
        idx_j,
        shift,
        structure.positions.reshape(-1, 3),
        shared_lattice=shared_lattice,
        system_lattices=system_lattices,
        atoms_per_system=structure.numbers.shape[-1],
    )

    if torch.any(distance_squared < threshold * threshold):
        raise StructureError("Too close interatomic distances found")

    return None


def content_checks(
    numbers: Tensor,
    positions: Tensor,
    max_element: int = pse.MAX_ELEMENT,
    allow_batched: bool = True,
) -> bool | NoReturn:
    """
    Check the content of the numbers and positions tensors.

    Raises on failure, so call it as a plain statement: wrapping it in
    ``assert`` would skip the check under ``python -O``.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers for all atoms in the system of shape ``(..., nat)``.
    positions : Tensor
        Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
    max_element : int, optional
        Maximum atomic number allowed. Defaults to
        :data:`tad_mctc.data.pse.MAX_ELEMENT`.
    allow_batched : bool, optional
        Allow batched tensors. Defaults to ``True``.

    Returns
    -------
    bool
        ``True`` if content is correct.

    Raises
    ------
    StructureError
        Atomic number too large or too small.
    """
    if not is_functorch_tensor(numbers):
        if numbers.max() > max_element:
            raise StructureError(
                f"Atomic number larger than {max_element} found."
            )

    if allow_batched is False:
        if numbers.min() < 1:
            raise StructureError(
                "Atomic number smaller than 1 found. This may indicate "
                "residual padding. Remove before writing to file."
            )

    return True


def deflatable_check(
    positions: Tensor, fileobj: IO[Any] | None = None, **kwargs: Any
) -> bool | NoReturn:
    """
    Check for the last coordinate being at the origin as this might clash with
    padding.

    Raises on failure (with ``raise_padding_exception=True``), so call it as
    a plain statement: wrapping it in ``assert`` would skip the check under
    ``python -O``. With ``shift_for_last=True``, a clash shifts
    ``positions`` in place.

    Parameters
    ----------
    positions : Tensor
        A 2D tensor of shape ``(nat, 3)`` containing atomic positions.
    fileobj : IO[Any] | None, optional
        The file-like object from which is read (only for printing).

    Returns
    -------
    bool
        True if content is correct.

    Raises
    ------
    StructureError
        Padding clashes with coordinates. Requires the keyword argument
        ``raise_padding_exception=True``.
    """
    # collect the padding value
    pad = kwargs.pop("padding_value", 0)

    # do not deflate the coordinate axis (all z-coordinates could be zero)
    dpos = deflate(positions, value=pad, axis=1)
    if dpos.shape != positions.shape:
        msg = (
            f"The position tensor from '{fileobj}' cannot handle the padding "
            f"value '{pad}'. This commonly occurs for zero-padding if the last "
            "atom is in the origin."
        )

        # raise exception
        if kwargs.pop("raise_padding_exception", False):
            raise StructureError(msg)

        # shift all atoms
        if kwargs.pop("shift_for_last", False):
            positions += kwargs.pop("shift_value", 1.0)
            return True

        # issue warning
        if kwargs.pop("raise_padding_warning", True):
            # pylint: disable=import-outside-toplevel
            from warnings import warn

            warn(msg, StructureWarning)

    return True


def dimension_check(
    x: Any,
    min_ndim: int = -1,
    max_ndim: int = 9999,
) -> bool | NoReturn:
    """
    Check if the number of dimensions of a tensor is within a certain range.

    Parameters
    ----------
    x : Any
        The tensor to check.
    min_ndim : int, optional
        Minimum number of dimensions for the tensor. Defaults to ``-1``.
    max_ndim : int, optional
        Maximum number of dimensions for the tensor. Defaults to ``9999``.

    Returns
    -------
    None | NoReturn
        Returns ``None`` if the tensor has the correct number of dimensions.

    Raises
    ------
    TypeError
        If the input is not a tensor.
    RuntimeError
        If the number of dimensions is not within the specified range.

    Examples
    --------
    >>> import torch
    >>> from tad_mctc.io.checks.structure import dimension_check
    >>> x = torch.tensor([1, 2, 3])
    >>> dimension_check(x, min_ndim=1, max_ndim=1)
    True
    >>> dimension_check(x, min_ndim=2, max_ndim=2)
    Traceback (most recent call last):
    ...
    RuntimeError: The tensor should not fall below 2 dimensions.
    """
    if not isinstance(x, Tensor):
        raise TypeError(f"Variable is not a tensor but '{type(x)}'.")

    if x.ndim < min_ndim:
        raise RuntimeError(
            f"The tensor should not fall below {min_ndim} dimensions."
        )
    if x.ndim > max_ndim:
        raise RuntimeError(
            f"The tensor should not exceed {max_ndim} dimensions."
        )

    return True


def structure_check(
    numbers: Tensor,
    positions: Tensor,
    charge: Tensor | None = None,
    uhf: Tensor | None = None,
    lattice: Tensor | None = None,
    periodic: Tensor | None = None,
    bonds: Tensor | None = None,
    bond_orders: Tensor | None = None,
) -> bool | NoReturn:
    """
    Check a :class:`~tad_mctc.io.structure.Structure`'s tensors for consistent
    shape, dtype and device. A plain function over the tensor fields, so
    any caller can validate them without building a ``Structure``.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers for all atoms in the system of shape ``(..., nat)``.
    positions : Tensor
        Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
    charge : Tensor | None, optional
        Total charge, shape ``(...,)`` or scalar. Not checked if ``None``.
    uhf : Tensor | None, optional
        Number of unpaired electrons, shape ``(...,)`` or scalar. Not
        checked if ``None``.
    lattice : Tensor | None, optional
        Lattice vectors as rows, shape ``(..., 3, 3)``. Not checked (the
        structure is not periodic) if ``None``.
    periodic : Tensor | None, optional
        Boolean mask marking periodic lattice axes, shape ``(3,)`` or the
        lattice's ``(..., 3)``. Requires ``lattice``. Not checked if
        ``None``.
    bonds : Tensor | None, optional
        Atom-index pairs describing bond connectivity, shape
        ``(..., nbond, 2)``. Not checked if ``None``.
    bond_orders : Tensor | None, optional
        Bond order per entry in ``bonds``, shape ``(..., nbond)``. Only
        meaningful alongside ``bonds`` -- giving one without the other is
        an error, since a bond order without the atom pair it belongs to
        is meaningless.

    Returns
    -------
    bool
        ``True`` if all checks pass.

    Raises
    ------
    RuntimeError
        A tensor has the wrong number of dimensions or an inconsistent
        shape, ``bond_orders`` is given without ``bonds``, or
        ``periodic`` without ``lattice``.
    DtypeError
        ``numbers``, ``periodic`` or ``bonds`` has the wrong dtype.
    DeviceError
        The tensors do not all live on the same device.
    """
    dimension_check(numbers, min_ndim=1, max_ndim=2)
    dimension_check(positions, min_ndim=2, max_ndim=3)

    if charge is not None:
        dimension_check(charge, min_ndim=0, max_ndim=1)

    if uhf is not None:
        dimension_check(uhf, min_ndim=0, max_ndim=1)

    if lattice is not None:
        dimension_check(lattice, min_ndim=2, max_ndim=3)
        if tuple(lattice.shape[-2:]) != (3, 3):
            raise RuntimeError(
                "Lattice vectors must be given as a '(..., 3, 3)' tensor, "
                f"but shape is '{tuple(lattice.shape)}'."
            )

    if periodic is not None:
        dimension_check(periodic, min_ndim=1, max_ndim=2)
        if periodic.shape[-1] != 3:
            raise RuntimeError(
                "Periodicity mask must be given as a '(..., 3)' tensor, "
                f"but shape is '{tuple(periodic.shape)}'."
            )
        if periodic.dtype != torch.bool:
            raise DtypeError(
                "Dtype of periodicity mask must be 'torch.bool', but is "
                f"'{periodic.dtype}'."
            )
        if lattice is None:
            raise RuntimeError(
                "'periodic' was given without 'lattice': a periodicity mask "
                "without the lattice vectors it refers to is meaningless."
            )
        if periodic.shape != (3,) and periodic.shape != lattice.shape[:-1]:
            raise RuntimeError(
                "Periodicity mask must be a '(3,)' tensor or match the "
                f"lattice's batch shape '{tuple(lattice.shape[:-1])}', but "
                f"shape is '{tuple(periodic.shape)}'."
            )

    if bond_orders is not None and bonds is None:
        raise RuntimeError(
            "'bond_orders' was given without 'bonds': a bond order without "
            "the atom-index pair it belongs to is meaningless."
        )

    if bonds is not None:
        dimension_check(bonds, min_ndim=2, max_ndim=3)
        if bonds.shape[-1] != 2:
            raise RuntimeError(
                "Bonds must be given as a '(..., nbond, 2)' tensor of "
                f"atom-index pairs, but shape is '{tuple(bonds.shape)}'."
            )
        allowed_bond_dtypes = (torch.long, torch.int16, torch.int32)
        if bonds.dtype not in allowed_bond_dtypes:
            raise DtypeError(
                "Dtype of bond indices must be one of the following to "
                f"allow indexing: "
                f"'{', '.join(str(x) for x in allowed_bond_dtypes)}', but "
                f"is '{bonds.dtype}'."
            )

        if bond_orders is not None:
            dimension_check(bond_orders, min_ndim=1, max_ndim=2)
            if bond_orders.shape[-1] != bonds.shape[-2]:
                raise RuntimeError(
                    f"Number of bond orders ({bond_orders.shape[-1]}) does "
                    f"not match the number of bonds ({bonds.shape[-2]})."
                )

    allowed_dtypes = (torch.long, torch.int16, torch.int32, torch.int64)
    if numbers.dtype not in allowed_dtypes:
        raise DtypeError(
            "Dtype of atomic numbers must be one of the following to allow "
            f"indexing: '{', '.join(str(x) for x in allowed_dtypes)}', "
            f"but is '{numbers.dtype}'"
        )

    optional_tensors = (
        charge,
        uhf,
        lattice,
        periodic,
        bonds,
        bond_orders,
    )
    all_tensors = (numbers, positions) + optional_tensors
    devices = {t.device for t in all_tensors if isinstance(t, Tensor)}
    if len(devices) > 1:
        raise DeviceError("All tensors must be on the same device!")

    if numbers.shape != positions.shape[:-1]:
        raise RuntimeError(
            f"Shape of positions ({positions.shape[:-1]}) is not "
            f"consistent with atomic numbers ({numbers.shape})."
        )

    return True
