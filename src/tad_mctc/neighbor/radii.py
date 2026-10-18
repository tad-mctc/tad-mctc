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
r"""
Neighbour search: Per-atom radii
================================

Pairs of atoms within a radius of their own, combined in quadrature:
:math:`R_{ij}^2 \le r_i^2 + r_j^2`.

This is the rule for products of Gaussian functions. The product of
:math:`e^{-a |\mathbf{r} - \mathbf{R}_i|^2}` and
:math:`e^{-b |\mathbf{r} - \mathbf{R}_j|^2}` carries the factor
:math:`e^{-ab R_{ij}^2 / (a + b)}`, which is at least :math:`e^{-T}` exactly
if :math:`R_{ij}^2 \le T/a + T/b`. With the radius :math:`r = \sqrt{T/a}` of
the most diffuse function of each atom, the pairs within the radii are the
pairs of atoms whose functions overlap by more than :math:`e^{-T}`, e.g. the
shell pairs worth evaluating in an integral code.

Example
-------
>>> import torch
>>> from tad_mctc.io.structure import Structure
>>> from tad_mctc.neighbor import pairs_within_radii, periodic_pairs_within_radii
>>> structure = Structure(
...     numbers=torch.tensor([1, 1, 1]),
...     positions=torch.tensor([[0.0, 0, 0], [0, 0, 1.4], [0, 0, 4.0]]),
... )
>>> i, j = pairs_within_radii(structure, torch.tensor([1.0, 1.0, 3.0]))
>>> sorted(sorted(pair) for pair in zip(i.tolist(), j.tolist()))
[[0, 1], [1, 2]]

In a periodic structure, the pairs are between an atom and the images of the
others (and of itself), each with its integer image shift:

>>> cell = Structure(
...     numbers=torch.tensor([1]),
...     positions=torch.zeros((1, 3)),
...     lattice=3.0 * torch.eye(3),
... )
>>> i, j, shift = periodic_pairs_within_radii(cell, torch.tensor([2.5]))
>>> sorted(shift.abs().tolist())  # one of the images n and -n each
[[0, 0, 1], [0, 1, 0], [1, 0, 0]]
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from ..autograd.unwrap import unwrap_gradtracking
from ..typing import Tensor
from .list import SearchKernelName, _outside_transforms, build_neighborlist

if TYPE_CHECKING:
    from ..io.structure import Structure  # circular import at runtime

__all__ = ["pairs_within_radii", "periodic_pairs_within_radii"]


@torch.compiler.disable
@torch.no_grad()
def pairs_within_radii(
    structure: Structure,
    radii: Tensor,
    *,
    tile: int = 32,
    distance_kernel: SearchKernelName | None = None,
) -> tuple[Tensor, Tensor]:
    r"""
    Pairs of atoms :math:`i, j` within their radii added in quadrature,
    :math:`R_{ij}^2 \le r_i^2 + r_j^2`.

    The pairs are found by a neighbour search at the largest possible
    distance (:func:`.build_neighborlist` at :math:`\sqrt{2} \max r`),
    then filtered pair by pair. Like the neighbour search, this is
    construction-side code: the number of pairs depends on the data, it
    reads the values of the tensors (also inside ``jacrev`` and friends)
    and carries no gradient.

    Parameters
    ----------
    structure : Structure
        One molecule, ``numbers`` of shape ``(nat,)``, or a batch,
        ``(batch, nat)``, with ``0`` marking padding atoms.
    radii : Tensor
        Radius of every atom, shape of ``numbers``. Ignored for padding.
    tile : int, optional
        Maximum atoms per tile of the neighbour search.
    distance_kernel : SearchKernelName | None, optional
        Kernel of the neighbour search, e.g. ``"native"`` for the native
        CPU extension; see :func:`.build_neighborlists`.

    Returns
    -------
    tuple[Tensor, Tensor]
        First and second atom of each pair, both ``(npairs,)`` and
        ``torch.long``. Each pair occurs once, in no particular order and
        without pairs of an atom with itself. Atoms of a batch are numbered
        flattened, ``b * nat + i``, as in :class:`.NeighborList`.

    Raises
    ------
    ValueError
        If ``radii`` does not have the shape of ``numbers``, a radius of a
        real atom is negative or not finite, or the structure is periodic.
    RuntimeError
        If ``structure`` is batched by ``torch.func.vmap``.
    """
    if structure.lattice is not None:
        raise ValueError("Periodic structures are not supported.")
    if radii.shape != structure.numbers.shape:
        raise ValueError(
            f"Expected one radius per atom, shape "
            f"{tuple(structure.numbers.shape)}, got {tuple(radii.shape)}."
        )

    radii = unwrap_gradtracking(radii)
    real_radii = radii[structure.numbers != 0]
    if not torch.all(torch.isfinite(real_radii) & (real_radii >= 0)):
        raise ValueError("Radii must be finite and non-negative.")

    # Everything below runs outside the transforms: in older PyTorch (2.7),
    # operations on plain tensors inside `jacrev` return wrapped ones.
    with _outside_transforms(structure) as plain:
        if real_radii.numel() == 0 or real_radii.max() == 0:
            no_pairs = torch.zeros(0, dtype=torch.long, device=radii.device)
            return no_pairs, no_pairs

        # no pair is further apart than sqrt(2) times the largest radius
        cutoff = math.sqrt(2.0) * real_radii.max().item()
        neighbors = build_neighborlist(
            plain, cutoff, tile=tile, distance_kernel=distance_kernel
        )
        i, j = (idx.long() for idx in neighbors.real_entries()[:2])

        positions = plain.positions.reshape(-1, 3)
        distance2 = (positions[i] - positions[j]).pow(2).sum(-1)
        radii2 = radii.reshape(-1) ** 2
        within = distance2 <= radii2[i] + radii2[j]
        return i[within], j[within]


@torch.compiler.disable
@torch.no_grad()
def periodic_pairs_within_radii(
    structure: Structure,
    radii: Tensor,
    *,
    tile: int = 32,
    distance_kernel: SearchKernelName | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    r"""
    Pairs of an atom :math:`i` and a periodic image of an atom :math:`j`
    within their radii added in quadrature,
    :math:`|\mathbf{R}_j + \mathbf{n} \mathbf{A} - \mathbf{R}_i|^2 \le
    r_i^2 + r_j^2`, with the lattice vectors :math:`\mathbf{A}` as rows.

    The periodic counterpart of :func:`pairs_within_radii`, with the same
    neighbour search (:func:`.build_neighborlist`, at :math:`\sqrt{2}
    \max r`), along the periodic axes of each structure only. Like the
    neighbour search, this is construction-side code: the number of pairs
    depends on the data, it reads the values of the tensors (also inside
    ``jacrev`` and friends) and carries no gradient.

    As in :class:`.NeighborList`, each pair stands for both directions,
    :math:`(i, j, \mathbf{n})` and :math:`(j, i, -\mathbf{n})`, and occurs
    once. This includes the images of an atom itself (:math:`i = j`,
    :math:`\mathbf{n} \ne 0`): of :math:`\mathbf{n}` and
    :math:`-\mathbf{n}`, only one occurs. The atom itself
    (:math:`i = j`, :math:`\mathbf{n} = 0`) does not occur.

    Parameters
    ----------
    structure : Structure
        One periodic structure, ``numbers`` of shape ``(nat,)``, or a
        batch, ``(batch, nat)``, with ``0`` marking padding atoms.
    radii : Tensor
        Radius of every atom, shape of ``numbers``. Ignored for padding.
    tile : int, optional
        Maximum atoms per tile of the neighbour search.
    distance_kernel : SearchKernelName | None, optional
        Kernel of the neighbour search, e.g. ``"native"`` for the native
        CPU extension; see :func:`.build_neighborlists`.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        First and second atom of each pair, both ``(npairs,)`` and
        ``torch.long``, and the integer image shift :math:`\mathbf{n}` of
        the second atom, ``(npairs, 3)`` and ``torch.long``, in no
        particular order. The shifts refer to the given positions (not
        folded into the cell). Atoms of a batch are numbered flattened,
        ``b * nat + i``, as in :class:`.NeighborList`.

    Raises
    ------
    ValueError
        If the structure is not periodic (has no ``lattice``), ``radii``
        does not have the shape of ``numbers``, or a radius of a real atom
        is negative or not finite.
    RuntimeError
        If ``structure`` is batched by ``torch.func.vmap``.
    """
    if structure.lattice is None:
        raise ValueError(
            "The structure is not periodic (no `lattice`); use "
            "`pairs_within_radii`."
        )
    if radii.shape != structure.numbers.shape:
        raise ValueError(
            f"Expected one radius per atom, shape "
            f"{tuple(structure.numbers.shape)}, got {tuple(radii.shape)}."
        )

    radii = unwrap_gradtracking(radii)
    real_radii = radii[structure.numbers != 0]
    if not torch.all(torch.isfinite(real_radii) & (real_radii >= 0)):
        raise ValueError("Radii must be finite and non-negative.")

    with _outside_transforms(structure) as plain:
        if real_radii.numel() == 0 or real_radii.max() == 0:
            no_pairs = torch.zeros(0, dtype=torch.long, device=radii.device)
            no_shifts = torch.zeros(
                (0, 3), dtype=torch.long, device=radii.device
            )
            return no_pairs, no_pairs, no_shifts

        # no pair is further apart than sqrt(2) times the largest radius
        cutoff = math.sqrt(2.0) * real_radii.max().item()
        neighbors = build_neighborlist(
            plain, cutoff, tile=tile, distance_kernel=distance_kernel
        )
        idx_i, idx_j, shift = neighbors.real_entries()
        i, j, shift = idx_i.long(), idx_j.long(), shift.long()

        # the lattice of each pair's structure
        lattice = plain.lattice
        assert lattice is not None
        if lattice.ndim == 3:
            lattice = lattice[i // plain.numbers.shape[-1]]
        translation = (shift.to(lattice.dtype).unsqueeze(-2) @ lattice).squeeze(
            -2
        )

        positions = plain.positions.reshape(-1, 3)
        distance2 = (positions[j] + translation - positions[i]).pow(2).sum(-1)
        radii2 = radii.reshape(-1) ** 2
        within = distance2 <= radii2[i] + radii2[j]
        return i[within], j[within], shift[within]
