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
>>> from tad_mctc.neighbor import pairs_within_radii
>>> structure = Structure(
...     numbers=torch.tensor([1, 1, 1]),
...     positions=torch.tensor([[0.0, 0, 0], [0, 0, 1.4], [0, 0, 4.0]]),
... )
>>> i, j = pairs_within_radii(structure, torch.tensor([1.0, 1.0, 3.0]))
>>> sorted(sorted(pair) for pair in zip(i.tolist(), j.tolist()))
[[0, 1], [1, 2]]
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

__all__ = ["pairs_within_radii"]


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
