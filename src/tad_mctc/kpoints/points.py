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
k-points: Container
===================

A set of k-points with their weights (:class:`KPoints`), and the conversion
between Cartesian and fractional coordinates.

A batch of structures has one set per structure. Sets of different size are
padded (:func:`pack_kpoints`): the padding k-points are masked out
(``mask`` is ``False``) and have zero weight. Neither a sentinel value
(:math:`\\mathbf{k} = 0` is the Γ point) nor a zero weight (band-structure
paths carry zero-weight k-points) can mark padding, hence the mask.

k-points belong to a calculation, not to the structure: one structure may
need several sets, e.g. a grid for the SCF and a path for the band structure.

Example
-------
>>> import torch
>>> from tad_mctc.kpoints import KPoints, pack_kpoints
>>> gamma = KPoints(points=torch.zeros((1, 3), dtype=torch.double))
>>> gamma.weights
tensor([1.], dtype=torch.float64)
>>> two = KPoints(points=torch.tensor([[0.0, 0, 0], [0.5, 0, 0]]))
>>> batch = pack_kpoints([gamma.type(torch.float), two])
>>> batch.mask
tensor([[ True, False],
        [ True,  True]])
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from ..batch import pack
from ..tree import Node, child
from ..typing import DD, Tensor

__all__ = [
    "KPoints",
    "pack_kpoints",
    "reciprocal_lattice",
    "to_cartesian",
    "to_fractional",
]


class KPoints(Node):
    """
    A set of k-points of one structure, or one set per structure of a batch.

    On construction, absent ``weights`` are filled in as uniform weights
    ``1/nk``, as in PySCF, and an absent ``mask`` as all ``True``. A
    ``weights`` tensor with another floating-point dtype than ``points`` is
    cast to the dtype of ``points``.

    Parameters
    ----------
    points : Tensor
        Cartesian k-points, shape ``([batch,] nk, 3)``, in 1/Bohr.
    weights : Tensor | None, optional
        Weight of each k-point, shape ``([batch,] nk)``. Defaults to
        ``1/nk``.
    mask : Tensor | None, optional
        ``True`` for a real k-point, ``False`` for padding, shape
        ``([batch,] nk)``, ``torch.bool``. Defaults to all ``True``.

    Raises
    ------
    RuntimeError
        ``points`` is not of shape ``([batch,] nk, 3)``, ``weights`` or
        ``mask`` does not match it, or ``mask`` is not boolean.
    """

    points: Tensor = child()
    weights: Tensor | None = child(default=None)
    mask: Tensor | None = child(default=None)

    def _normalize(self) -> dict[str, Tensor]:
        updates: dict[str, Tensor] = {}
        shape = self.points.shape[:-1]
        dd: DD = {"device": self.points.device, "dtype": self.points.dtype}

        if self.weights is None:
            nk = max(shape[-1], 1) if len(shape) > 0 else 1
            updates["weights"] = torch.full(shape, 1.0 / nk, **dd)
        elif (
            self.weights.is_floating_point()
            and self.weights.dtype != self.points.dtype
        ):
            updates["weights"] = self.weights.to(dtype=self.points.dtype)

        if self.mask is None:
            updates["mask"] = torch.ones(
                shape, dtype=torch.bool, device=self.points.device
            )
        return updates

    def _validate(self) -> None:
        if self.points.ndim not in (2, 3) or self.points.shape[-1] != 3:
            raise RuntimeError(
                "`points` must be of shape `([batch,] nk, 3)`, but shape is "
                f"{tuple(self.points.shape)}."
            )
        if not self.points.is_floating_point():
            raise RuntimeError(
                "`points` must be a floating-point tensor, but dtype is "
                f"'{self.points.dtype}'."
            )
        shape = self.points.shape[:-1]
        for name in ("weights", "mask"):
            value = getattr(self, name)
            if value is not None and value.shape != shape:
                raise RuntimeError(
                    f"`{name}` must be of shape {tuple(shape)} (one entry "
                    f"per k-point), but shape is {tuple(value.shape)}."
                )
        if self.mask is not None and self.mask.dtype != torch.bool:
            raise RuntimeError(
                "`mask` must be a `torch.bool` tensor, but dtype is "
                f"'{self.mask.dtype}'."
            )

    @property
    def nkpts(self) -> int:
        """Number of k-points (per structure, including padding)"""
        return self.points.shape[-2]


def pack_kpoints(kpoints: Sequence[KPoints]) -> KPoints:
    """
    Pack the k-point sets of several structures into one batched set.

    The points are zero-padded to the largest set, with zero weight and
    ``mask`` set to ``False`` for the padding. Padding is data-dependent:
    call this while building the input, outside any
    ``vmap``/``jacrev``/``torch.compile`` region, as
    :func:`~tad_mctc.io.structure.pack_structures`.

    Parameters
    ----------
    kpoints : Sequence[KPoints]
        Unbatched k-point sets, all on the same device and dtype.

    Returns
    -------
    KPoints
        One set with a leading batch dimension.

    Raises
    ------
    ValueError
        ``kpoints`` is empty, or one of the sets is already batched.
    """
    if len(kpoints) == 0:
        raise ValueError("Cannot pack an empty sequence of k-point sets.")
    if any(k.points.ndim != 2 for k in kpoints):
        raise ValueError("Only unbatched k-point sets can be packed.")

    # filled in on construction (see `KPoints`)
    weights = [k.weights for k in kpoints]
    masks = [k.mask for k in kpoints]
    assert all(w is not None for w in weights)
    assert all(m is not None for m in masks)
    return KPoints(
        points=pack([k.points for k in kpoints]),
        weights=pack(weights),  # type: ignore[arg-type]
        mask=pack(masks, value=False),  # type: ignore[arg-type]
    )


def reciprocal_lattice(lattice: Tensor) -> Tensor:
    """
    Reciprocal lattice vectors as rows, :math:`2\\pi \\mathbf{A}^{-T}`, so
    that :math:`\\mathbf{a}_i \\cdot \\mathbf{b}_j = 2\\pi \\delta_{ij}`.

    Parameters
    ----------
    lattice : Tensor
        Lattice vectors as rows, shape ``([batch,] 3, 3)``, in Bohr.

    Returns
    -------
    Tensor
        Reciprocal lattice vectors as rows, shape ``([batch,] 3, 3)``, in
        1/Bohr.
    """
    return 2.0 * math.pi * torch.linalg.inv(lattice).mT


def to_fractional(points: Tensor, lattice: Tensor) -> Tensor:
    """
    Fractional coordinates of Cartesian k-points,
    :math:`\\boldsymbol{\\kappa} = \\mathbf{k} \\mathbf{A}^T / 2\\pi`.

    Parameters
    ----------
    points : Tensor
        Cartesian k-points, shape ``([batch,] nk, 3)``, in 1/Bohr.
    lattice : Tensor
        Lattice vectors as rows, shape ``([batch,] 3, 3)``, in Bohr.

    Returns
    -------
    Tensor
        Fractional coordinates, shape ``([batch,] nk, 3)``.
    """
    return points @ lattice.mT / (2.0 * math.pi)


def to_cartesian(fractional: Tensor, lattice: Tensor) -> Tensor:
    """
    Cartesian k-points of fractional coordinates,
    :math:`\\mathbf{k} = \\boldsymbol{\\kappa} \\mathbf{B}`.

    Parameters
    ----------
    fractional : Tensor
        Fractional coordinates, shape ``([batch,] nk, 3)``.
    lattice : Tensor
        Lattice vectors as rows, shape ``([batch,] 3, 3)``, in Bohr.

    Returns
    -------
    Tensor
        Cartesian k-points, shape ``([batch,] nk, 3)``, in 1/Bohr.
    """
    return fractional @ reciprocal_lattice(lattice)
