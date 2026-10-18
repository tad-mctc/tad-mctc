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
k-points: Uniform grids
=======================

Uniform (Monkhorst-Pack) grids of k-points, as PySCF's
``Cell.make_kpts``: the same keywords, the same order of the points (the
first axis runs slowest, the Γ point first in a Γ-centred grid) and uniform
weights.

Along each axis with ``n`` points, the fractional coordinates are

- ``i/n`` for a Γ-centred grid (``with_gamma_point=True``, the default),
- ``(2i + 1 - n)/(2n)`` otherwise, the original Monkhorst-Pack grid, which
  contains Γ only for odd ``n``,

for ``i = 0, ..., n - 1``, optionally wrapped into ``[-1/2, 1/2)`` and
shifted by ``scaled_center``. Non-periodic axes take a single point at 0.

The points are formed from ``lattice`` with differentiable torch operations
and static Python values only, so the grid works under ``vmap``,
``jacrev`` and ``torch.compile(fullgraph=True)``.

Example
-------
>>> import torch
>>> from tad_mctc.kpoints import make_kpoints, to_fractional
>>> lattice = 4.0 * torch.eye(3, dtype=torch.double)
>>> kpts = make_kpoints(lattice, (2, 1, 1))
>>> to_fractional(kpts.points, lattice)
tensor([[0.0000, 0.0000, 0.0000],
        [0.5000, 0.0000, 0.0000]], dtype=torch.float64)
>>> kpts.weights
tensor([0.5000, 0.5000], dtype=torch.float64)
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from ..typing import DD, Tensor
from .points import KPoints, to_cartesian

__all__ = ["make_kpoints"]


def make_kpoints(
    lattice: Tensor,
    mesh: Sequence[int],
    *,
    periodic: Sequence[bool] | Tensor | None = None,
    with_gamma_point: bool = True,
    wrap_around: bool = False,
    scaled_center: Sequence[float] | None = None,
) -> KPoints:
    """
    Uniform grid of k-points with uniform weights.

    Parameters
    ----------
    lattice : Tensor
        Lattice vectors as rows, shape ``([batch,] 3, 3)``, in Bohr. A batch
        of lattices shares the mesh and gets one grid each.
    mesh : Sequence[int]
        Number of points along each lattice axis.
    periodic : Sequence[bool] | Tensor | None, optional
        Which axes are periodic, e.g. ``structure.periodic``, also of a
        batch, ``(batch, 3)``. The mesh must have one point along a
        non-periodic axis, and ``scaled_center`` no component; in a batch,
        which shares the mesh, along an axis that is not periodic in any one
        of the structures. Its values are read, so pass it only outside
        transforms and compiled code. Defaults to every axis periodic (no
        check).
    with_gamma_point : bool, optional
        Γ-centred grid (``i/n``), or the original Monkhorst-Pack grid
        (``(2i + 1 - n)/(2n)``). Defaults to ``True``, as in PySCF.
    wrap_around : bool, optional
        Wrap the fractional coordinates into ``[-1/2, 1/2)``. Defaults to
        ``False``.
    scaled_center : Sequence[float] | None, optional
        Shift of the grid in fractional coordinates. A Γ-centred grid is
        shifted (also with ``with_gamma_point=False``, as in PySCF).
        Defaults to no shift.

    Returns
    -------
    KPoints
        ``prod(mesh)`` Cartesian k-points per lattice, shape
        ``([batch,] nk, 3)``, in 1/Bohr, with weights ``1/nk``.

    Raises
    ------
    ValueError
        ``mesh`` does not have three positive entries, ``scaled_center``
        three entries, or a non-periodic axis has more than one point or a
        shift.
    """
    mesh = tuple(int(n) for n in mesh)
    if len(mesh) != 3 or min(mesh) < 1:
        raise ValueError(
            f"Expected three positive numbers of points, got {mesh}."
        )
    center = (0.0, 0.0, 0.0) if scaled_center is None else scaled_center
    center = tuple(float(c) for c in center)
    if len(center) != 3:
        raise ValueError(
            f"Expected a `scaled_center` of three entries, got {center}."
        )

    if periodic is not None:
        if isinstance(periodic, Tensor):
            axes = periodic.reshape(-1, 3).all(0).tolist()
        else:
            axes = [bool(p) for p in periodic]
        for axis, (n, c, p) in enumerate(zip(mesh, center, axes)):
            if not p and (n != 1 or c != 0.0):
                raise ValueError(
                    f"Axis {axis} is not periodic, but has {n} points and a "
                    f"shift of {c}. Use one point and no shift."
                )

    dd: DD = {"device": lattice.device, "dtype": lattice.dtype}
    axes_points = []
    for n in mesh:
        i = torch.arange(n, **dd)
        if with_gamma_point or scaled_center is not None:
            ks = i / n
        else:
            ks = (i + 0.5) / n - 0.5
        if wrap_around:
            ks = torch.where(ks >= 0.5, ks - 1.0, ks)
        axes_points.append(ks)

    fractional = torch.cartesian_prod(*axes_points).reshape(-1, 3)
    fractional = fractional + torch.tensor(center, **dd)
    if lattice.ndim == 3:
        fractional = fractional.expand(lattice.shape[0], -1, -1)
    return KPoints(points=to_cartesian(fractional, lattice))
