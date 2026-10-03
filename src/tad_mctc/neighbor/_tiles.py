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
Neighbour search: Tiles
========================

Grouping of atoms into spatially bounded tiles of at most ``tile`` atoms
each, and the exact bounding-box screen that turns a bin grid into a list
of candidate tile pairs.

Tiles are cut out of a fine grid of bins rather than off a Morton-sorted
curve. A Morton (Z-order) cut gives a near-ideal *median* tile radius, but
the curve jumps at octree boundaries, so a handful of tiles end up spanning
most of the box. Downstream, the tile-pair stencil has to be wide enough
for the *largest* tile present, so a few oversized tiles quietly widen the
search for every tile in the system. Cutting tiles out of bins instead
bounds every tile's extent by one bin diagonal, by construction, so the
stencil width stops growing with system size.

Neither :class:`.Tiles` nor :func:`tile_pairs` carries a gradient: both are
data-dependent (`argsort`, `bincount`, boolean masks) and are meant to run
under ``torch.no_grad()`` as part of neighbour-list *construction*, not
*consumption*.

Example
-------
>>> import torch
>>> from tad_mctc.neighbor._tiles import Tiles, tile_pairs
>>>
>>> positions = torch.tensor([
...     [0.0, 0.0, 0.0],
...     [1.0, 0.0, 0.0],
...     [0.0, 1.0, 0.0],
...     [10.0, 10.0, 10.0],
... ])
>>> tiles = Tiles(positions, tile=2)
>>> a, b = tile_pairs(tiles, cutoff=2.0)
>>> bool((a <= b).all())
True
"""

from __future__ import annotations

import math

import torch

from ..typing import Tensor

__all__ = ["Tiles", "tile_pairs"]


# A planar or linear system has a bounding-box extent of exactly zero along
# one or more axes. Clamping the extent before it enters the bin-volume
# formula keeps the bin width finite; see step 1 of `Tiles.__init__`. The
# floor only has to be positive, not physically meaningful: the pair set
# returned by `tile_pairs` comes from an exact bounding-box test, so any
# positive bin width yields the same pairs, only at a different cost.
_MIN_EXTENT = 1e-6


class Tiles:
    """
    Atoms grouped into spatially bounded tiles of at most ``tile`` atoms.

    Attributes
    ----------
    index : Tensor
        ``(ntile, tile)``, the atom index held by each tile slot. Padded
        slots repeat the tile's first (real) atom index, so that indexing
        ``positions`` with ``index`` never reads out of bounds.
    valid : Tensor
        ``(ntile, tile)``, ``True`` for a slot holding a real atom.
    lo : Tensor
        ``(ntile, 3)``, the bounding-box minimum of each tile, ignoring
        padded slots.
    hi : Tensor
        ``(ntile, 3)``, the bounding-box maximum of each tile, ignoring
        padded slots.
    """

    __slots__ = [
        "tile",
        "nat",
        "width",
        "n_axis",
        "strides",
        "ncells",
        "ntile",
        "index",
        "valid",
        "lo",
        "hi",
        "tile_cc",
    ]

    def __init__(
        self,
        positions: Tensor,
        tile: int = 32,
        max_bin_width: Tensor | None = None,
    ) -> None:
        """
        Parameters
        ----------
        positions : Tensor
            Cartesian coordinates, shape ``(nat, 3)``, of a single system.
        tile : int, optional
            Maximum number of atoms per tile. Defaults to ``32``.
        max_bin_width : Tensor | None, optional
            ``(3,)``, an optional per-axis upper bound the coarsening loop
            (Step 2 below) may never cross, even if honouring it means
            exceeding the ``tile``-derived ``max_cells`` budget. ``None``
            (default) applies no bound. A batched search
            (:func:`tad_mctc.neighbor.list._tiles_of_separated_systems`)
            passes the guaranteed minimum gap between two adjacent systems
            (at least ``2 * search_cutoff``, see
            :func:`~tad_mctc.neighbor.list._separate_systems`) along the one
            axis the systems are shifted along, so that the coarsening loop
            can never widen a bin enough to merge atoms from two systems
            into it. An axis with no bound should carry ``float("inf")``.
        """
        nat = positions.shape[0]
        self.tile = tile
        self.nat = nat

        if nat == 0:
            self._build_empty(positions, tile)
            return

        lo_box = positions.min(dim=0).values
        hi_box = positions.max(dim=0).values
        extent = hi_box - lo_box

        # Step 1: bin width targeting one tile per bin. The extent feeding
        # the volume is floored (see `_MIN_EXTENT`) so a planar or linear
        # system does not collapse the box volume, and hence the bin
        # width, to zero. This is only a *target* width used to pick
        # `n_axis` below (Step 2); the bin width actually used to place
        # atoms (Step 3) is the per-axis width recomputed from the final
        # `n_axis`, not this scalar.
        extent_for_volume = extent.clamp_min(_MIN_EXTENT)
        volume = extent_for_volume.prod()
        target_width = float((volume * tile / nat) ** (1.0 / 3.0))

        # Step 2: bins per axis, bounded. `ceil(extent / target_width)`
        # alone is not a defensive fallback, it is required: on a thin
        # slab or a strongly elongated box it produces a finite but
        # enormous grid, and on a perfectly planar or linear system
        # (extent exactly zero along an axis) it produces zero bins along
        # that axis. Clamping each axis to at least one bin handles both:
        # a zero-extent axis collapses to a single bin, and the halving
        # loop below bounds the total bin count on an elongated box.
        n_axis = torch.ceil(extent / target_width).long().clamp_min(1)

        # Per-axis floor on the number of bins, derived from
        # `max_bin_width`: enough bins that `extent_for_volume / n_axis`
        # (the per-axis width Step 3 below recomputes) stays strictly
        # below the caller's bound, not just at or under it -- `floor(...)
        # + 1` rather than `ceil(...)` so an exact division does not leave
        # the width sitting exactly on the boundary. An unconstrained axis
        # (`max_bin_width` entry `inf`, or `max_bin_width is None`) floors
        # to `1`, i.e. no constraint at all, matching every caller that
        # does not pass this argument.
        min_axis = torch.ones_like(n_axis)
        if max_bin_width is not None:
            min_axis = torch.clamp(
                torch.floor(extent_for_volume / max_bin_width).long() + 1,
                min=1,
            )
        n_axis = torch.maximum(n_axis, min_axis)

        max_cells = 4 * math.ceil(nat / tile)
        while int(n_axis.prod().item()) > max_cells:
            coarsened = torch.clamp(n_axis // 2, min=min_axis)
            if torch.equal(coarsened, n_axis):
                # Every axis is already at its `min_axis` floor: further
                # coarsening would violate `max_bin_width` on at least one
                # axis, so the loop stops here even though `max_cells` is
                # still exceeded, trading search cost for correctness.
                # Without `max_bin_width`, `min_axis` is all ones and this
                # only trips at a single bin, `n_axis == [1, 1, 1]`.
                break
            n_axis = coarsened
        self.n_axis = n_axis

        # Per-axis bin width that exactly tiles `extent_for_volume` into
        # `n_axis` bins, recomputed from the *final* `n_axis` rather than
        # reused from Step 1's scalar target. `target_width` alone would
        # under-cover the box whenever `n_axis` was halved by the loop
        # above (the grid is coarser than one `target_width`-sized bin per
        # axis), and even without any halving `ceil(extent / target_width)`
        # can itself leave slack on the last bin along an axis. Either way
        # the stale scalar width would let the Step 3 clamp below silently
        # fold atoms that are many bins away into the last bin along an
        # axis, not just genuine floating-point edge cases -- this
        # per-axis recomputation is what keeps every tile's extent bounded
        # by one (possibly anisotropic) bin diagonal of the grid that was
        # actually built, matching the module docstring's cost-bound
        # design for the coarsened grid too.
        width = extent_for_volume / n_axis
        self.width = width

        strides = torch.stack(
            [
                n_axis[1] * n_axis[2],
                n_axis[2],
                torch.ones_like(n_axis[2]),
            ]
        )
        self.strides = strides
        ncells = int(n_axis.prod().item())
        self.ncells = ncells

        # Step 3: assign atoms to bins. The raw cell coordinate is clamped
        # into the grid; with `width` now matching `n_axis` exactly (one
        # bin width per axis over the whole extent), the clamp only ever
        # catches genuine floating-point edge cases (a position exactly at
        # `hi_box`), never atoms that are actually many bins away.
        raw_cc = torch.floor((positions - lo_box) / width).long()
        cc = torch.clamp(raw_cc, min=torch.zeros_like(n_axis), max=n_axis - 1)
        cell_id = (cc * strides).sum(-1)

        order = torch.argsort(cell_id)
        cell_id_sorted = cell_id[order]
        counts = torch.bincount(cell_id_sorted, minlength=ncells)
        cumulative_counts = torch.cumsum(counts, dim=0)
        bin_start = torch.cat([counts.new_zeros(1), cumulative_counts])[:ncells]

        # Step 4: cut tiles within each bin, not across bin boundaries. A
        # bin of `n` atoms gives `ceil(n / tile)` tiles.
        rank_in_bin = (
            torch.arange(nat, device=positions.device)
            - bin_start[cell_id_sorted]
        )
        ntile_per_bin = (counts + tile - 1) // tile
        tile_start = torch.cat(
            [ntile_per_bin.new_zeros(1), torch.cumsum(ntile_per_bin, dim=0)]
        )
        ntile = int(tile_start[-1].item())
        self.ntile = ntile

        tile_id = tile_start[cell_id_sorted] + rank_in_bin // tile
        slot = rank_in_bin % tile

        index = torch.zeros(
            ntile, tile, dtype=torch.long, device=positions.device
        )
        valid = torch.zeros(
            ntile, tile, dtype=torch.bool, device=positions.device
        )
        index[tile_id, slot] = order
        valid[tile_id, slot] = True

        # Padded slots repeat the tile's own first atom, a real index, so
        # that indexing `positions` with `index` is always safe. This is
        # unrelated to the phantom-row padding of the neighbour list
        # (`tad_mctc.neighbor.list`): there, padding must point at a
        # constant with no gradient path; here, it only has to be a
        # harmless duplicate for a bounding-box computation that ignores
        # it via `valid`.
        first_atom_of_tile = index[:, :1].expand_as(index)
        self.index = torch.where(valid, index, first_atom_of_tile)
        self.valid = valid

        # Step 5: per-tile bounding boxes, ignoring padded slots.
        tile_positions = positions[self.index]
        sentinel = torch.finfo(positions.dtype).max
        lo_masked = torch.where(valid.unsqueeze(-1), tile_positions, sentinel)
        hi_masked = torch.where(valid.unsqueeze(-1), tile_positions, -sentinel)
        self.lo = lo_masked.min(dim=1).values
        self.hi = hi_masked.max(dim=1).values

        # Bin coordinate shared by every atom in a tile (tiles never cross
        # a bin boundary), needed to bucket tiles by bin in `tile_pairs`.
        self.tile_cc = cc[self.index[:, 0]]

    def _build_empty(self, positions: Tensor, tile: int) -> None:
        """Fill in a degenerate, atom-free Tiles instance."""
        self.width = torch.ones(
            3, dtype=positions.dtype, device=positions.device
        )
        self.n_axis = torch.ones(3, dtype=torch.long, device=positions.device)
        self.strides = torch.tensor(
            [1, 1, 1], dtype=torch.long, device=positions.device
        )
        self.ncells = 1
        self.ntile = 0
        self.index = positions.new_zeros((0, tile), dtype=torch.long)
        self.valid = positions.new_zeros((0, tile), dtype=torch.bool)
        self.lo = positions.new_zeros((0, 3))
        self.hi = positions.new_zeros((0, 3))
        self.tile_cc = positions.new_zeros((0, 3), dtype=torch.long)


def _stencil_offsets(span: Tensor) -> Tensor:
    """
    Integer bin offsets covering a box of half-widths ``span`` per axis.

    Parameters
    ----------
    span : Tensor
        Per-axis half-width of the stencil, shape ``(3,)``.

    Returns
    -------
    Tensor
        Offsets, shape ``(prod(2 * span + 1), 3)``.
    """
    axis_ranges = []
    for axis in range(3):
        half_width = int(span[axis].item())
        axis_ranges.append(
            torch.arange(-half_width, half_width + 1, device=span.device)
        )

    grids = torch.meshgrid(*axis_ranges, indexing="ij")
    return torch.stack(grids, dim=-1).reshape(-1, 3)


def tile_pairs(
    tiles: Tiles, cutoff: float, anchors: Tensor | None = None
) -> tuple[Tensor, Tensor]:
    """
    Candidate tile pairs within ``cutoff``, screened by exact bounding-box
    separation.

    Tiles are already binned, so the stencil is the bin grid itself,
    widened per axis by how many bins ``cutoff`` can reach. No extra
    binning pass is needed, and the search width does not depend on the
    largest tile in the system (see the module docstring).

    Parameters
    ----------
    tiles : Tiles
        Tiles built by :class:`.Tiles`.
    cutoff : float
        Real-space cutoff.
    anchors : Tensor | None, optional
        ``(ntile,)``, ``True`` for the tiles that matter. Only tile pairs
        with at least one anchor tile are returned, and they are
        enumerated from the anchor tiles alone, so the cost follows the
        number of anchors instead of the number of tiles. A ghost pool
        passes the tiles that hold a primary-cell atom. ``None`` (default)
        returns every tile pair.

    Returns
    -------
    tuple[Tensor, Tensor]
        Tile indices ``a``, ``b`` with ``a <= b`` whose bounding boxes are
        within ``cutoff`` of each other.
    """
    if tiles.ntile == 0:
        empty = torch.zeros(0, dtype=torch.long, device=tiles.lo.device)
        return empty, empty

    # Two tiles whose bins are `k` apart along an axis are separated
    # along that axis by at least `(k - 1) * width`, not `k * width`,
    # because a tile can sit anywhere inside its own bin. `width` is
    # per-axis (the grid need not be isotropic), so the bound is computed
    # per axis too: `ceil(cutoff / width) + 1` would search too wide;
    # `cutoff // width + 1` is the tight bound along each axis. The width
    # is taken in at least float32: along a flat axis it is the tiny
    # `_MIN_EXTENT` floor, and `cutoff / width` would overflow float16 to
    # an infinity that `.long()` cannot convert.
    width_dtype = torch.promote_types(tiles.width.dtype, torch.float32)
    raw_span = (cutoff // tiles.width.to(width_dtype)).long() + 1

    # Offsets beyond the grid address bins that do not exist. In this
    # Cartesian (non-modular) binning they are simply empty, so this clamp
    # only avoids wasted stencil steps, it is not needed for correctness.
    span = torch.clamp(tiles.n_axis - 1, max=raw_span)
    offsets = _stencil_offsets(span)

    n_axis, strides = tiles.n_axis, tiles.strides
    ncells = tiles.ncells

    # The tiles a pair is enumerated from: every tile, unless `anchors`
    # narrows it to the anchor tiles, where every wanted pair is found.
    # The default path does no extra work for this.
    first_tiles = None if anchors is None else anchors.nonzero().squeeze(-1)
    first_cc = (
        tiles.tile_cc if first_tiles is None else tiles.tile_cc[first_tiles]
    )

    # Bucket tiles by bin, so a bin-pair stencil expands straight to tile
    # pairs without a second binning pass.
    tile_cell_id = (tiles.tile_cc * strides).sum(-1)
    order = torch.argsort(tile_cell_id)
    tile_cell_id_sorted = tile_cell_id[order]
    bin_ids = torch.arange(ncells, device=tile_cell_id.device)
    bin_start = torch.searchsorted(tile_cell_id_sorted, bin_ids)
    bin_stop = torch.searchsorted(tile_cell_id_sorted, bin_ids, right=True)

    neighbor_cc = first_cc.unsqueeze(-2) + offsets
    inside_grid = ((neighbor_cc >= 0) & (neighbor_cc < n_axis)).all(-1)
    neighbor_id = (neighbor_cc.clamp_min(0) * strides).sum(-1)
    neighbor_id = neighbor_id.clamp(max=ncells - 1)

    neighbor_start = bin_start[neighbor_id]
    neighbor_stop = bin_stop[neighbor_id]
    neighbor_count = torch.where(
        inside_grid,
        neighbor_stop - neighbor_start,
        torch.zeros_like(neighbor_id),
    )

    flat_counts = neighbor_count.reshape(-1)
    total_candidates = int(flat_counts.sum().item())
    owning_slot = torch.repeat_interleave(
        torch.arange(flat_counts.numel(), device=flat_counts.device),
        flat_counts,
    )
    slot_offset = torch.cumsum(flat_counts, dim=0) - flat_counts
    position_within_slot = (
        torch.arange(total_candidates, device=flat_counts.device)
        - slot_offset[owning_slot]
    )

    a = owning_slot // offsets.shape[0]
    if first_tiles is not None:
        a = first_tiles[a]
    flat_neighbor_start = neighbor_start.reshape(-1)[owning_slot]
    b = order[flat_neighbor_start + position_within_slot]

    keep_upper_triangle = a <= b
    if anchors is not None:
        # A pair of two anchor tiles is enumerated from both sides, and
        # the `a <= b` copy is kept. A pair with a tile that is not an
        # anchor is enumerated from its anchor only: keep it as well, and
        # put it in `a <= b` order.
        keep_upper_triangle |= ~anchors[b]
    a, b = a[keep_upper_triangle], b[keep_upper_triangle]
    if anchors is not None:
        a, b = torch.minimum(a, b), torch.maximum(a, b)

    # Exact bounding-box separation: the closest possible distance between
    # the two boxes along each axis, zero if they overlap on that axis.
    gap = (tiles.lo[a] - tiles.hi[b]).clamp_min(0.0)
    gap = gap + (tiles.lo[b] - tiles.hi[a]).clamp_min(0.0)
    gap_squared = (gap * gap).sum(-1)

    keep_within_cutoff = gap_squared <= cutoff * cutoff
    return a[keep_within_cutoff], b[keep_within_cutoff]
