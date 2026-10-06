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
Distance kernels: how to compute one tile pair's squared distances
=====================================================================

``_atom_pairs_within_thresholds`` (:mod:`.list`) needs the exact squared
distance between every atom of tile A and every atom of tile B, per
candidate tile pair, and there is more than one way to compute that on a
given device, dtype and tile width -- measured to matter by 3-12x on
CPU and to reverse sign entirely across CUDA precisions. A
**distance kernel** is one such way: a name, a predicate saying when it
applies, and the computation itself. There are three:

* ``"triton"`` -- a hand-written GPU kernel, the same shape of kernel
  LASP-D3's CUDA and NVIDIA's Warp ``cluster_tile`` kernel already use for
  the same problem. It never materialises the oversized intermediate
  tensor that makes the broadcast form memory-bound, so it wins across
  every ``(dtype, tile_width, problem size)`` combination tried, by
  1.6-3.5x over whichever tensor-op formula was already winning at that
  point. Preferred whenever it is available: CUDA, and the optional
  ``triton`` dependency installed (``pip install tad_mctc[triton]``).
* ``"baddbmm"`` -- a batched-matmul quadratic expansion, CPU only (see
  :func:`_baddbmm_distance_squared` for why it is not extended to CUDA).
* ``"broadcast"`` -- the plain broadcast difference. Universally
  applicable, so it is always last in :data:`_KERNELS` and is what a
  device that is neither CPU nor CUDA (and CUDA without ``triton``) falls
  back to.

:func:`select_kernel` is the whole interface: it walks the list in
priority order and returns the first applicable one, or honours an
explicit ``force`` override. Each kernel's own applicability check is
self-contained, so a new kernel is one more :class:`DistanceKernel` in
:data:`_KERNELS`, without changes to this function.

This is deliberately a registry for *this one problem*, not a general
dispatch mechanism for the package.

The ``"triton"`` kernel is written in Triton (pure Python, JIT-compiled at
call time -- no C/CUDA extension to build, the same mechanism
``torch.compile``'s own CUDA backend already relies on). It is an
**optional accelerator**, not a required dependency of ``tad_mctc``:
``triton`` is only importable when installed explicitly, and it only ever
runs on CUDA -- :func:`is_available` reports whether both conditions hold.
Every entry point concerned with it is concerned with list *construction*
only (see :mod:`.list`'s own module docstring for that split) -- it always
runs under ``torch.no_grad()``, never inside
``vmap``/``jacrev``/``torch.compile(fullgraph=True)``, so it carries none
of the differentiability contract the rest of this package is strict
about; nothing here needs a backward pass. Because ``triton.jit``-decorated
code cannot be meaningfully type-checked or exercised without a CUDA
device, its untypeable lines carry their own targeted ``# type: ignore``
and lines only reachable with ``triton`` installed and a CUDA device
present carry their own targeted ``# pragma: no cover``, rather than
excluding this whole, otherwise ordinary module from mypy/pyright/coverage.

Example
-------
>>> import torch
>>> from tad_mctc.neighbor._distance_kernels import is_available, select_kernel
>>> is_available(torch.device("cpu"))
False
>>> select_kernel(torch.device("cpu")).name
'baddbmm'
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, NamedTuple

import torch

from .._version import __tversion__
from ..tools import is_compiling
from ..typing import Tensor

try:  # pragma: no cover - exercised only with the optional `triton` extra
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True

    @triton.jit
    def _pairwise_distance_squared_kernel(  # type: ignore[no-untyped-def]
        pa_ptr, pb_ptr, out_ptr, tile_width, BLOCK: tl.constexpr
    ):
        """One program per tile pair: load both tiles once, then compute
        every pairwise squared distance in registers -- the same shape of
        kernel LASP-D3's and NVIDIA's hand-written GPU kernels use for this
        exact problem. Only the final ``(tile_width, tile_width)`` block is
        ever written to global memory; the ``(tile_width, tile_width, 3)``
        difference this module's docstring cites as the broadcast form's
        cost never exists outside registers here.

        ``tl.arange`` needs a power-of-two length, so the program works on
        a ``BLOCK``-wide square (``tile_width`` rounded up to a power of
        two) and masks off the slots past ``tile_width``."""
        pid = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        in_tile = offs < tile_width
        base = pid * tile_width * 3

        ax = tl.load(pa_ptr + base + offs * 3 + 0, mask=in_tile, other=0.0)
        ay = tl.load(pa_ptr + base + offs * 3 + 1, mask=in_tile, other=0.0)
        az = tl.load(pa_ptr + base + offs * 3 + 2, mask=in_tile, other=0.0)

        bx = tl.load(pb_ptr + base + offs * 3 + 0, mask=in_tile, other=0.0)
        by = tl.load(pb_ptr + base + offs * 3 + 1, mask=in_tile, other=0.0)
        bz = tl.load(pb_ptr + base + offs * 3 + 2, mask=in_tile, other=0.0)

        dx = tl.expand_dims(ax, 1) - tl.expand_dims(bx, 0)
        dy = tl.expand_dims(ay, 1) - tl.expand_dims(by, 0)
        dz = tl.expand_dims(az, 1) - tl.expand_dims(bz, 0)
        dist_sq = dx * dx + dy * dy + dz * dz

        row = tl.expand_dims(offs, 1)
        col = tl.expand_dims(offs, 0)
        in_block = tl.expand_dims(in_tile, 1) & tl.expand_dims(in_tile, 0)
        out = out_ptr + pid * tile_width * tile_width
        tl.store(out + row * tile_width + col, dist_sq, mask=in_block)

except ImportError:
    TRITON_AVAILABLE = False

__all__ = [
    "DistanceKernel",
    "DistanceKernelName",
    "gather_index",
    "is_available",
    "pair_distance_squared",
    "pair_distance_squared_from_columns",
    "position_columns",
    "select_kernel",
    "split_lattice",
]

DistanceKernelName = Literal["triton", "baddbmm", "broadcast"]
"""The names of the kernels in :data:`_KERNELS`, which a neighbour-list
build can force with its ``distance_kernel`` argument."""


class DistanceKernel(NamedTuple):
    """
    One way to compute a tile pair's squared distances.

    Attributes
    ----------
    name : str
        Stable identifier, used by :func:`select_kernel`'s ``force``
        override and in its error messages.
    applicable : Callable[[torch.device], bool]
        Whether this kernel should be considered for a given device.
        Self-contained: a kernel decides for itself, so
        :func:`select_kernel` never needs to know why.
    compute : Callable[[Tensor, Tensor], Tensor]
        ``(positions_a, positions_b)``, each ``(chunk, tile_width, 3)``,
        to ``distance_squared``, ``(chunk, tile_width, tile_width)``.
    """

    name: str
    applicable: Callable[[torch.device], bool]
    compute: Callable[[Tensor, Tensor], Tensor]


def is_available(device: torch.device) -> bool:
    """
    Whether the ``"triton"`` kernel can run for ``device``.

    ``False`` whenever ``triton`` is not installed, or on any device other
    than CUDA -- Triton's GPU backend does not target CPU, and this
    module's CPU path already has its own, separately-measured formula
    (:func:`_baddbmm_distance_squared`) that does not need it.

    Parameters
    ----------
    device : torch.device
        Device the caller's tensors live on.

    Returns
    -------
    bool
        ``True`` only when ``triton`` imported successfully and ``device``
        is CUDA.
    """
    return TRITON_AVAILABLE and device.type == "cuda"


def pairwise_distance_squared(  # pragma: no cover
    positions_a: Tensor, positions_b: Tensor
) -> Tensor:
    """
    Exact squared distance between every atom of tile A and every atom of
    tile B, per tile pair, computed by the Triton kernel above.

    Parameters
    ----------
    positions_a : Tensor
        ``(chunk, tile_width, 3)``, one tile's worth of atom positions per
        candidate tile pair.
    positions_b : Tensor
        ``(chunk, tile_width, 3)``, matching ``positions_a``.

    Returns
    -------
    Tensor
        ``(chunk, tile_width, tile_width)``, squared distances.

    Raises
    ------
    RuntimeError
        If ``triton`` is not installed, or the tensors are not on CUDA --
        call :func:`is_available` first; this function does not fall back
        on its own.
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError(
            "pairwise_distance_squared requires the optional `triton` "
            "dependency (`pip install tad_mctc[triton]`); call "
            "`is_available(device)` before reaching here."
        )
    if positions_a.device.type != "cuda":
        raise RuntimeError(
            "pairwise_distance_squared only runs on CUDA; call "
            "`is_available(device)` before reaching here."
        )

    chunk, tile_width, _ = positions_a.shape
    out = torch.empty(
        chunk,
        tile_width,
        tile_width,
        dtype=positions_a.dtype,
        device=positions_a.device,
    )
    if chunk == 0:
        return out

    # `tl.arange` needs a power-of-two length, so the kernel works on the
    # smallest power of two >= `tile_width` and masks the rest.
    block = 1 << (tile_width - 1).bit_length()
    _pairwise_distance_squared_kernel[(chunk,)](
        positions_a.contiguous(),
        positions_b.contiguous(),
        out,
        tile_width,
        BLOCK=block,  # pyright: ignore[reportArgumentType]
    )
    return out


# Relative slack, in units of the machine epsilon of the positions' dtype,
# on the squared threshold. The kernels evaluate a squared distance in
# different arithmetic (the `baddbmm` expansion, the broadcast difference,
# Triton, the native extension), so a pair right at a threshold, e.g. the
# image of an atom one cell edge away for a cutoff that is a multiple of
# it, rounds to either side depending on the kernel. The slack makes all of
# them keep such a pair; a consumer wanting the exact cutoff re-masks the
# list with its own distances (as the dispersion models do).
_THRESHOLD_SLACK = 8.0


def squared_threshold(threshold: float, dtype: torch.dtype) -> float:
    """
    The squared `threshold` that a squared distance is compared against, with
    a few ulps of slack (see ``_THRESHOLD_SLACK``) so that the result at a
    tie does not depend on the distance kernel.
    """
    slack = _THRESHOLD_SLACK * torch.finfo(dtype).eps
    return float(threshold) * float(threshold) * (1.0 + slack)


def _baddbmm_distance_squared(
    positions_a: Tensor, positions_b: Tensor
) -> Tensor:
    """
    |a - b|^2 = |a|^2 + |b|^2 - 2 a.b, computed with one batched matmul
    instead of materialising the broadcast difference tensor, shape
    (chunk, tile_width, tile_width, 3): that broadcast subtraction (and
    the following square) each allocate 3x more elements than the
    (chunk, tile_width, tile_width) result actually needs.

    CPU-only: a consistent win across every combination measured
    (float32 and float64, 1 and 16 threads, tile_width 4 through 32) --
    3-12x at this module's default tile_width=32, the one clear
    regression found being a ~7% slowdown at the degenerate
    tile_width=2 with float64 and many threads, judged not worth a
    tile-width floor on top of the CPU-only gate. Deliberately not
    extended to CUDA: the win there depends on the GPU, precision and
    tile_width all at once and sometimes reverses sign rather than just
    shrinking -- on one card tested (RTX 4070), float64 wins only above
    roughly tile_width 24, while float32 *loses* to broadcast at every
    tile_width tried, including this module's default. No single
    threshold here is worth hard-coding from one consumer GPU.

    Only ever compared against ``threshold**2``, never square-rooted, so
    the tiny negative values floating-point cancellation can produce for
    near-coincident points are harmless -- they still compare as "within
    threshold", which is correct.
    """
    # Measure positions from each tile pair's own first atom. Only relative
    # positions matter, and this keeps `|a|` and `|b|` near the tile size,
    # which `|a|^2 + |b|^2 - 2 a.b` needs at float32: far from the origin
    # it cancels catastrophically and loses real pairs. The other kernels
    # subtract the positions directly, which needs no such shift.
    local_origin = positions_a[:, 0:1, :]
    positions_a = positions_a - local_origin
    positions_b = positions_b - local_origin

    squared_norm_a = (positions_a * positions_a).sum(-1)
    squared_norm_b = (positions_b * positions_b).sum(-1)
    squared_norm_sum = squared_norm_a.unsqueeze(2) + squared_norm_b.unsqueeze(1)
    return torch.baddbmm(
        squared_norm_sum,
        positions_a,
        positions_b.transpose(1, 2),
        alpha=-2.0,
        beta=1.0,
    )


def _broadcast_distance_squared(
    positions_a: Tensor, positions_b: Tensor
) -> Tensor:
    """
    Plain broadcast difference, squared and summed. The universal
    fallback: correct on every device, dtype and tile width, just not
    always the fastest one available (see :func:`_baddbmm_distance_squared`
    and :func:`pairwise_distance_squared` for the two that can beat it).
    """
    difference = positions_a.unsqueeze(2) - positions_b.unsqueeze(1)
    return (difference * difference).sum(-1)


_KERNELS: list[DistanceKernel] = [
    DistanceKernel(
        name="triton",
        applicable=is_available,
        compute=pairwise_distance_squared,
    ),
    DistanceKernel(
        name="baddbmm",
        applicable=lambda device: device.type == "cpu",
        compute=_baddbmm_distance_squared,
    ),
    DistanceKernel(
        name="broadcast",
        applicable=lambda device: True,
        compute=_broadcast_distance_squared,
    ),
]
"""Priority order: the first applicable kernel wins. ``"broadcast"`` is
last and unconditionally applicable, so this list always yields one."""


def select_kernel(
    device: torch.device,
    *,
    force: str | None = None,
    kernels: list[DistanceKernel] | None = None,
) -> DistanceKernel:
    """
    Pick the distance kernel to use on this device.

    Parameters
    ----------
    device : torch.device
        Device the positions live on.
    force : str | None, optional
        Skip priority selection and use the kernel with this
        :attr:`DistanceKernel.name` instead. ``None`` (default) selects
        the first applicable kernel in priority order.
    kernels : list[DistanceKernel] | None, optional
        The candidates to choose among, tried in list order. ``None``
        (default) uses the real, production kernels in :data:`_KERNELS`;
        a test passes its own list to check this function's selection
        logic in isolation, with no GPU and no real tensor computation.

    Returns
    -------
    DistanceKernel
        The chosen kernel.

    Raises
    ------
    ValueError
        If ``force`` names a kernel that is not in ``kernels``, or that
        is in ``kernels`` but reports itself not applicable for this
        device -- never silently substituted for
        another kernel.
    RuntimeError
        If no kernel in ``kernels`` is applicable and ``force`` is not
        given. Unreachable with the real, default kernels (`"broadcast"`
        is always applicable), so this only fires against a caller-
        supplied ``kernels`` list that omits a universal fallback.
    """
    candidates = _KERNELS if kernels is None else kernels

    if force is not None:
        by_name = {kernel.name: kernel for kernel in candidates}
        if force not in by_name:
            raise ValueError(
                f"distance_kernel={force!r} is not a known kernel; "
                f"must be one of {sorted(by_name)}"
            )
        kernel = by_name[force]
        if not kernel.applicable(device):
            raise ValueError(
                f"distance_kernel={force!r} requested but is not "
                f"applicable for device={device}"
            )
        return kernel

    for kernel in candidates:
        if kernel.applicable(device):
            return kernel

    raise RuntimeError(
        f"no distance kernel is applicable for device={device}; the "
        "`kernels` list passed to `select_kernel` must end in an "
        "unconditionally applicable fallback."
    )


def split_lattice(
    lattice: Tensor | None,
) -> tuple[Tensor | None, Tensor | None]:
    """
    Sort a structure's lattice into the two forms that
    :func:`pair_distance_squared` takes.

    A single cell, however it is shaped -- ``(3, 3)`` or ``(1, 3, 3)`` --
    is shared by every system of a batch, so it is returned as a
    ``(3, 3)`` tensor and never gathered per pair. Only a lattice with
    several cells is one cell per system.

    Parameters
    ----------
    lattice : Tensor | None
        ``None`` for a molecule, else the cell(s), shape ``(3, 3)`` or
        ``(..., 3, 3)``.

    Returns
    -------
    tuple[Tensor | None, Tensor | None]
        ``(shared_lattice, system_lattices)``: ``(3, 3)`` or ``None``, and
        ``(n_systems, 3, 3)`` or ``None``. At most one is set.
    """
    if lattice is None:
        return None, None

    cells = lattice.reshape(-1, 3, 3)
    if cells.shape[0] == 1:
        return cells[0], None
    return None, cells


def _image_translation(shift: Tensor, cell: Tensor) -> Tensor:
    """
    The Cartesian translation of each integer lattice shift.

    Parameters
    ----------
    shift : Tensor
        ``(n, 3)``, integer lattice translations.
    cell : Tensor
        One ``(3, 3)`` cell for every row, or one ``(n, 3, 3)`` cell per
        row, lattice vectors as rows.

    Returns
    -------
    Tensor
        ``(n, 3)``, in the dtype of ``cell``.
    """
    displacement = shift.to(cell.dtype)
    if cell.ndim == 2:
        return displacement @ cell
    return (displacement.unsqueeze(-2) @ cell).squeeze(-2)


# Whether the backward pass of `index_select` and `index_add` (and the
# batching rules of `vmap`, `jacrev` and `jacfwd`) accept an `int32` index.
# Before PyTorch 2.8 they thread it through `gather`, which raises
# `gather(): Expected dtype int64 for index`. A plain Python constant, read
# when a consumer runs, so it never becomes part of a traced graph.
_INT32_INDEX_BACKWARD = __tversion__ >= (2, 8, 0)


# `torch.compile(torch.func.jacrev(f))` returns wrong gradients (every output
# row is the sum of the rows, not the row of its own input) for an `f` that
# calls `Tensor.index_select`, on PyTorch 2.5 to 2.13 (2.14 is correct). Plain
# `torch.compile`, eager `jacrev` and `jacfwd` are fine, and so is `table[idx]`.
# Read when a consumer runs, like `_INT32_INDEX_BACKWARD`.
_COMPILED_INDEX_SELECT_JACREV_BUG = __tversion__ < (2, 14, 0)


def gather_rows(table: Tensor, index: Tensor) -> Tensor:
    """
    ``table.index_select(0, index)``, except while ``torch.compile`` traces
    on PyTorch before 2.14, where it is ``table[index]``.

    Compiled ``jacrev`` of an ``index_select`` is wrong there (the cause is
    in PyTorch, not here), and ``table[index]`` is not. ``index_select`` is
    kept everywhere else: on CUDA the backward pass of advanced indexing
    sorts all indices, and on CPU its gather is several times slower.

    Parameters
    ----------
    table : Tensor
        The values to gather from, indexed along dimension 0.
    index : Tensor
        One-dimensional indices, as :func:`gather_index` returns them.

    Returns
    -------
    Tensor
        ``(len(index), *table.shape[1:])``.
    """
    if _COMPILED_INDEX_SELECT_JACREV_BUG and is_compiling():
        return table[index]
    return table.index_select(0, index)


def gather_index(index: Tensor) -> Tensor:
    """
    A slice of a neighbour list's (``int32``) atom indices, in a dtype
    that every index operation and its backward pass accept.

    On PyTorch 2.8 and later this is ``index`` itself: an ``int32`` slice
    of a list is a view, so autograd keeps it for the backward pass
    without a copy. Before 2.8 it is ``index.long()``, a new ``int64``
    tensor. ``.long()`` is always correct; this only avoids the copy where
    PyTorch allows it. Call it on a chunk, never on a whole list.

    Parameters
    ----------
    index : Tensor
        Atom indices, ``int32`` or ``int64``.

    Returns
    -------
    Tensor
        ``index`` itself, or an ``int64`` copy on PyTorch before 2.8.
    """
    if _INT32_INDEX_BACKWARD:
        return index
    return index.long()


def position_columns(positions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """
    The ``x``, ``y`` and ``z`` columns of ``(..., n, 3)`` positions, each
    a contiguous ``(..., n)`` tensor, for
    :func:`pair_distance_squared_from_columns`.

    Split once per evaluation, not per chunk: the backward pass of the
    split then runs once, however many chunks gather from the columns.

    Parameters
    ----------
    positions : Tensor
        Cartesian coordinates, ``(..., n, 3)``.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        The three columns, ``(..., n)`` each.
    """
    x, y, z = (column.contiguous() for column in positions.unbind(-1))
    return x, y, z


def pair_distance_squared(
    idx_i: Tensor,
    idx_j: Tensor,
    shift: Tensor,
    positions: Tensor,
    *,
    shared_lattice: Tensor | None,
    system_lattices: Tensor | None,
    atoms_per_system: int,
) -> Tensor:
    """
    Squared distance from atom ``idx_i`` to the image of atom ``idx_j``
    at ``shift``, for each pair of a :class:`.list.NeighborList` (or a
    chunk of one).

    Unlike the tile-pair kernels above, this operates on pairs already
    picked out by a built neighbour list, not on a tile search; it is a
    consumer-side helper shared by :mod:`.ncoord.common` and
    :mod:`.io.checks.structure`, not by :mod:`.list` itself. It is public
    (``tad_mctc.neighbor.pair_distance_squared``) for the same reason, for
    other consumers of a neighbour list.

    It splits ``positions`` into columns on every call. A consumer that
    walks a list in chunks splits them once with :func:`position_columns`
    and calls :func:`pair_distance_squared_from_columns` per chunk.

    Parameters
    ----------
    idx_i, idx_j : Tensor
        ``(n_pairs,)``, atom indices into ``positions``, ``int32`` (as a
        list stores them) or ``int64``.
    shift : Tensor
        ``(n_pairs, 3)``, integer lattice translation of atom ``idx_j``.
    positions : Tensor
        ``(total_atoms, 3)``, Cartesian coordinates of the flattened
        batch, numbered like the list (``b * atoms_per_system + i``).
    shared_lattice : Tensor | None
        One ``(3, 3)`` cell used by every pair, or ``None``.
    system_lattices : Tensor | None
        One cell per system, ``(n_systems, 3, 3)``, or ``None``. At most
        one of the two lattices is set; neither is for a molecule.
    atoms_per_system : int
        Atoms per system, so that atom ``k`` belongs to system
        ``k // atoms_per_system``.

    Returns
    -------
    Tensor
        ``(n_pairs,)``, squared pair distances.
    """
    return pair_distance_squared_from_columns(
        idx_i,
        idx_j,
        shift,
        position_columns(positions),
        shared_lattice=shared_lattice,
        system_lattices=system_lattices,
        atoms_per_system=atoms_per_system,
    )


def pair_distance_squared_from_columns(
    idx_i: Tensor,
    idx_j: Tensor,
    shift: Tensor,
    columns: tuple[Tensor, Tensor, Tensor],
    *,
    shared_lattice: Tensor | None,
    system_lattices: Tensor | None,
    atoms_per_system: int,
) -> Tensor:
    """
    :func:`pair_distance_squared` from positions already split into
    columns by :func:`position_columns`.

    Each coordinate is gathered from its own ``(total_atoms,)`` column,
    not as rows of a ``(total_atoms, 3)`` table. On CPU, PyTorch runs the
    backward pass of a gather from a table with rows through
    ``index_add`` for an ``int32`` index, about 4x slower than the
    ``scatter_add`` it uses for an ``int64`` one (122 vs 32 ns per pair,
    PyTorch 2.14); the backward pass of a gather from a single column is
    equally fast for both, and faster than either.

    Gathers use :func:`gather_rows`: ``index_select``, not ``column[idx]``
    (on CUDA the backward pass of advanced indexing sorts all indices, and
    on CPU its gather is several times slower), except under
    ``torch.compile`` before PyTorch 2.14, which miscompiles the ``jacrev``
    of an ``index_select``.

    Parameters
    ----------
    idx_i, idx_j, shift
        As for :func:`pair_distance_squared`.
    columns : tuple[Tensor, Tensor, Tensor]
        ``x``, ``y`` and ``z`` of every atom, ``(total_atoms,)`` each.
    shared_lattice, system_lattices, atoms_per_system
        As for :func:`pair_distance_squared`.

    Returns
    -------
    Tensor
        ``(n_pairs,)``, squared pair distances.
    """
    # A no-op on PyTorch 2.8 and later; before, it widens the chunk (or the
    # real entries) it is handed, never a whole stored list.
    idx_i = gather_index(idx_i)
    idx_j = gather_index(idx_j)
    x, y, z = (
        gather_rows(column, idx_j) - gather_rows(column, idx_i)
        for column in columns
    )

    # The translation of each pair's lattice shift, one column at a time:
    # `sum_m shift[m] * cell[m, k]` for k = x, y, z. A `(n, 3) @ (3, 3)`
    # product (per system, a batch of `(1, 3) @ (3, 3)`) writes an `(n, 3)`
    # temporary that is then read through stride-3 slices; this is 12% (one
    # cell) to 28% (a cell per system) faster for 131k float64 pairs on CPU,
    # with the same bits.
    if shared_lattice is not None:
        # One `(3, 3)` cell for every pair: nine scalars.
        def entry(m: int, k: int) -> Tensor:
            return shared_lattice[m, k]

    elif system_lattices is not None:
        # Each pair takes the cell of its own system: nine `(n,)` columns.
        # `int64`, because `gather_rows` needs it before PyTorch 2.8.
        system = (idx_i // atoms_per_system).long()

        def entry(m: int, k: int) -> Tensor:
            return gather_rows(system_lattices[:, m, k].contiguous(), system)

    else:
        return x * x + y * y + z * z  # a molecule: nothing to translate

    s0, s1, s2 = (step.to(x.dtype) for step in shift.unbind(-1))
    x, y, z = (
        c + (s0 * entry(0, k) + s1 * entry(1, k) + s2 * entry(2, k))
        for k, c in enumerate((x, y, z))
    )

    # Written out over the three components: `(difference**2).sum(-1)` is a
    # reduction over an axis of length 3, which PyTorch evaluates ~10x slower
    # than the same sum of three columns (0.54 ms vs 5.8 ms for 131k float64
    # pairs on CPU), and dominated the sparse coordination number.
    return x * x + y * y + z * z
