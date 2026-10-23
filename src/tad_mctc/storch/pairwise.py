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
SafeOps: Pairwise Kernel Products
=================================

Matrix-free products with a matrix of pairwise interactions,

.. math::

    y_i = \sum_{j \neq i} k(r_{ij}, p_i, p_j) \, x_j,

for a symmetric kernel :math:`k` of the distance :math:`r_{ij}` and
per-atom parameters :math:`p`, without storing the ``n x n`` matrix: the
pairs are evaluated in tiles of rows, so memory is ``O(n * tile)``. This is
the operator of a charge model or a Coulomb interaction for an iterative
solver (:mod:`tad_mctc.storch.linsolve`); the diagonal is not included.

:func:`pair_product` is a ``torch.autograd.Function`` whose backward and
forward-mode rules recompute each tile instead of storing it, so the first
derivatives with respect to the positions, the parameters and the vector
also take ``O(n * tile)`` memory, and it composes with ``torch.func``
transforms and with the implicit derivatives of
:func:`~tad_mctc.storch.linsolve.implicit_solve`. Higher derivatives are
exact as well, but keep the graphs of the tiles (``O(n^2)`` memory). That
includes the mixed derivatives of a parameter fit to forces
(:math:`\partial^2 E / \partial R \, \partial \theta`) and Hessians: these
are bounded by the memory of the dense matrix, not by the tiles.
:func:`pair_cell_sums` gives the sums of the matrix columns over cells,
which the balancing preconditioner needs
(:func:`~tad_mctc.storch.linsolve.balancing`).

The work is ``O(n^2)``. In eager mode, each tile takes about ten
elementwise passes over its ``rows x n`` pair terms. ``fused=True``
compiles the product of a tile with ``torch.compile`` into one reduction,
which never stores the pair terms: on an RTX 4070, a float32 product takes
2.7 ms for 20k atoms and 61 ms for 100k, 35x faster than eager. The
derivatives are evaluated eagerly; so are the products under ``torch.func``
transforms (e.g. the derivative solves of
:func:`~tad_mctc.storch.linsolve.implicit_solve` under ``jacrev``), while a
plain ``.backward()`` keeps them fused. Everything evaluated eagerly uses
the eager tiles, so the memory bound holds on every path. ``torch.compile``
of code that calls the Function is not supported.

The products may be evaluated in a lower precision than the vector
(``dtype``), e.g. float32 on GPUs with few float64 units, where a float64
product of 100k atoms takes 3.7 s. Their relative error is then about
``1e-6``, so a float32 operator must not be used alone where float64
accuracy is needed. Pass it as the ``fast_matvec`` of
:func:`~tad_mctc.storch.linsolve.implicit_solve` or
:func:`~tad_mctc.storch.linsolve.constrained_solve` instead, with the
float64 product as the operator: every solve, including the derivative
solves, is then refined to the float64 tolerance, and float32 only speeds
up the iterations. For a charge model of 10k atoms at ``rtol=1e-12``, the
charges agree with a float64 Cholesky solve to ``4e-14`` and the forces to
``5e-15``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from ..autograd.checks import is_functorch_tensor
from ..autograd.nested import nested_jvp
from ..tools.compile import (
    get_compile_backend,
    is_compile_supported,
    is_compiling,
)
from ..typing import Tensor

__all__ = ["PairKernel", "pair_cell_sums", "pair_product"]


PairKernel = Callable[[Tensor, tuple[Tensor, ...], tuple[Tensor, ...]], Tensor]
"""
``kernel(r, params_i, params_j) -> values``: the pair terms for distances
``r`` of shape ``(..., rows, n)``, with the per-atom parameters of the rows
(``(..., rows, 1)`` each) and of the columns (``(..., 1, n)`` each). It must
be symmetric, ``kernel(r, p_i, p_j) == kernel(r, p_j, p_i)``, and smooth
for ``r > 0``; excluded pairs are evaluated at ``r = 1`` and dropped.
The positions of distinct atoms must differ: at ``r = 0``, the derivative
of the distance is infinite, so a coincident pair gives NaN (in the values
for a kernel such as ``erf(a r) / r``, in the derivatives for any kernel).
Kernels with a finite limit at ``r = 0`` belong to the model, which can
handle such pairs explicitly.
"""

# Pair entries per tile, as for the coordination number (see
# `tad_mctc.ncoord.common`): small enough to stay in cache on CPU, large
# enough to fill a GPU.
_TILE_PAIRS_CPU = 131_072
_TILE_PAIRS_GPU = 1_048_576

# A fused tile does not store its pair terms, so it can be much larger. Its
# rows are fixed and the number of atoms is a dynamic dimension, so that a
# kernel compiles once for all system sizes. On an RTX 4070 (float32), 2048
# rows are as fast as 4096 from 1k to 100k atoms, and 1.8x faster than 1024
# at 10k; tiles of 64 rows are 2.6x slower at 100k.
_TILE_ROWS_FUSED = 2048


def _eager_rows(n: int, device: torch.device) -> int:
    """Rows of an eager tile, which stores its pair terms."""
    if device.type == "cpu":
        pairs = _TILE_PAIRS_CPU
    else:  # pragma: no cover - GPU only
        pairs = _TILE_PAIRS_GPU
    return max(1, pairs // max(n, 1))


def _tile_rows(
    positions: Tensor, tile: int | None, fused: bool
) -> tuple[int, int]:
    """
    Rows of the compiled and of the eager tiles. With ``fused``, a given
    ``tile`` sets the compiled tiles; the eager ones (derivatives and
    fallbacks) never exceed the eager default, as they store their terms.
    """
    if tile is not None and tile < 1:
        raise ValueError(f"The tile size ({tile}) must be positive.")
    n = positions.shape[-2]
    eager = _eager_rows(n, positions.device)
    if not fused:
        return (eager, eager) if tile is None else (tile, tile)
    if tile is None:
        tile = _TILE_ROWS_FUSED
    return tile, min(tile, eager)


class _Spec:
    """Static arguments of the Function (not tensors, not differentiated)."""

    def __init__(
        self,
        kernel: PairKernel,
        tile: int,
        dtype: torch.dtype | None,
        fused: bool = False,
        eager_tile: int | None = None,
    ) -> None:
        self.kernel = kernel
        self.tile = tile
        self.dtype = dtype
        self.fused = fused
        # Eager tiles (derivatives, fallbacks of `fused`) store their terms.
        self.eager_tile = tile if eager_tile is None else eager_tile


def _block(
    kernel: PairKernel,
    dtype: torch.dtype | None,
    rows: Tensor,
    positions_i: Tensor,
    positions_j: Tensor,
    params_i: tuple[Tensor, ...],
    params_j: tuple[Tensor, ...],
    mask_i: Tensor | None,
    mask_j: Tensor | None,
) -> Tensor:
    """
    Pair terms of the atoms ``rows`` (coordinates ``positions_i``) with all
    atoms, shape ``(..., rows, n)``.
    """
    if dtype is not None:
        positions_i, positions_j = positions_i.to(dtype), positions_j.to(dtype)
        params_i = tuple(
            p.to(dtype) if p.is_floating_point() else p for p in params_i
        )
        params_j = tuple(
            p.to(dtype) if p.is_floating_point() else p for p in params_j
        )

    diff = positions_i.unsqueeze(-2) - positions_j.unsqueeze(-3)
    columns = torch.arange(positions_j.shape[-2], device=rows.device)
    valid = rows.unsqueeze(-1) != columns
    if mask_i is not None and mask_j is not None:
        valid = valid & mask_i.unsqueeze(-1) & mask_j.unsqueeze(-2)

    # Excluded pairs at r = 1, so that neither the kernel nor its
    # derivatives see a zero distance.
    r = torch.sqrt(torch.where(valid, (diff * diff).sum(-1), 1.0))
    values = kernel(
        r,
        tuple(p.unsqueeze(-1) for p in params_i),
        tuple(p.unsqueeze(-2) for p in params_j),
    )
    return torch.where(valid, values, 0.0)


def _block_product(
    kernel: PairKernel,
    dtype: torch.dtype | None,
    rows: Tensor,
    positions_i: Tensor,
    positions_j: Tensor,
    params_i: tuple[Tensor, ...],
    params_j: tuple[Tensor, ...],
    mask_i: Tensor | None,
    mask_j: Tensor | None,
    x: Tensor,
) -> Tensor:
    values = _block(
        kernel,
        dtype,
        rows,
        positions_i,
        positions_j,
        params_i,
        params_j,
        mask_i,
        mask_j,
    )
    if dtype is None:
        return torch.einsum("...ij,...j->...i", values, x)
    product = torch.einsum("...ij,...j->...i", values, x.to(dtype))
    return product.to(x.dtype)


def _block_product_reduction(
    kernel: PairKernel,
    dtype: torch.dtype | None,
    rows: Tensor,
    positions_i: Tensor,
    positions_j: Tensor,
    params_i: tuple[Tensor, ...],
    params_j: tuple[Tensor, ...],
    mask_i: Tensor | None,
    mask_j: Tensor | None,
    x: Tensor,
) -> Tensor:  # pragma: no cover - Dynamo-traced
    """
    :func:`_block_product` as a sum of an elementwise product, which the
    compiler fuses with the pair terms into one reduction (a contraction
    is lowered to a matrix-vector call on the stored terms).

    Beyond its recompilation limit, Dynamo runs this function eagerly, on
    a tile sized for the compiled reduction: it is then split into eager
    tiles, which store their terms.
    """
    if not is_compiling():
        return _split_block_product(
            kernel,
            dtype,
            rows,
            positions_i,
            positions_j,
            params_i,
            params_j,
            mask_i,
            mask_j,
            x,
        )
    values = _block(
        kernel,
        dtype,
        rows,
        positions_i,
        positions_j,
        params_i,
        params_j,
        mask_i,
        mask_j,
    )
    vector = x if dtype is None else x.to(dtype)
    return (values * vector.unsqueeze(-2)).sum(-1).to(x.dtype)


def _split_block_product(
    kernel: PairKernel,
    dtype: torch.dtype | None,
    rows: Tensor,
    positions_i: Tensor,
    positions_j: Tensor,
    params_i: tuple[Tensor, ...],
    params_j: tuple[Tensor, ...],
    mask_i: Tensor | None,
    mask_j: Tensor | None,
    x: Tensor,
) -> Tensor:
    """:func:`_block_product` in eager tiles of the rows."""
    size = _eager_rows(positions_j.shape[-2], positions_j.device)
    tiles = [
        _block_product(
            kernel,
            dtype,
            rows[start:stop],
            positions_i[..., start:stop, :],
            positions_j,
            tuple(p[..., start:stop] for p in params_i),
            params_j,
            None if mask_i is None else mask_i[..., start:stop],
            mask_j,
            x,
        )
        for start, stop in _ranges(rows.shape[-1], size)
    ]
    return torch.cat(tiles, dim=-1).to(x.dtype)


_FUSED: list[Callable[..., Tensor]] = []
"""The compiled :func:`_block_product_reduction`, created on first use."""


def _fused_block_product() -> Callable[..., Tensor]:
    if not _FUSED:
        _FUSED.append(
            torch.compile(
                _block_product_reduction,
                dynamic=False,
                backend=get_compile_backend(),
            )
        )
    return _FUSED[0]


def _tile_arguments(
    positions: Tensor,
    params: tuple[Tensor, ...],
    mask: Tensor | None,
    start: int,
    stop: int,
) -> tuple[Any, ...]:
    rows = torch.arange(start, stop, device=positions.device)
    return (
        rows,
        positions[..., start:stop, :],
        positions,
        tuple(p[..., start:stop] for p in params),
        params,
        None if mask is None else mask[..., start:stop],
        mask,
    )


def _values(
    spec: _Spec,
    positions: Tensor,
    params: tuple[Tensor, ...],
    mask: Tensor | None,
    start: int,
    stop: int,
) -> Tensor:
    """Pair terms of the rows ``start:stop``, shape ``(..., rows, n)``."""
    args = _tile_arguments(positions, params, mask, start, stop)
    return _block(spec.kernel, spec.dtype, *args)


def _tile_product(
    spec: _Spec,
    positions: Tensor,
    x: Tensor,
    params: tuple[Tensor, ...],
    mask: Tensor | None,
    start: int,
    stop: int,
) -> Tensor:
    args = _tile_arguments(positions, params, mask, start, stop)
    product: Callable[..., Tensor] = _block_product
    return product(spec.kernel, spec.dtype, *args, x)


def _product(
    spec: _Spec,
    positions: Tensor,
    x: Tensor,
    params: tuple[Tensor, ...],
    mask: Tensor | None,
) -> Tensor:
    tensors = (positions, x, *params) + (() if mask is None else (mask,))
    fused = (
        spec.fused
        and is_compile_supported()
        and not is_compiling()
        and not any(is_functorch_tensor(t) for t in tensors)
        and positions.shape[-2] > 0
    )
    if fused:
        # The forward needs no graph. Dynamo guards on `requires_grad` and
        # the strides (e.g. of an expanded gradient in the backward), so
        # each pattern of them would compile anew.
        positions = positions.detach().contiguous()
        x = x.detach().contiguous()
        params = tuple(p.detach().contiguous() for p in params)
        mask = None if mask is None else mask.contiguous()
    if spec.dtype is not None:
        # Once, not per tile: in a fused tile, the cast of the coordinates
        # is not hoisted and doubles its time.
        positions = positions.to(spec.dtype)
        params = tuple(
            p.to(spec.dtype) if p.is_floating_point() else p for p in params
        )
    if fused:
        return _fused_product(spec, positions, x, params, mask)
    tiles = [
        _tile_product(spec, positions, x, params, mask, start, stop)
        for start, stop in _ranges(positions.shape[-2], spec.eager_tile)
    ]
    return torch.cat(tiles, dim=-1)


def _fused_product(
    spec: _Spec,
    positions: Tensor,
    x: Tensor,
    params: tuple[Tensor, ...],
    mask: Tensor | None,
) -> Tensor:
    """
    The compiled tiles, all of one shape: the last tile is padded with
    excluded rows, and the number of atoms is a dynamic dimension, so that
    all system sizes compile once (Dynamo recompiles for every new shape,
    up to its limit).
    """
    n, tile = positions.shape[-2], spec.tile
    if mask is None:
        mask = torch.ones(
            positions.shape[:-1], dtype=torch.bool, device=positions.device
        )
    product = _fused_block_product()
    dtype = x.dtype
    if spec.dtype is not None:
        x = x.to(spec.dtype)  # once, as the coordinates
    for t in (positions, x, mask, *params):
        dim = t.ndim - (2 if t is positions else 1)
        torch._dynamo.maybe_mark_dynamic(t, dim)
    tiles = []
    for start, stop in _ranges(n, tile):
        rows = torch.arange(start, start + tile, device=positions.device)
        real = rows < n
        index = torch.where(real, rows, n - 1)
        out = product(
            spec.kernel,
            spec.dtype,
            rows,
            positions[..., index, :],
            positions,
            tuple(p[..., index] for p in params),
            params,
            mask[..., index] & real,
            mask,
            x,
        )
        tiles.append(out[..., : stop - start])
    return torch.cat(tiles, dim=-1).to(dtype)


def _ranges(n: int, tile: int) -> list[tuple[int, int]]:
    return [(s, min(s + tile, n)) for s in range(0, max(n, 1), tile)]


def _differentiable(params: tuple[Any, ...]) -> list[int]:
    return [
        i
        for i, p in enumerate(params)
        if isinstance(p, Tensor) and p.is_floating_point()
    ]


def _sum_to(tensor: Tensor, shape: torch.Size) -> Tensor:
    """Sum a broadcast tensor back to ``shape``."""
    lead = tensor.ndim - len(shape)
    if lead > 0:
        tensor = tensor.sum(dim=tuple(range(lead)))
    dims = tuple(
        i for i, s in enumerate(shape) if s == 1 and tensor.shape[i] != 1
    )
    return tensor.sum(dim=dims, keepdim=True) if dims else tensor


class _PairProduct(torch.autograd.Function):
    """
    ``y = K x`` for the pair matrix ``K`` of a symmetric kernel, with
    derivatives that recompute the tiles.
    """

    @staticmethod
    def forward(
        positions: Tensor,
        x: Tensor,
        mask: Tensor | None,
        spec: _Spec,
        *params: Tensor,
    ) -> Tensor:
        return _product(spec, positions, x, params, mask)

    @staticmethod
    def setup_context(
        ctx: Any, inputs: tuple[Any, ...], output: Tensor
    ) -> None:
        positions, x, mask, spec, *params = inputs
        ctx.spec = spec
        ctx.has_mask = mask is not None
        tensors = (positions, x, *params) + (
            (mask,) if mask is not None else ()
        )
        ctx.save_for_backward(*tensors)
        ctx.save_for_forward(*tensors)

    @staticmethod
    def backward(  # pyright: ignore[reportIncompatibleMethodOverride]
        ctx: Any, grad: Tensor
    ) -> tuple[Any, ...]:
        positions, x, params, mask = _unpack(ctx)
        spec: _Spec = ctx.spec
        needs = ctx.needs_input_grad

        # K is symmetric: the gradient of the vector is another product.
        grad_x = None
        if needs[1]:
            grad_x = _sum_to(
                _PairProduct.apply(positions, grad, mask, spec, *params),
                x.shape,
            )

        index = [i for i in _differentiable(params) if needs[4 + i]]
        grad_positions = None
        grads: list[Tensor | None] = [None] * len(params)
        if needs[0] or len(index) > 0:
            totals: list[Tensor] = []
            for start, stop in _ranges(positions.shape[-2], spec.eager_tile):

                def tile(pos: Tensor, *values: Tensor) -> Tensor:
                    full = list(params)
                    for i, v in zip(index, values):
                        full[i] = v
                    return _tile_product(
                        spec, pos, x, tuple(full), mask, start, stop
                    )

                pullback = torch.func.vjp(
                    tile, positions, *(params[i] for i in index)
                )[1]
                parts = pullback(grad[..., start:stop])
                totals = (
                    list(parts)
                    if not totals
                    else [t + p for t, p in zip(totals, parts)]
                )
            grad_positions = totals[0] if needs[0] else None
            for i, g in zip(index, totals[1:]):
                grads[i] = g

        return (grad_positions, grad_x, None, None, *grads)

    @staticmethod
    @nested_jvp
    def jvp(  # pyright: ignore[reportIncompatibleMethodOverride]
        ctx: Any,
        positions_dot: Tensor | None,
        x_dot: Tensor | None,
        mask_dot: None,
        spec_dot: None,
        *params_dot: Tensor | None,
    ) -> Tensor:
        del mask_dot, spec_dot
        positions, x, params, mask = _unpack(ctx)
        spec: _Spec = ctx.spec

        # PyTorch passes zeros for a missing tangent; None is defensive.
        tangent = None
        if x_dot is not None:  # pragma: no branch
            tangent = _PairProduct.apply(positions, x_dot, mask, spec, *params)

        index = [
            i for i in _differentiable(params) if params_dot[i] is not None
        ]
        if positions_dot is not None or len(index) > 0:  # pragma: no branch
            pos_dot = (
                torch.zeros_like(positions)
                if positions_dot is None
                else positions_dot
            )
            tiles = []
            for start, stop in _ranges(positions.shape[-2], spec.eager_tile):

                def tile(pos: Tensor, *values: Tensor) -> Tensor:
                    full = list(params)
                    for i, v in zip(index, values):
                        full[i] = v
                    return _tile_product(
                        spec, pos, x, tuple(full), mask, start, stop
                    )

                tiles.append(
                    torch.func.jvp(
                        tile,
                        (positions, *(params[i] for i in index)),
                        (pos_dot, *(params_dot[i] for i in index)),
                    )[1]
                )
            part = torch.cat(tiles, dim=-1)
            tangent = part if tangent is None else tangent + part
        if tangent is None:  # pragma: no cover - see above
            tangent = torch.zeros_like(
                _PairProduct.apply(positions, x, mask, spec, *params)
            )
        return tangent

    @staticmethod
    def vmap(  # pyright: ignore[reportIncompatibleMethodOverride]
        info: Any,
        in_dims: tuple[Any, ...],
        positions: Tensor,
        x: Tensor,
        mask: Tensor | None,
        spec: _Spec,
        *params: Tensor,
    ) -> tuple[Tensor, int]:
        # The product broadcasts over leading dimensions: the `vmap`
        # dimension becomes the first of them, after padding the batch
        # ranks of the batched inputs to a common rank.
        inputs = [positions, x, mask, *params]
        dims = [in_dims[0], in_dims[1], in_dims[2], *in_dims[4:]]
        vector = [2, 1, 1] + [1] * len(params)  # trailing non-batch dims

        def rank(t: Tensor | None, d: int | None, v: int) -> int:
            if t is None:
                return 0
            return t.ndim - v - (0 if d is None else 1)

        common = max(rank(t, d, v) for t, d, v in zip(inputs, dims, vector))
        moved: list[Any] = []
        for t, d, v in zip(inputs, dims, vector):
            if t is None or d is None:
                moved.append(t)
                continue
            t = t.movedim(d, 0)
            for _ in range(common - rank(t, 0, v)):
                t = t.unsqueeze(1)
            moved.append(t)
        return (
            _PairProduct.apply(moved[0], moved[1], moved[2], spec, *moved[3:]),
            0,
        )


def _unpack(
    ctx: Any,
) -> tuple[Tensor, Tensor, tuple[Tensor, ...], Tensor | None]:
    saved = ctx.saved_tensors
    if ctx.has_mask:
        *saved, mask = saved
    else:
        mask = None
    positions, x, *params = saved
    return positions, x, tuple(params), mask


def pair_product(
    kernel: PairKernel,
    positions: Tensor,
    x: Tensor,
    *params: Tensor,
    mask: Tensor | None = None,
    tile: int | None = None,
    dtype: torch.dtype | None = None,
    fused: bool = False,
) -> Tensor:
    r"""
    Product :math:`y_i = \sum_{j \neq i} k(r_{ij}, p_i, p_j) x_j` with the
    matrix of a symmetric pair kernel, without storing the matrix.

    Parameters
    ----------
    kernel : PairKernel
        Symmetric kernel ``kernel(r, params_i, params_j)``, see
        :data:`PairKernel`.
    positions : Tensor
        Cartesian coordinates, shape ``(..., n, 3)``.
    x : Tensor
        Vector, shape ``(..., n)``; its leading dimensions broadcast with
        those of ``positions`` (e.g. several vectors for one structure).
    *params : Tensor
        Per-atom parameters of the kernel, shape ``(..., n)`` each.
    mask : Tensor | None, optional
        Atoms that interact, shape ``(..., n)``; pairs with another atom
        (e.g. padding) are zero. Defaults to all.
    tile : int | None, optional
        Rows per tile. Defaults to about a million pairs per tile on GPU
        and 131k on CPU, or 2048 rows with ``fused`` (whose tiles do not
        store their terms). With ``fused``, it sets the compiled tiles
        only: whatever is evaluated eagerly (the derivatives and the
        fallbacks below) uses tiles of at most the eager default, so that
        the memory stays bounded.
    dtype : torch.dtype | None, optional
        Dtype in which the pair terms and the products are evaluated; the
        result has the dtype of ``x``. Defaults to that of ``positions``.
        See the module documentation for the tolerance of a solve with a
        float32 product.
    fused : bool, optional
        Compile the products of the tiles with ``torch.compile`` (see the
        module documentation). Pass a module-level ``kernel``: a new
        function object (e.g. a lambda created per call) triggers a new
        compilation, and so does every new dtype, tile size and batch
        shape, but not a new number of atoms (a dynamic dimension; the
        last tile is padded). Beyond Dynamo's recompilation limit (8 by
        default), and where compiling is not
        possible (under ``torch.func`` transforms, inside
        ``torch.compile``, or without Dynamo support), the products are
        evaluated eagerly, in eager tiles. Defaults to ``False``.

    Returns
    -------
    Tensor
        The product, shape ``(..., n)`` (broadcast batch shape).

    Example
    -------
    >>> import torch
    >>> from tad_mctc.storch.pairwise import pair_product
    >>> positions = torch.tensor([[0.0, 0, 0], [1.0, 0, 0], [3.0, 0, 0]])
    >>> coulomb = lambda r, pi, pj: 1.0 / r
    >>> pair_product(coulomb, positions, torch.ones(3))
    tensor([1.3333, 1.5000, 0.8333])
    """
    if positions.ndim < 2 or positions.shape[-1] != 3:
        raise ValueError(
            f"The positions must have shape (..., n, 3), not "
            f"{tuple(positions.shape)}."
        )
    n = positions.shape[-2]
    for name, t in (("vector", x), *(("parameter", p) for p in params)):
        if t.shape[-1] != n:
            raise ValueError(
                f"The {name} must have the size n = {n} of the positions "
                f"in its last dimension, not {tuple(t.shape)}."
            )
    rows, eager_rows = _tile_rows(positions, tile, fused)
    spec = _Spec(kernel, rows, dtype, fused, eager_tile=eager_rows)
    return _PairProduct.apply(positions, x, mask, spec, *params)


def pair_cell_sums(
    kernel: PairKernel,
    positions: Tensor,
    cells: Tensor,
    size: int,
    *params: Tensor,
    mask: Tensor | None = None,
    tile: int | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    r"""
    Sums of the columns of the pair matrix over cells,
    :math:`(K Z)_{ic} = \sum_{j \in c} k(r_{ij}, p_i, p_j)`, at the cost of
    one :func:`pair_product`.

    This is the setup of the balancing preconditioner
    (:func:`~tad_mctc.storch.linsolve.balancing`), which needs the cell sums
    of the full matrix: add the diagonal ``d_i`` of the operator to entry
    ``(i, cells[i])``. The result is not differentiable.

    Parameters
    ----------
    kernel : PairKernel
        Symmetric kernel, see :func:`pair_product`.
    positions : Tensor
        Cartesian coordinates, shape ``(..., n, 3)``.
    cells : Tensor
        Cell of each atom in ``[0, size)``, or ``-1`` for none, shape
        ``(..., n)`` (e.g. from :func:`~tad_mctc.storch.linsolve.cell_index`).
    size : int
        Number of cells.
    *params : Tensor
        Per-atom parameters of the kernel, shape ``(..., n)`` each.
    mask : Tensor | None, optional
        Atoms that interact, see :func:`pair_product`.
    tile : int | None, optional
        Rows per tile, see :func:`pair_product`.
    dtype : torch.dtype | None, optional
        Dtype of the pair terms and of the result. Defaults to the dtype of
        ``positions``.

    Returns
    -------
    Tensor
        Cell sums, shape ``(..., n, size)``.
    """
    spec = _Spec(kernel, _tile_rows(positions, tile, False)[0], dtype)
    n = positions.shape[-2]
    index = torch.where(cells < 0, size, cells).long()

    with torch.no_grad():
        tiles = []
        for start, stop in _ranges(n, spec.tile):
            values = _values(spec, positions, params, mask, start, stop)
            columns = index.unsqueeze(-2).expand_as(values)
            out = values.new_zeros(*values.shape[:-1], size + 1)
            tiles.append(out.scatter_add(-1, columns, values)[..., :size])
        return torch.cat(tiles, dim=-2)
