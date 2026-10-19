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
Implicit Differentiation: Roots, Fixed Points and Minima
========================================================

Derivatives of the solution :math:`z^*(p)` of a nonlinear equation by the
implicit function theorem, independent of the solver that found it.

For a root of :math:`F(z, p) = 0` with invertible Jacobian :math:`F_z`,

- forward mode: :math:`F_z \\dot z = -F_p \\dot p`,
- reverse mode: :math:`F_z^T \\lambda = \\bar z`, :math:`\\bar p =
  -F_p^T \\lambda`.

Both linear systems are solved matrix-free with
:func:`~tad_mctc.storch.linsolve.implicit_solve`, whose operator depends on
:math:`z^*` and :math:`p`. Nested derivatives are therefore derivatives of
these linear systems (another linear solve), never of the iterations of
the nonlinear or the linear solver, so the derivatives exist to any order
in forward, reverse and mixed mode and under ``vmap``.

The Krylov iterations reuse one recorded pullback of :math:`F` (the
tangent operator as the pullback of the pullback), so they do not
re-evaluate :math:`F`.

A derivative system that does not converge gives NaN (fail-closed), never
a silently wrong derivative.

Only real floating point tensors are supported.

Under ``torch.compile``, the forward pass and first derivatives through
``Tensor.backward`` / ``torch.autograd.grad`` work, if Dynamo may break the
graph. Everything else raises: Dynamo cannot trace ``torch.func``
transforms (``jacrev``) of these functions inside compiled code, since the
response solve applies an ``autograd.Function`` inside another one's
backward (PyTorch 2.7 to 2.14), and forward-mode transforms inside compiled
code are rejected (see :func:`~tad_mctc.tools.reject_forward_mode`).

The formulation follows ``jaxopt.implicit_diff`` (``custom_root``,
``custom_fixed_point``) and ``torchopt.diff.implicit``
(``custom_root``), Apache-2.0.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import torch
from torch._C._functorch import (  # pylint: disable=no-name-in-module
    peek_interpreter_stack,
)

from ...autograd.nested import nested_jvp
from ...tools.compile import is_compiling, reject_forward_mode
from ...typing import Tensor
from ..linsolve.implicit import Method, implicit_solve
from ..linsolve.result import STATUS_CONVERGED, LinearSolveResult, require_real

__all__ = ["implicit_root", "implicit_fixed_point", "implicit_argmin"]


Residual = Callable[..., Tensor]
"""Residual ``F(z, *params)`` with the shape of ``z``."""


def _is_differentiable(value: Any) -> bool:
    return isinstance(value, Tensor) and (
        value.is_floating_point() or value.is_complex()
    )


def _with_values(
    params: tuple[Any, ...], index: list[int], values: tuple[Tensor, ...]
) -> tuple[Any, ...]:
    full = list(params)
    for i, value in zip(index, values):
        full[i] = value
    return tuple(full)


def _jvp(
    fn: Callable[..., Tensor],
    primals: tuple[Tensor, ...],
    tangents: tuple[Tensor, ...],
) -> Tensor:
    """JVP of ``fn``, also inside ``torch.autograd.forward_ad``."""
    if peek_interpreter_stack() is not None:
        return torch.func.jvp(fn, primals, tangents)[1]

    # `torch.func.jvp` would be nested forward-mode AD. The pullback is
    # linear in the cotangent; the pullback of the pullback is the JVP.
    recorded = torch.func.vjp(fn, *primals)
    out, pullback = recorded[0], recorded[1]
    double = torch.func.vjp(pullback, torch.zeros_like(out))[1]
    return double(tangents)[0]


@dataclass(frozen=True)
class _Root:
    """Static description of the root problem and its response solves."""

    residual: Residual
    symmetric: bool
    method: Method | Callable[..., LinearSolveResult]
    preconditioner: Callable[..., Tensor] | None
    transpose_preconditioner: Callable[..., Tensor] | None
    options: tuple[tuple[str, Any], ...]

    def batched(self, dims: tuple[Any, ...]) -> _Root:
        """The problem for a leading batch dimension of the root."""

        def vmapped(fn: Callable[..., Tensor] | None, *extra: int) -> Any:
            if fn is None:
                return None
            return torch.vmap(fn, in_dims=(*extra, 0, *dims))

        # residual(z, *p); preconditioners (r, z, *p)
        return replace(
            self,
            residual=vmapped(self.residual),
            preconditioner=vmapped(self.preconditioner, 0),
            transpose_preconditioner=vmapped(self.transpose_preconditioner, 0),
        )

    # Operators of the response equations, as functions of the vector `v`,
    # the root `z` and the parameters. `implicit_solve` differentiates them
    # with respect to `z` and the parameters for nested derivatives.

    def _f(self, params: tuple[Any, ...]) -> Callable[[Tensor], Tensor]:
        return lambda z: self.residual(z, *params)

    def tangent(self, v: Tensor, z: Tensor, *params: Any) -> Tensor:
        """``F_z v``."""
        return _jvp(self._f(params), (z,), (v,))

    def adjoint(self, v: Tensor, z: Tensor, *params: Any) -> Tensor:
        """``F_z^T v``."""
        return torch.func.vjp(self._f(params), z)[1](v)[0]

    def adjoint_linearization(
        self, z: Tensor, *params: Any
    ) -> Callable[[Tensor], Tensor]:
        pullback = torch.func.vjp(self._f(params), z)[1]
        return lambda v: pullback(v)[0]

    def tangent_linearization(
        self, z: Tensor, *params: Any
    ) -> Callable[[Tensor], Tensor]:
        # The pullback is linear in the cotangent, so its own pullback is the
        # JVP. Both are recorded once; no product re-evaluates `F`.
        recorded = torch.func.vjp(self._f(params), z)
        out, pullback = recorded[0], recorded[1]
        jvp = torch.func.vjp(lambda c: pullback(c)[0], torch.zeros_like(out))[1]
        return lambda v: jvp(v)[0]

    def solve(
        self, rhs: Tensor, z: Tensor, params: tuple[Any, ...], adjoint: bool
    ) -> Tensor:
        """
        Solve ``F_z x = rhs`` (or ``F_z^T x = rhs``); NaN for systems that
        did not converge.
        """
        if self.symmetric:
            ops = (self.adjoint, self.adjoint)
            lins = (self.adjoint_linearization, self.adjoint_linearization)
        else:
            ops = (self.tangent, self.adjoint)
            lins = (self.tangent_linearization, self.adjoint_linearization)

        precs = (self.preconditioner, self.transpose_preconditioner)
        if adjoint:
            ops, lins, precs = ops[::-1], lins[::-1], precs[::-1]

        options = dict(self.options)
        result = implicit_solve(
            ops[0],
            rhs,
            z,
            *params,
            method=self.method,
            rmatvec=ops[1],
            symmetric=self.symmetric,
            preconditioner=precs[0],
            transpose_preconditioner=precs[1],
            linearize=lins[0],
            rlinearize=lins[1],
            **options,
        )
        converged = result.status == STATUS_CONVERGED
        for _ in range(options["vdims"]):
            converged = converged.unsqueeze(-1)
        return torch.where(
            converged,
            result.solution,
            torch.full_like(result.solution, torch.nan),
        )

    def attach(self, root: Tensor, params: tuple[Any, ...]) -> Tensor:
        if not is_compiling():
            return _ImplicitRoot.apply(root, self, *params)

        reject_forward_mode("implicit_root", root, *params)
        # Dynamo drops derivatives of graph inputs that reach a Function
        # unused by its `forward`; a view avoids it (see `implicit_solve`).
        params = tuple(
            p.view_as(p) if isinstance(p, Tensor) else p for p in params
        )
        return _ImplicitRootReverse.apply(root, self, *params)


def _unpack(ctx: Any) -> tuple[Tensor, tuple[Any, ...]]:
    root, *tensors = ctx.saved_tensors
    params = list(ctx.static_params)
    for i, tensor in zip(ctx.tensor_index, tensors):
        params[i] = tensor
    return root, tuple(params)


class _ImplicitRootReverse(torch.autograd.Function):
    """
    Attach implicit reverse-mode derivatives to a (detached) root. The
    forward is the identity on the root.
    """

    @staticmethod
    def forward(root: Tensor, problem: _Root, *params: Any) -> Tensor:
        return root.clone()

    @staticmethod
    def setup_context(
        ctx: Any, inputs: tuple[Any, ...], output: Tensor
    ) -> None:
        _, problem, *params = inputs
        ctx.problem = problem
        ctx.tensor_index = [
            i for i, p in enumerate(params) if isinstance(p, Tensor)
        ]
        ctx.static_params = [
            None if isinstance(p, Tensor) else p for p in params
        ]
        tensors = [params[i] for i in ctx.tensor_index]

        # The output (not the detached input) is saved, so that higher
        # derivatives see the dependence of the root on the parameters.
        ctx.save_for_backward(output, *tensors)
        ctx.save_for_forward(output, *tensors)

    @staticmethod
    def backward(  # pyright: ignore[reportIncompatibleMethodOverride]
        ctx: Any, grad: Tensor
    ) -> tuple[Any, ...]:
        root, params = _unpack(ctx)
        problem: _Root = ctx.problem

        index = [
            i
            for i, p in enumerate(params)
            if _is_differentiable(p) and ctx.needs_input_grad[2 + i]
        ]
        grads: list[Tensor | None] = [None] * len(params)
        if len(index) == 0:
            return (None, None, *grads)

        # F_z^T lambda = grad, grad_p = -F_p^T lambda
        adjoint = problem.solve(grad, root, params, adjoint=True)
        pullback = torch.func.vjp(
            lambda *v: problem.residual(root, *_with_values(params, index, v)),
            *(params[i] for i in index),
        )[1]
        for i, g in zip(index, pullback(adjoint)):
            grads[i] = -g
        return (None, None, *grads)

    @staticmethod
    def vmap(
        info: Any,
        in_dims: tuple[Any, ...],
        root: Tensor,
        problem: _Root,
        *params: Any,
    ) -> tuple[Tensor, int]:
        # The `vmap` dimension becomes a leading batch dimension of the root,
        # and the residual is `vmap`-ped over it.
        if in_dims[0] is None:
            root = root.expand(info.batch_size, *root.shape)
        else:
            root = root.movedim(in_dims[0], 0)

        dims = tuple(in_dims[2:])
        params = tuple(
            p.movedim(d, 0) if d is not None else p
            for p, d in zip(params, dims)
        )
        dims = tuple(None if d is None else 0 for d in dims)
        return problem.batched(dims).attach(root, params), 0


class _ImplicitRoot(_ImplicitRootReverse):
    """Attach implicit reverse- and forward-mode derivatives to a root."""

    @staticmethod
    @nested_jvp
    def jvp(  # pyright: ignore[reportIncompatibleMethodOverride]
        ctx: Any,
        root_dot: Tensor | None,
        problem_dot: None,
        *params_dot: Tensor | None,
    ) -> Tensor:
        # The tangent of the detached root is meaningless; the root is
        # defined by the equation alone.
        del root_dot, problem_dot

        root, params = _unpack(ctx)
        problem: _Root = ctx.problem

        index = [
            i
            for i, (p, t) in enumerate(zip(params, params_dot))
            if _is_differentiable(p) and t is not None
        ]
        if len(index) == 0:
            return torch.zeros_like(root)

        # F_z z_dot = -F_p p_dot
        rhs = _jvp(
            lambda *v: problem.residual(root, *_with_values(params, index, v)),
            tuple(params[i] for i in index),
            tuple(params_dot[i] for i in index),  # type: ignore[misc]
        )
        return problem.solve(-rhs, root, params, adjoint=False)


def implicit_root(
    residual: Residual,
    root: Tensor,
    *params: Any,
    method: Method | Callable[..., LinearSolveResult] = "gmres",
    symmetric: bool = False,
    preconditioner: Callable[..., Tensor] | None = None,
    transpose_preconditioner: Callable[..., Tensor] | None = None,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    maxiter: int | None = None,
    vdims: int | None = None,
) -> Tensor:
    r"""
    Attach implicit derivatives to a root of :math:`F(z, p) = 0`.

    ``root`` is a solution found by any (non-differentiable) solver. The
    returned tensor has the value of ``root`` and the derivatives of the
    exact solution :math:`z^*(p)` with respect to every floating point
    tensor in ``params``, in reverse and forward mode, to any order and
    under ``vmap`` (see module docstring). The derivatives are exact if
    ``root`` is an accurate root and :math:`F_z` is invertible there. The
    caller is responsible for the convergence of ``root``: the derivatives
    are evaluated at the given point, so an inaccurate root gives finite,
    inaccurate derivatives. Validate the residual before calling this.

    Parameters
    ----------
    residual : Callable[..., Tensor]
        Residual ``F(z, *params)``, returning a tensor of the shape of
        ``z``. It must be differentiable with ``torch.func``.
    root : Tensor
        The root :math:`z^*`. It is detached.
    *params : Any
        Arguments of ``residual``. Floating point tensors receive
        derivatives; other arguments are passed through.
    method : {"gmres", "bicgstab", "cg"} | Callable, optional
        Krylov method for the response equations with :math:`F_z` and
        :math:`F_z^T` (see :func:`~tad_mctc.storch.linsolve.implicit_solve`).
        Defaults to ``"gmres"``; ``"cg"`` requires a symmetric positive
        definite :math:`F_z`.
    symmetric : bool, optional
        Whether :math:`F_z` is symmetric (e.g. a Hessian), which saves the
        tangent operator. Defaults to ``False``.
    preconditioner : Callable[..., Tensor] | None, optional
        Preconditioner of :math:`F_z`, ``preconditioner(r, z, *params)``.
    transpose_preconditioner : Callable[..., Tensor] | None, optional
        Preconditioner of :math:`F_z^T`. Defaults to ``preconditioner``.
    rtol, atol : float, optional
        Tolerances of the response equations. Defaults to ``1e-10`` and
        ``1e-12``.
    maxiter : int | None, optional
        Iteration limit of the response equations. Defaults to twice the
        vector size.
    vdims : int | None, optional
        Number of trailing dimensions of ``root`` forming one system. The
        leading dimensions are independent systems, which requires that
        :math:`F` does not couple them. Defaults to all dimensions (one
        system).

    Returns
    -------
    Tensor
        ``root`` with implicit derivatives. A derivative whose response
        equation did not converge is NaN.

    Raises
    ------
    DtypeError
        ``root`` is not real floating point.

    Example
    -------
    >>> import torch
    >>> from tad_mctc.storch.implicit import implicit_root
    >>> def residual(z, p):  # z**3 = p
    ...     return z**3 - p
    >>> p = torch.tensor(8.0, dtype=torch.double, requires_grad=True)
    >>> z = implicit_root(residual, p.detach() ** (1 / 3), p)
    >>> (dz,) = torch.autograd.grad(z, p)
    >>> print(z.item(), round(dz.item(), 6))  # 1 / (3 z**2)
    2.0 0.083333
    """
    require_real("root", root)
    if vdims is None:
        vdims = max(root.ndim, 1)
    squeeze = root.ndim == 0

    if squeeze:
        # a scalar root is a vector of one element
        scalar_residual = residual

        def residual(z: Tensor, *p: Any) -> Tensor:
            return scalar_residual(z.squeeze(-1), *p).unsqueeze(-1)

        root = root.unsqueeze(-1)

    problem = _Root(
        residual=residual,
        symmetric=symmetric,
        method=method,
        preconditioner=preconditioner,
        transpose_preconditioner=transpose_preconditioner,
        options=(
            ("rtol", rtol),
            ("atol", atol),
            ("maxiter", maxiter),
            ("vdims", vdims),
        ),
    )
    out = problem.attach(root.detach(), params)
    return out.squeeze(-1) if squeeze else out


def implicit_fixed_point(
    fixed_point_map: Residual,
    fixed_point: Tensor,
    *params: Any,
    method: Method | Callable[..., LinearSolveResult] = "gmres",
    preconditioner: Callable[..., Tensor] | None = None,
    transpose_preconditioner: Callable[..., Tensor] | None = None,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    maxiter: int | None = None,
    vdims: int | None = None,
) -> Tensor:
    r"""
    Attach implicit derivatives to a fixed point :math:`x = G(x, p)`.

    The root of :math:`F(x, p) = x - G(x, p)` (see :func:`implicit_root`):
    the response equations are :math:`(I - G_x)\,\dot x = G_p \dot p` and
    :math:`(I - G_x)^T \lambda = \bar x`, :math:`\bar p = G_p^T \lambda`.
    They require :math:`I - G_x` to be invertible, not :math:`G` to be a
    contraction. This is the formulation of deep equilibrium models and of
    self-consistent field iterations.

    Parameters
    ----------
    fixed_point_map : Callable[..., Tensor]
        Map ``G(x, *params)`` with the shape of ``x``.
    fixed_point : Tensor
        The fixed point :math:`x^*`, found by any solver. It is detached.
    *params : Any
        Arguments of ``fixed_point_map``.
    method, preconditioner, transpose_preconditioner, rtol, atol, maxiter, vdims
        See :func:`implicit_root`. Preconditioners approximate
        :math:`(I - G_x)^{-1}`.

    Returns
    -------
    Tensor
        ``fixed_point`` with implicit derivatives (NaN where a response
        equation did not converge).

    Example
    -------
    >>> import torch
    >>> from tad_mctc.storch.implicit import implicit_fixed_point
    >>> def g(x, a, b):  # x = a x + b, x* = b / (1 - a)
    ...     return a * x + b
    >>> a = torch.tensor(0.5, dtype=torch.double, requires_grad=True)
    >>> b = torch.tensor(1.0, dtype=torch.double)
    >>> x = implicit_fixed_point(g, torch.tensor(2.0).double(), a, b)
    >>> (da,) = torch.autograd.grad(x, a)
    >>> print(x.item(), da.item())  # b / (1 - a)**2
    2.0 4.0
    """

    def residual(x: Tensor, *p: Any) -> Tensor:
        return x - fixed_point_map(x, *p)

    return implicit_root(
        residual,
        fixed_point,
        *params,
        method=method,
        preconditioner=preconditioner,
        transpose_preconditioner=transpose_preconditioner,
        rtol=rtol,
        atol=atol,
        maxiter=maxiter,
        vdims=vdims,
    )


def implicit_argmin(
    objective: Callable[..., Tensor],
    minimum: Tensor,
    *params: Any,
    method: Method | Callable[..., LinearSolveResult] = "cg",
    preconditioner: Callable[..., Tensor] | None = None,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    maxiter: int | None = None,
    vdims: int | None = None,
) -> Tensor:
    r"""
    Attach implicit derivatives to a minimizer of :math:`f(z, p)`.

    The root of the optimality condition :math:`\nabla_z f(z, p) = 0` (see
    :func:`implicit_root`). Its Jacobian is the Hessian
    :math:`\nabla^2_{zz} f`, which is symmetric, and positive definite at a
    strict minimum, so CG is the default. For saddle points or maxima, use
    ``method="gmres"``.

    Parameters
    ----------
    objective : Callable[..., Tensor]
        Objective ``f(z, *params)``. A scalar, or one value per independent
        system (leading dimensions of ``z`` when ``vdims`` is given); the
        values are summed before taking the gradient.
    minimum : Tensor
        The minimizer :math:`z^*`, found by any optimizer. It is detached.
    *params : Any
        Arguments of ``objective``.
    method, preconditioner, rtol, atol, maxiter, vdims
        See :func:`implicit_root`. The preconditioner approximates the
        inverse Hessian.

    Returns
    -------
    Tensor
        ``minimum`` with implicit derivatives (NaN where a response
        equation did not converge).

    Example
    -------
    >>> import torch
    >>> from tad_mctc.storch.implicit import implicit_argmin
    >>> def f(z, p):  # ridge: (z - p)^2 + 0.5 z^2, z* = 2 p / 3
    ...     return ((z - p) ** 2 + 0.5 * z**2).sum()
    >>> p = torch.tensor([3.0, 6.0], dtype=torch.double, requires_grad=True)
    >>> z = implicit_argmin(f, 2 * p.detach() / 3, p)
    >>> (dp,) = torch.autograd.grad(z.sum(), p)
    >>> print(z.tolist(), [round(v, 4) for v in dp.tolist()])
    [2.0, 4.0] [0.6667, 0.6667]
    """

    def residual(z: Tensor, *p: Any) -> Tensor:
        return torch.func.grad(lambda z: objective(z, *p).sum())(z)

    return implicit_root(
        residual,
        minimum,
        *params,
        method=method,
        symmetric=True,
        preconditioner=preconditioner,
        transpose_preconditioner=preconditioner,
        rtol=rtol,
        atol=atol,
        maxiter=maxiter,
        vdims=vdims,
    )
