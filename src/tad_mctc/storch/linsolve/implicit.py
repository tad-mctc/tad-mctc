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
Linear Solve: Implicit Differentiation
======================================

Derivatives of the solution of :math:`A(p)\,x = b` that are defined by the
linear system, not by the iterations of the solver.

For a tangent :math:`(\dot b, \dot p)`, the solution changes by

.. math::

    A \dot x = \dot b - \dot A x, \qquad \dot A x = \partial_p (A(p) x) \dot p,

and for a cotangent :math:`\bar x`, the adjoint :math:`\lambda` solves

.. math::

    A^T \lambda = \bar x, \qquad \bar b = \lambda, \qquad
    \bar p = -\partial_p (A(p) x)^T \lambda.

Both derivative solves are again implicit linear solves, so higher-order
derivatives (``jacrev(jacrev(f))``, ``jacfwd(jacfwd(f))``, ...) are
derivatives of linear systems as well and never differentiate a Krylov
iteration. This mirrors ``jax.lax.custom_linear_solve`` as used by
`JAXopt <https://github.com/google/jaxopt>`__ (Apache-2.0).

The primal solve runs on detached inputs and only selects the value of the
solution. Consequently, the derivatives do not depend on the method, the
initial guess, the preconditioner or the iteration count, as long as every
solve converges. A derivative solve that does not converge returns NaN
(fail-closed), since its status cannot be raised inside ``torch.func``
transforms.

Implementation notes
--------------------
The derivatives are attached by a ``torch.autograd.Function`` whose forward
is the identity on the detached solution (as dxtb's implicit SCF root).

- ``vmap``: a batching rule moves the ``vmap`` dimension to the front, where
  it becomes a batch dimension of the solves, and ``vmap``-s the operator.
  The derivative rules then run on ordinary batched tensors.
- ``jvp``: PyTorch hides a custom ``jvp`` from enclosing forward-mode
  transforms, see :func:`~tad_mctc.autograd.nested_jvp`, which fixes it.
  With the batching rule above, every nesting of ``jacfwd`` is correct.
- ``torch.compile``: only first derivatives in reverse mode are supported:

  - ``torch.compile(f)`` with ``Tensor.backward`` / ``torch.autograd.grad``;
  - ``torch.compile(jacrev(f))`` (PyTorch >= 2.11; earlier versions raise).

  Forward mode inside compiled code (``torch.compile(jacfwd(f))``) raises:
  Dynamo would ignore the ``jvp`` rule and return wrong tangents, see
  :func:`~tad_mctc.tools.reject_forward_mode`. Transforms of a compiled
  function (``jacrev(torch.compile(f))``, ``jacfwd(...)``) and double
  backward raise in Dynamo or AOTAutograd. **Higher derivatives inside
  compiled code are not supported** and cannot be rejected reliably: a
  nested derivative of a Function misses the dependence of its
  ``backward`` on saved intermediate tensors, so from PyTorch 2.11 on,
  ``torch.compile(jacrev(jacrev(f)))`` is silently wrong (earlier versions
  raise). Take higher derivatives in eager mode.
  Inputs of the compiled function are passed through a view, without which
  ``jacrev`` inside compiled code returns zero gradients for them.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, Literal

import torch
from torch._C._functorch import (  # pylint: disable=no-name-in-module
    peek_interpreter_stack,
)

from ...autograd.nested import nested_jvp
from ...tools.compile import is_compiling, reject_forward_mode
from ...typing import Tensor
from .krylov import Operator, bicgstab, cg, gmres
from .result import STATUS_CONVERGED, LinearSolveResult, require_real

__all__ = ["implicit_solve"]


Method = Literal["cg", "bicgstab", "gmres"]
"""Iterative method used by :func:`implicit_solve`."""

Linearization = Callable[..., Callable[[Tensor], Tensor]]
"""
``linearize(*params) -> (x -> A x)``: the operator for fixed arguments,
which may reuse work between products (e.g. a cached pullback).
"""

_METHODS: dict[str, Callable[..., LinearSolveResult]] = {
    "cg": cg,
    "bicgstab": bicgstab,
    "gmres": gmres,
}


def _transpose(matvec: Operator) -> Operator:
    """
    Transposed operator ``v -> A^T v`` of the linear operator ``matvec``,
    as the pullback of ``matvec`` with respect to its first argument.
    """

    def rmatvec(v: Tensor, *params: Any) -> Tensor:
        # `matvec` is linear in `x`, so the pullback at any point is A^T.
        pullback = torch.func.vjp(
            lambda x: matvec(x, *params), torch.zeros_like(v)
        )[1]
        return pullback(v)[0]

    return rmatvec


def _vmapped(fn: Operator, dims: tuple[Any, ...]) -> Operator:
    """Operator acting on a leading ``vmap`` dimension of its arguments."""
    return torch.vmap(fn, in_dims=(0, *dims))


@dataclass(frozen=True)
class _LinearSystem:
    """Static (non-tensor) description of a linear system and its solver."""

    matvec: Operator
    rmatvec: Operator
    method: Callable[..., LinearSolveResult]
    transpose_method: Callable[..., LinearSolveResult]
    preconditioner: Operator | None
    transpose_preconditioner: Operator | None
    linearize: Linearization | None
    rlinearize: Linearization | None
    options: tuple[tuple[str, Any], ...]

    def transposed(self) -> _LinearSystem:
        """The system with the transposed operator."""
        return replace(
            self,
            matvec=self.rmatvec,
            rmatvec=self.matvec,
            method=self.transpose_method,
            transpose_method=self.method,
            preconditioner=self.transpose_preconditioner,
            transpose_preconditioner=self.preconditioner,
            linearize=self.rlinearize,
            rlinearize=self.linearize,
        )

    def batched(self, dims: tuple[Any, ...]) -> _LinearSystem:
        """
        The system for a leading batch dimension of the vectors and the
        given batch dimensions (or ``None``) of the operator arguments.
        """
        return replace(
            self,
            matvec=_vmapped(self.matvec, dims),
            rmatvec=_vmapped(self.rmatvec, dims),
            preconditioner=(
                None
                if self.preconditioner is None
                else _vmapped(self.preconditioner, dims)
            ),
            transpose_preconditioner=(
                None
                if self.transpose_preconditioner is None
                else _vmapped(self.transpose_preconditioner, dims)
            ),
            # The `vmap`-ped operator is evaluated directly.
            linearize=None,
            rlinearize=None,
        )

    def solve(
        self, rhs: Tensor, params: tuple[Any, ...], x0: Tensor | None = None
    ) -> LinearSolveResult:
        """Solve on detached inputs and attach the implicit derivatives."""
        with torch.no_grad():
            detached = tuple(_detach(p) for p in params)

            matvec = self.matvec
            if self.linearize is not None:
                operator = self.linearize(*detached)
                matvec = lambda x, *_: operator(x)  # noqa: E731

            result = self.method(
                matvec,
                rhs.detach(),
                *detached,
                x0=None if x0 is None else x0.detach(),
                preconditioner=self.preconditioner,
                **dict(self.options),
            )

        # Not `result._replace`, which Dynamo (PyTorch 2.7) does not trace
        return LinearSolveResult(
            solution=self.attach(result.solution, rhs, params),
            residual=result.residual,
            target=result.target,
            iterations=result.iterations,
            status=result.status,
        )

    def attach(
        self, solution: Tensor, rhs: Tensor, params: tuple[Any, ...]
    ) -> Tensor:
        """Attach the implicit derivatives to a (detached) solution."""
        if not is_compiling():
            return _ImplicitLinearSolve.apply(solution, rhs, self, *params)

        # Under `torch.compile`, `jacrev` returns zero gradients for an input
        # of the compiled function that reaches the Function directly and is
        # unused by its `forward` (PyTorch 2.7 to 2.14). A view avoids it.
        reject_forward_mode("implicit_solve", rhs, *params)
        rhs = rhs.view_as(rhs)
        params = tuple(
            p.view_as(p) if isinstance(p, Tensor) else p for p in params
        )
        return _ImplicitLinearSolveReverse.apply(solution, rhs, self, *params)

    def derivative(self, rhs: Tensor, params: tuple[Any, ...]) -> Tensor:
        """
        Solution of a derivative system, NaN for systems that did not
        converge.
        """
        result = self.solve(rhs, params)
        # Not the `converged` property, which Dynamo (PyTorch 2.7) does not
        # trace
        converged = result.status == STATUS_CONVERGED
        for _ in range(dict(self.options)["vdims"]):
            converged = converged.unsqueeze(-1)
        return torch.where(
            converged,
            result.solution,
            torch.full_like(result.solution, torch.nan),
        )


def _detach(value: Any) -> Any:
    return value.detach() if isinstance(value, Tensor) else value


def _unpack(ctx: Any) -> tuple[Tensor, tuple[Any, ...]]:
    """Saved solution and operator arguments of the Function context."""
    solution, *tensors = ctx.saved_tensors
    params = list(ctx.static_params)
    for i, tensor in zip(ctx.tensor_index, tensors):
        params[i] = tensor
    return solution, tuple(params)


def _is_differentiable(value: Any) -> bool:
    return isinstance(value, Tensor) and (
        value.is_floating_point() or value.is_complex()
    )


def _with_values(
    params: tuple[Any, ...], index: list[int], values: tuple[Any, ...]
) -> tuple[Any, ...]:
    """The operator arguments with the entries at `index` replaced."""
    full = list(params)
    for i, value in zip(index, values):
        full[i] = value
    return tuple(full)


class _ImplicitLinearSolveReverse(torch.autograd.Function):
    """
    Attach implicit reverse-mode derivatives to the (detached) solution of
    a linear system. The forward is the identity on the solution.
    """

    @staticmethod
    def forward(
        solution: Tensor, rhs: Tensor, system: _LinearSystem, *params: Any
    ) -> Tensor:
        return solution.clone()

    @staticmethod
    def setup_context(
        ctx: Any, inputs: tuple[Any, ...], output: Tensor
    ) -> None:
        _, _, system, *params = inputs
        ctx.system = system

        # Tensors must be saved through `save_for_*`; other arguments are
        # static and kept on the context.
        ctx.tensor_index = [
            i for i, p in enumerate(params) if isinstance(p, Tensor)
        ]
        ctx.static_params = [
            None if isinstance(p, Tensor) else p for p in params
        ]
        tensors = [params[i] for i in ctx.tensor_index]

        # The output (not the detached input) is saved, so that higher
        # derivatives see the dependence of the solution on the inputs.
        ctx.save_for_backward(output, *tensors)
        ctx.save_for_forward(output, *tensors)

    @staticmethod
    def backward(  # pyright: ignore[reportIncompatibleMethodOverride]
        ctx: Any, grad: Tensor
    ) -> tuple[Any, ...]:
        solution, params = _unpack(ctx)
        system: _LinearSystem = ctx.system

        # A^T lambda = grad
        adjoint = system.transposed().derivative(grad, params)

        # grad_p = -(d(A x)/dp)^T lambda
        index = [
            i
            for i, p in enumerate(params)
            if _is_differentiable(p) and ctx.needs_input_grad[3 + i]
        ]
        grads: list[Tensor | None] = [None] * len(params)
        if len(index) > 0:
            pullback = torch.func.vjp(
                lambda *v: system.matvec(
                    solution, *_with_values(params, index, v)
                ),
                *(params[i] for i in index),
            )[1]
            for i, g in zip(index, pullback(adjoint)):
                grads[i] = -g

        return (None, adjoint, None, *grads)

    @staticmethod
    def vmap(
        info: Any,
        in_dims: tuple[Any, ...],
        solution: Tensor,
        rhs: Tensor,
        system: _LinearSystem,
        *params: Any,
    ) -> tuple[Tensor, int]:
        # The `vmap` dimension becomes a leading batch dimension of the
        # vectors, and the operator is `vmap`-ped over it.
        def front(value: Tensor, dim: int | None) -> Tensor:
            if dim is None:
                return value.expand(info.batch_size, *value.shape)
            return value.movedim(dim, 0)

        solution = front(solution, in_dims[0])
        rhs = front(rhs, in_dims[1])
        batched = system.batched(tuple(in_dims[3:]))
        return batched.attach(solution, rhs, params), 0


class _ImplicitLinearSolve(_ImplicitLinearSolveReverse):
    """
    Attach implicit reverse- and forward-mode derivatives to the (detached)
    solution of a linear system.
    """

    @staticmethod
    @nested_jvp
    def jvp(  # pyright: ignore[reportIncompatibleMethodOverride]
        ctx: Any,
        solution_dot: Tensor | None,
        rhs_dot: Tensor | None,
        system_dot: None,
        *params_dot: Tensor | None,
    ) -> Tensor:
        # The tangent of the detached primal solution is meaningless; the
        # solution is defined by the linear system alone.
        del solution_dot, system_dot

        solution, params = _unpack(ctx)
        return _tangent(ctx.system, solution, params, rhs_dot, params_dot)


def _tangent(
    system: _LinearSystem,
    solution: Tensor,
    params: tuple[Any, ...],
    rhs_dot: Tensor | None,
    params_dot: tuple[Tensor | None, ...],
) -> Tensor:
    """Tangent of the solution, ``A x_dot = b_dot - A_dot x``."""
    tangent = (
        torch.zeros_like(solution)
        if rhs_dot is None
        else rhs_dot.expand_as(solution)
    )

    index = [
        i
        for i, (p, t) in enumerate(zip(params, params_dot))
        if _is_differentiable(p) and t is not None
    ]
    if len(index) > 0:
        primals = tuple(params[i] for i in index)
        tangents = tuple(params_dot[i] for i in index)

        def apply(*values: Tensor) -> Tensor:
            return system.matvec(solution, *_with_values(params, index, values))

        if peek_interpreter_stack() is None:
            # Inside `torch.autograd.forward_ad`, `torch.func.jvp` would be
            # nested forward-mode AD. The pullback is linear in the
            # cotangent; the pullback of the pullback is the JVP.
            out, pullback = torch.func.vjp(apply, *primals)[:2]
            double = torch.func.vjp(pullback, torch.zeros_like(out))[1]
            adot_x = double(tangents)[0]
        else:
            adot_x = torch.func.jvp(apply, primals, tangents)[1]
        tangent = tangent - adot_x

    return system.derivative(tangent, params)


def implicit_solve(
    matvec: Operator,
    rhs: Tensor,
    *params: Any,
    method: Method | Callable[..., LinearSolveResult] = "cg",
    rmatvec: Operator | None = None,
    symmetric: bool | None = None,
    preconditioner: Operator | None = None,
    transpose_preconditioner: Operator | None = None,
    linearize: Linearization | None = None,
    rlinearize: Linearization | None = None,
    x0: Tensor | None = None,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    maxiter: int | None = None,
    vdims: int = 1,
    early_exit: bool | None = None,
) -> LinearSolveResult:
    r"""
    Solve :math:`A(p)\,x = b` with implicitly defined derivatives.

    The solution is differentiable with respect to ``rhs`` and every tensor
    in ``params`` in reverse and forward mode and to any order, under
    ``torch.func.vmap``, ``jacrev`` and ``jacfwd``. The derivatives are
    those of the exact solution (see module docstring); differentiating
    through the iterations is avoided.

    Parameters
    ----------
    matvec : Callable[..., Tensor]
        Linear operator ``matvec(x, *params) -> A x``. It must be a pure
        function of its arguments: tensors that require derivatives must be
        passed via ``params`` and not be captured in a closure.
    rhs : Tensor
        Right-hand side ``b``. The trailing ``vdims`` dimensions form the
        vector, leading dimensions are batch dimensions.
    *params : Any
        Arguments of the operator. Floating point tensors receive gradients.
    method : {"cg", "bicgstab", "gmres"} | Callable[..., LinearSolveResult], optional
        Iterative method, or a solver with the signature of
        :func:`~tad_mctc.storch.linsolve.cg`. Defaults to ``"cg"``, which
        requires a symmetric positive definite operator. Method-specific
        options are set with ``functools.partial``, e.g.
        ``method=partial(gmres, restart=40)``. A callable is also used for
        the transposed systems of the derivatives, so it must solve with the
        operator it is given (as the Krylov methods do).
    rmatvec : Callable[..., Tensor] | None, optional
        Transposed operator ``rmatvec(v, *params) -> A^T v``. Defaults to
        the operator itself for symmetric operators and to the pullback of
        ``matvec`` otherwise.
    symmetric : bool | None, optional
        Whether the operator is symmetric. Defaults to ``True`` for CG and
        ``False`` otherwise.
    preconditioner : Callable[..., Tensor] | None, optional
        Preconditioner ``preconditioner(r, *params) -> M r`` of the
        operator. It is evaluated on detached inputs and affects only the
        convergence of the solves, never the derivatives.
    transpose_preconditioner : Callable[..., Tensor] | None, optional
        Preconditioner of the transposed operator, used by the adjoint
        solves. Defaults to ``preconditioner``, which is correct for
        symmetric preconditioners such as Jacobi.
    linearize : Callable[..., Callable[[Tensor], Tensor]] | None, optional
        ``linearize(*params)`` returns the operator ``x -> A x`` for fixed
        arguments. If given, the iterations of all solves with :math:`A` use
        it instead of ``matvec``, so that work shared by all products is done
        once per solve, e.g. the forward pass of a pullback
        (``torch.func.vjp``). It is called on detached arguments; the
        derivatives always use ``matvec``. Must agree with ``matvec``.
    rlinearize : Callable[..., Callable[[Tensor], Tensor]] | None, optional
        As ``linearize``, for the transposed operator. Defaults to
        ``linearize`` for symmetric operators.
    x0 : Tensor | None, optional
        Initial guess of the primal solve. Defaults to zero.
    rtol : float, optional
        Relative tolerance of all solves. Defaults to ``1e-10``.
    atol : float, optional
        Absolute tolerance of all solves. Defaults to ``1e-12``.
    maxiter : int | None, optional
        Iteration limit of all solves. Defaults to twice the vector size.
    vdims : int, optional
        Number of trailing dimensions forming the vector. Defaults to ``1``.
    early_exit : bool | None, optional
        Loop exit policy, see :func:`~tad_mctc.storch.linsolve.cg`.

    Returns
    -------
    LinearSolveResult
        Differentiable solution and the diagnostics of the primal solve.
        The primal solution is returned as is, also if it did not converge,
        and its derivatives are then those at this inaccurate solution;
        check :attr:`~LinearSolveResult.status` or call
        :func:`~tad_mctc.storch.linsolve.require_convergence`. Derivative
        solves that do not converge yield NaN.

    Raises
    ------
    ValueError
        Unknown method.
    DtypeError
        ``rhs`` or ``x0`` is not real floating point.
    NotImplementedError
        Forward-mode derivatives inside ``torch.compile``.

    Example
    -------
    >>> import torch
    >>> from tad_mctc.storch.linsolve import implicit_solve
    >>> a = torch.tensor([[4.0, 1.0], [1.0, 3.0]], dtype=torch.double,
    ...                  requires_grad=True)
    >>> b = torch.tensor([1.0, 2.0], dtype=torch.double, requires_grad=True)
    >>> x = implicit_solve(lambda x, a: a @ x, b, a).solution
    >>> x.sum().backward()
    >>> torch.allclose(b.grad, torch.linalg.solve(a.T, torch.ones(2).double()))
    True
    """
    require_real("right-hand side", rhs)
    require_real("initial guess", x0)
    if isinstance(method, str):
        if method not in _METHODS:
            raise ValueError(
                f"Unknown method '{method}'. Choose from {list(_METHODS)}."
            )
        if symmetric is None:
            symmetric = method == "cg"
        method = _METHODS[method]
    elif symmetric is None:
        symmetric = False

    if rmatvec is None:
        rmatvec = matvec if symmetric else _transpose(matvec)
    if transpose_preconditioner is None:
        transpose_preconditioner = preconditioner
    if rlinearize is None and symmetric:
        rlinearize = linearize

    system = _LinearSystem(
        matvec=matvec,
        rmatvec=rmatvec,
        method=method,
        transpose_method=method,
        preconditioner=preconditioner,
        transpose_preconditioner=transpose_preconditioner,
        linearize=linearize,
        rlinearize=rlinearize,
        options=(
            ("rtol", rtol),
            ("atol", atol),
            ("maxiter", maxiter),
            ("vdims", vdims),
            ("early_exit", early_exit),
        ),
    )
    return system.solve(rhs, params, x0=x0)
