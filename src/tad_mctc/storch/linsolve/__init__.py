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
SafeOps: Linear Solve
=====================

Iterative, matrix-free solvers for linear systems ``A x = b`` and their
implicit differentiation.

- :func:`cg`: preconditioned conjugate gradient (symmetric positive definite)
- :func:`bicgstab`: preconditioned BiCGSTAB (general nonsymmetric)
- :func:`gmres`: restarted, preconditioned GMRES (nonsymmetric, indefinite)
- :func:`implicit_solve`: solve with derivatives defined by the linear system
  (reverse and forward mode, any order), independent of the iterations
  (optionally refined with a cheaper, e.g. float32, operator to the
  accuracy of the exact one)
- :func:`dense_solve`: LU solve with the same implicit derivatives, a
  replacement for ``torch.linalg.solve``, whose derivatives over its
  forward-mode rule and nested derivatives under ``vmap`` are wrong in
  PyTorch 2.7 to 2.14
- :func:`constrained_solve`: minimizer of a quadratic form under linear
  equality constraints (projected CG on the bordered system), with the same
  implicit derivatives
- :func:`jacobi`, :func:`balancing`: preconditioners; the balancing (BNN)
  preconditioner on a partition into cells (:func:`cell_index`) keeps the
  iteration count of Coulomb-like operators nearly independent of the size

All solvers are batched, report a structured :class:`LinearSolveResult`
with tensor-valued status codes, and work under ``torch.func`` transforms.

Example
-------
>>> import torch
>>> from tad_mctc.storch.linsolve import implicit_solve
>>> def matvec(x, a):
...     return torch.einsum("...ij,...j->...i", a, x)
>>> a = torch.tensor([[4.0, 1.0], [1.0, 3.0]], dtype=torch.double)
>>> b = torch.tensor([1.0, 2.0], dtype=torch.double)
>>> result = implicit_solve(matvec, b, a, method="cg")
>>> [round(v, 4) for v in result.solution.tolist()], result.converged.item()
([0.0909, 0.6364], True)
"""

from . import constrained, dense, implicit, krylov, preconditioners, result
from .constrained import *
from .dense import *
from .implicit import *
from .krylov import *
from .preconditioners import *
from .result import *

# Explicit, so that the star import of this package does not also export
# its submodules (`implicit` would shadow `tad_mctc.storch.implicit`).
__all__: list[str] = []
__all__ += constrained.__all__
__all__ += dense.__all__
__all__ += implicit.__all__
__all__ += krylov.__all__
__all__ += preconditioners.__all__
__all__ += result.__all__
