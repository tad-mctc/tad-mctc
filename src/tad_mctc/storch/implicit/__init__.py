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
SafeOps: Implicit Differentiation
=================================

Derivatives of solutions of nonlinear equations by the implicit function
theorem, independent of the solver that found them, to any order and in
reverse, forward and mixed mode:

- :func:`implicit_root`: root of ``F(z, p) = 0``
- :func:`implicit_fixed_point`: fixed point ``x = G(x, p)``
- :func:`implicit_argmin`: minimizer of ``f(z, p)``

The response equations are linear solves with
:func:`~tad_mctc.storch.linsolve.implicit_solve`.

Example
-------
>>> import torch
>>> from torch.func import jacfwd
>>> from tad_mctc.storch.implicit import implicit_fixed_point
>>> g = lambda x, a: torch.cos(a * x)  # x = cos(a x)
>>> def solve(a):
...     x = torch.tensor(0.7, dtype=a.dtype)
...     for _ in range(200):  # any solver, not differentiated
...         x = torch.cos(a.detach() * x)
...     return implicit_fixed_point(g, x, a)
>>> a = torch.tensor(1.0, dtype=torch.double)
>>> round(jacfwd(jacfwd(solve))(a).item(), 6)
0.153338
"""

from . import root
from .root import *

__all__: list[str] = []
__all__ += root.__all__
