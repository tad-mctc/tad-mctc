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
Torch Autodiff Utilities
========================

This library is a collection of utility functions that are used in PyTorch (re-)
implementations of projects from the
`Grimme group <https://github.com/grimme-lab>`__.
In particular, the *tad-mctc* library provides:

- autograd functions (Jacobian, Hessian)

- atomic data (radii, EN, example molecules, ...)

- batch utility (packing, masks, ...)

- conversion functions (numpy, atomic symbols/numbers, ...)

- coordination numbers (DFT-D3, DFT-D4, EEQ)

- io (reading/writing coordinate files)

- molecular properties (bond lengths/orders/angles, moment of inertia, ...)

- safeops (autograd-safe implementations of common functions)

- typing (base class for tensor-like behavior of arbitrary classes)

- units

The name is inspired by the Fortran pendant "modular computation tool chain
library" (`mctc-lib <https://github.com/grimme-lab/mctc-lib/>`__).

.. note::

   This project is still in early development and the API is subject to change.
   Contributions are welcome, please checkout our
   `contributing guidelines <https://github.com/tad-mctc/tad-mctc/blob/main/CONTRIBUTING.md>`_.

Example
-------
>>> import torch
>>> import tad_mctc as mctc
>>>
>>> # S22 system 4: formamide dimer
>>> numbers = mctc.batch.pack((
...     mctc.convert.symbol_to_number("C C N N H H H H H H O O".split()),
...     mctc.convert.symbol_to_number("C O N H H H".split()),
... ))
>>>
>>> # coordinates in Bohr
>>> positions = mctc.batch.pack((
...     torch.tensor([
...         [-3.81469488143921, +0.09993441402912, 0.00000000000000],
...         [+3.81469488143921, -0.09993441402912, 0.00000000000000],
...         [-2.66030049324036, -2.15898251533508, 0.00000000000000],
...         [+2.66030049324036, +2.15898251533508, 0.00000000000000],
...         [-0.73178529739380, -2.28237795829773, 0.00000000000000],
...         [-5.89039325714111, -0.02589114569128, 0.00000000000000],
...         [-3.71254944801331, -3.73605775833130, 0.00000000000000],
...         [+3.71254944801331, +3.73605775833130, 0.00000000000000],
...         [+0.73178529739380, +2.28237795829773, 0.00000000000000],
...         [+5.89039325714111, +0.02589114569128, 0.00000000000000],
...         [-2.74426102638245, +2.16115570068359, 0.00000000000000],
...         [+2.74426102638245, -2.16115570068359, 0.00000000000000],
...     ]),
...     torch.tensor([
...         [-0.55569743203406, +1.09030425468557, 0.00000000000000],
...         [+0.51473634678469, +3.15152550263611, 0.00000000000000],
...         [+0.59869690244446, -1.16861263789477, 0.00000000000000],
...         [-0.45355203669134, -2.74568780438064, 0.00000000000000],
...         [+2.52721209544999, -1.29200800956867, 0.00000000000000],
...         [-2.63139587595376, +0.96447869452240, 0.00000000000000],
...     ]),
... ))
>>>
>>> # calculate coordination number
>>> cn = mctc.ncoord.cn_d4(numbers, positions)
>>>
>>> torch.set_printoptions(precision=10)
>>> print(cn)
tensor([[2.6886456013, 2.6886456013, 2.6314170361, 2.6314167976,
         0.8594539165, 0.9231414795, 0.8605306745, 0.8605306745,
         0.8594539165, 0.9231414795, 0.8568341732, 0.8568341732],
        [2.6886456013, 0.8568335176, 2.6314167976, 0.8605306745,
         0.8594532013, 0.9231414795, 0.0000000000, 0.0000000000,
         0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000]])
"""

import torch

from . import (
    autograd,
    batch,
    convert,
    data,
    exceptions,
    io,
    math,
    ncoord,
    storch,
    typing,
    units,
)
from ._version import __version__
from .io.read import read, read_chrg, read_uhf
from .io.write import write

__all__ = [
    "autograd",
    "batch",
    "convert",
    "data",
    "exceptions",
    "io",
    "math",
    "ncoord",
    "storch",
    "typing",
    "units",
    "__version__",
    "read",
    "read_chrg",
    "read_uhf",
    "write",
]
