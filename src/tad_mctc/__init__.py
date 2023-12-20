# This file is part of tad-mctc.
#
# SPDX-Identifier: LGPL-3.0
# Copyright (C) 2023 Marvin Friede
#
# tad-mctc is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# tad_mctc is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with tad-mctc. If not, see <https://www.gnu.org/licenses/>.
"""
Torch Autodiff Utilities
========================

This library is a collection of utility functions that are used in PyTorch (re-)implementations of projects from the `Grimme group <https://github.com/grimme-lab>`__.
In particular, the *tad-mctc* library provides:

- autograd functions (Jacobian, Hessian)

- batch utility (packing, masks, ...)

- atomic data (radii, EN, example molecules, ...)

- io (reading coordinate files)

- coordination numbers

- safeops (autograd-safe implementations of common functions)

- typing (base class for tensor-like behavior of arbitrary classes)

- units

The name is inspired by the Fortran pendant "modular computation tool chain library" (`mctc-lib <https://github.com/grimme-lab/mctc-lib/>`__).

.. note::

   This project is still in early development and the API is subject to change.
   Contributions are welcome, please checkout our
   `contributing guidelines <https://github.com/dftd4/tad_mctc/blob/main/CONTRIBUTING.md>`_.

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
    ncoord,
    storch,
    typing,
    units,
)
from .__version__ import __version__
