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
Coordination number
===================

Functions for calculating the coordination numbers.

Example
-------
>>> import torch
>>> import tad_mctc as mctc
>>>
>>> # S22 system 4: formamide dimer
>>> numbers = mctc.batch.pack((
...     mctc.utils.to_number("C C N N H H H H H H O O".split()),
...     mctc.utils.to_number("C O N H H H".split()),
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
>>> torch.set_printoptions(precision=7)
>>> print(mctc.cn_d4(numbers, positions))
tensor([[2.6886456, 2.6886456, 2.6314170, 2.6314168, 0.8594539, 0.9231414,
         0.8605307, 0.8605307, 0.8594539, 0.9231414, 0.8568342, 0.8568342],
        [2.6886456, 0.8568335, 2.6314168, 0.8605307, 0.8594532, 0.9231415,
         0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000]])
"""
from .count import *
from .d3 import *
from .d4 import *
from .eeq import *
