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
Autograd Utility
================

This module contains utility functions for automatic differentiation, which
includes:
- Jacobians without function transforms (row by row)
- gradient checks and checks for function-transformed tensors
- unwrapping of function-transformed tensors

For Jacobians, Hessians and vectorization, use PyTorch's own function
transforms in ``torch.func`` directly, e.g. ``jacrev(jacrev(f))`` for a
Hessian (reverse-over-reverse, which needs no forward-mode rules) or
``vmap`` over it for a batch.
"""

from .checks import *
from .gradcheck import *
from .nonfunctorch import *
from .unwrap import *
