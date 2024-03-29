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
Autograd Utility: PyTorch AD functions
======================================

This module imports PyTorch's own autograd functions, depending on the version.

Important! Before PyTorch 2.0.0, `functorch` does not work together with custom
autograd functions, which we definitely require. Additionally, `functorch`
imposes the implementation of a `forward` **and** `setup_context` method, i.e.,
the traditional way of using `forward` with the `ctx` argument does not work.
"""
from __future__ import annotations

import torch

__all__ = ["jacrev", "vmap"]


if torch.__version__ < (2, 0, 0):  # type: ignore[operator]
    try:
        from functorch import jacrev, vmap  # type: ignore[import-error]
    except ModuleNotFoundError:
        from .compat import jacrev_compat as jacrev
        from .compat import vmap_compat as vmap
else:
    from torch.func import jacrev, vmap  # type: ignore[import-error]
