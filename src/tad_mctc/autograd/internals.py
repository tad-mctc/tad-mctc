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

Note
----
`functorch` is shipped with PyTorch 1.13.0 and later. Earlier versions require
a separate installation.
"""

from __future__ import annotations

from .._version import __tversion__

__all__ = ["jacrev", "fjacrev", "vmap", "fvmap"]


if __tversion__ < (2, 0, 0):
    # We always use the compatiblity functions even if `functorch` is available,
    # because `functorch` does not work with custom autograd functions.
    from .compat import jacrev_compat as jacrev
    from .compat import vmap_compat as vmap

    try:
        from functorch import jacrev as fjacrev  # type: ignore[import-error]
        from functorch import vmap as fvmap  # type: ignore[import-error]
    except ModuleNotFoundError:
        # pylint: disable=invalid-name
        fjacrev = None
        fvmap = None
else:
    from torch.func import jacrev  # type: ignore[import-error]
    from torch.func import vmap  # type: ignore[import-error]

    fjacrev = jacrev
    fvmap = vmap
