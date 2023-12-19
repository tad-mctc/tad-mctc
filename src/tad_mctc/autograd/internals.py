# This file is part of tad_mctc.
#
# SPDX-Identifier: LGPL-3.0
# Copyright (C) 2023 Marvin Friede
#
# tad_mctc is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# tad_mctc is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with tad_mctc. If not, see <https://www.gnu.org/licenses/>.
"""
PyTorch AD functions
====================

This module imports PyTorch's own autograd functions, depending on the version.

Important! Before PyTorch 2.0.0, `functorch` does not work together with custom
autograd functions, which we definitely require. Additionally, `functorch`
imposes the implementation of a `forward` **and** `setup_context` method, i.e.,
the traditional way of using `forward` with the `ctx` argument does not work.
"""
import torch

__all__ = ["jacrev", "jacobian"]


if torch.__version__ < (2, 0, 0):  # type: ignore[import-error]
    try:
        from functorch import jacrev  # type: ignore[import-error]
    except ModuleNotFoundError:
        jacrev = None
        from torch.autograd.functional import jacobian  # type: ignore[import-error]

else:
    from torch.func import jacrev  # type: ignore[import-error]
