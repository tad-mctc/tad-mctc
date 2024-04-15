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
Batch Utility: Jitted Versions
==============================

Jitted versions of masking functions. Apparently, the traced versions are
not faster than the original functions. This is likely due to the fact that
the functions are already quite simple and do not benefit from the tracing
optimizations.
"""

import torch

from ...typing import Tensor
from .atoms import real_atoms
from .pairs import real_pairs_maskdiag, real_pairs_no_maskdiag

__all__ = ["real_atoms_traced", "real_pairs_traced"]


# Initial example input
example_input = torch.tensor([1, 0, 6, 0, 8])

# Additional inputs for checking
check_inputs = [
    (torch.tensor([3, 1]),),
    (torch.tensor([1, 1, 1, 0, 0, 0]),),
    (
        torch.tensor(
            [
                [9, 8, 7, 6, 5, 4, 3, 2, 1],
                [8, 1, 1, 0, 0, 0, 0, 0, 0],
            ]
        ),
    ),
]


################################################################################


real_atoms_traced: torch.jit.ScriptModule = torch.jit.trace(
    real_atoms, (example_input,), check_inputs=check_inputs
)  # type: ignore


################################################################################


real_pairs_maskdiag_traced: torch.jit.ScriptModule = torch.jit.trace(
    real_pairs_maskdiag, (example_input,), check_inputs=check_inputs
)  # type: ignore


real_pairs_no_maskdiag_traced: torch.jit.ScriptModule = torch.jit.trace(
    real_pairs_no_maskdiag, (example_input,), check_inputs=check_inputs
)  # type: ignore


def real_pairs_traced(numbers: Tensor, mask_diagonal: bool = True) -> Tensor:
    """
    Create a mask for pairs of atoms from atomic numbers, discerning padding
    and actual atoms. Padding value is zero.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers for all atoms.
    mask_diagonal : bool, optional
        Flag for also masking the diagonal, i.e., all pairs with the same
        indices. Defaults to `True`, i.e., writing False to the diagonal.

    Returns
    -------
    Tensor
        Mask for atom pairs that discerns padding and real atoms.
    """
    if mask_diagonal is True:
        return real_pairs_maskdiag_traced(numbers)
    return real_pairs_no_maskdiag_traced(numbers)
