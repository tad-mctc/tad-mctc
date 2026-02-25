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
I/O Read: ORCA
==============

Reader for ORCA files.
"""

from __future__ import annotations

import torch

from ...exceptions import EmptyFileError, FormatErrorORCA
from ...typing import IO, Any, Tensor
from .frompath import create_path_reader

__all__ = ["read_orca_engrad"]


def read_orca_engrad_fileobj(
    fileobj: IO[Any],
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[Tensor, Tensor]:
    """
    Read ORCA's engrad file.

    Parameters
    ----------
    fileobj : IO[Any]
        The file-like object to read from.
    device : :class:`torch.device` | None, optional
        Device to store the tensor on. Defaults to `None`.
    dtype : :class:`torch.dtype` | None, optional
        Floating point data type of the tensor. Defaults to `None`.

    Returns
    -------
    Tensor
        Tensor of energy

    Raises
    ------
    ValueError
        File does not conform with the expected format.
    """
    start_grad = -1
    grad = []

    start_energy = -1
    energy = None

    lines: list[str] = fileobj.readlines()
    for i, line in enumerate(lines):
        # energy
        if line.startswith("# The current total energy in Eh"):
            start_energy = i + 2

        if i == start_energy:
            l = line.strip()
            if len(l) == 0:
                raise FormatErrorORCA(f"No energy found in {fileobj}.")
            energy = torch.tensor(float(l), device=device, dtype=dtype)
            start_energy = -1

        # gradient
        if line.startswith("# The current gradient in Eh/bohr"):
            start_grad = i + 2

        if i == start_grad:
            # abort if we hit the next "#"
            if line.startswith("#"):
                break

            l = line.strip()
            if len(l) == 0:
                raise FormatErrorORCA(f"No gradient found in {fileobj}.")

            grad.append(float(l))
            start_grad += 1

    if energy is None:
        raise EmptyFileError(f"File '{fileobj}' appears to be empty.")

    return energy, torch.tensor(grad, device=device, dtype=dtype).reshape(-1, 3)


read_orca_engrad = create_path_reader(read_orca_engrad_fileobj)
