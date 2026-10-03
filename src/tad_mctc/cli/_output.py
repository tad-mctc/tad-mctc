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
Command line interface: output
==============================

The sections the tool prints after a run.
"""

from __future__ import annotations

import torch

from ..data import pse
from ..io.structure import Structure
from ..neighbor import _native
from ..neighbor.list import NeighborList
from ..properties.general import sum_formula
from ..typing import Tensor

__all__ = [
    "print_native_build",
    "print_neighborlist",
    "print_results",
    "print_system_info",
]


def _format_integers(values: Tensor) -> str:
    """A charge or unpaired-electron count, one value per frame of a
    batch unless all frames share it."""
    if values.numel() == 1:
        return f"{values.item():.0f}"
    return " ".join(f"{v:.0f}" for v in values.flatten().tolist())


def print_system_info(path: str, structure: Structure) -> None:
    """Print a short summary of the structure that was read: path, frame
    count of a batch (a multi-frame file), atom count and element
    composition over all frames, charge/unpaired-electron count if set,
    and whether it is periodic."""
    numbers = structure.numbers
    # Padding atoms of the shorter frames of a batch are not atoms.
    is_real = numbers != 0

    print()
    print("System")
    print("------")
    print(f"  file      {path}")
    if numbers.ndim > 1:
        print(f"  frames    {numbers.shape[:-1].numel()}")
    print(f"  atoms     {int(is_real.count_nonzero())}")
    print(f"  elements  {sum_formula(numbers[is_real])}")
    if structure.charge is not None:
        print(f"  charge    {_format_integers(structure.charge)}")
    if structure.uhf is not None:
        print(f"  uhf       {_format_integers(structure.uhf)}")
    print(f"  periodic  {'yes' if structure.lattice is not None else 'no'}")


def _describe_atom(numbers: Tensor, index: Tensor) -> str:
    """Name the atom at ``index``, ``(frame, atom)`` for a batch, one-based
    as in the structure file."""
    position = index.tolist()
    symbol = pse.Z2S[int(numbers[tuple(position)])]
    if len(position) == 1:
        return f"atom {position[0] + 1} {symbol}"
    frame, atom = position
    return f"frame {frame + 1}, atom {atom + 1} {symbol}"


def print_native_build(info: _native.BuildInfo) -> None:
    """Print how this run obtained the native CPU extension of the
    neighbour search, and whether it had to compile it."""
    if info.origin == "jit":
        origin = "compiled on first use (JIT), " + (
            "compiled in this run" if info.compiled_now else "cached build"
        )
    else:
        origin = {
            "precompiled": "precompiled (TAD_MCTC_BUILD_NATIVE=1 install)",
            "disabled": "disabled by TAD_MCTC_DISABLE_NATIVE, pure Python",
            "unavailable": "failed to load or compile, pure Python",
        }[info.origin]

    print()
    print("Native extension")
    print("----------------")
    print(f"  origin    {origin}")
    if info.library is not None:
        print(f"  library   {info.library}")
    if info.compiler is not None:
        print(f"  compiler  {info.compiler}")
    if info.cflags:
        print(f"  flags     {' '.join(info.cflags)}")
    if info.origin == "jit":
        print(f"  ninja     {info.ninja or 'not found'}")
    if info.error is not None:
        print(f"  error     {info.error}")
    print(f"  threads   {torch.get_num_threads()}")


def print_results(cn_name: str, cn: Tensor, structure: Structure) -> None:
    """Print summary statistics of the computed coordination number
    instead of the raw per-atom tensor, which is unreadable past a
    handful of atoms. For a batch, the statistics run over the real atoms
    of all frames; the zero CN of a padding atom is left out."""
    is_real = structure.numbers != 0
    real_cn = cn[is_real]
    # `nonzero` lists the real atoms in the same order as `cn[is_real]`.
    real_index = is_real.nonzero()
    imin = int(torch.argmin(real_cn))
    imax = int(torch.argmax(real_cn))

    print()
    print(f"Results ({cn_name})")
    print("-" * (10 + len(cn_name)))
    print(f"  mean      {real_cn.mean().item():.6f}")
    # Population std (`correction=0`): the real atoms are the whole set,
    # not a sample, and Bessel's correction is `nan` for a single atom.
    print(f"  std       {real_cn.std(correction=0).item():.6f}")
    print(
        f"  min       {real_cn[imin].item():.6f}  "
        f"({_describe_atom(structure.numbers, real_index[imin])})"
    )
    print(
        f"  max       {real_cn[imax].item():.6f}  "
        f"({_describe_atom(structure.numbers, real_index[imax])})"
    )


def print_neighborlist(nbl: NeighborList) -> None:
    """Print the size of the neighbour list that was built."""
    # `sum` would first copy the boolean mask to int64, 8 bytes per slot.
    n_pairs = int(nbl.mask.count_nonzero())

    print()
    print("Neighbour list")
    print("--------------")
    print(f"  cutoff    {nbl.cutoff:.4f}")
    print(f"  skin      {nbl.skin:.4f}")
    print(f"  pairs     {n_pairs}")
    print(f"  capacity  {nbl.mask.numel()}")
    print(f"  overflow  {'yes' if nbl.overflow else 'no'}")
