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
Molecule: Representation
========================

This module contains a class for the representation of important molecular
information.

Example
-------
>>> import torch
>>> from tad_mctc.molecule.container import Mol
>>>
>>> numbers = torch.tensor([14, 1, 1, 1, 1])
>>> positions = torch.tensor([
...     [+0.00000000000000, +0.00000000000000, +0.00000000000000],
...     [+1.61768389755830, +1.61768389755830, -1.61768389755830],
...     [-1.61768389755830, -1.61768389755830, -1.61768389755830],
...     [+1.61768389755830, -1.61768389755830, +1.61768389755830],
...     [-1.61768389755830, +1.61768389755830, +1.61768389755830],
... ])
>>> mol = Mol(numbers, positions)
"""
from __future__ import annotations

import torch

from .. import storch
from ..convert import any_to_tensor
from ..exceptions import DeviceError, DtypeError
from ..io.checks import dimension_check
from ..tools import memoize
from ..typing import NoReturn, Tensor, TensorLike

__all__ = ["Mol"]


class Mol(TensorLike):
    """
    Representation of a molecule.
    """

    __slots__ = [
        "_numbers",
        "_positions",
        "_charge",
        "_name",
        "__memoization_cache",
    ]

    def __init__(
        self,
        numbers: Tensor,
        positions: Tensor,
        charge: Tensor | float | int | str = 0,
        name: str | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__(device, dtype)

        # check and transform all (possibly) non-tensor inputs to tensors
        charge = any_to_tensor(charge, device=self.device, dtype=self.dtype)

        self._numbers = numbers
        self._positions = positions
        self._charge = charge
        self._name = name

        self.checks()

    @property
    def numbers(self) -> Tensor:
        """Atomic numbers of the molecule."""
        return self._numbers

    @numbers.setter
    def numbers(self, numbers: Tensor) -> None:
        self._numbers = numbers
        self.checks()

    @property
    def positions(self) -> Tensor:
        """Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``)."""
        return self._positions

    @positions.setter
    def positions(self, positions: Tensor) -> None:
        self._positions = positions
        self.checks()

    @property
    def charge(self) -> Tensor:
        """Charge of the molecule."""
        return self._charge

    @charge.setter
    def charge(self, charge: Tensor | float | int | str) -> None:
        self._charge = any_to_tensor(
            charge,
            device=self.device,
            dtype=self.dtype,
        )
        self.checks()

    @property
    def name(self) -> str | None:
        """Name of the molecule."""
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        self._name = name

    @memoize
    def distances(self) -> Tensor:
        """
        Calculate the distance matrix from the positions.

        .. warning::

            Memoization for this method creates a cache that stores the
            distances across all instances.

        Returns
        -------
        Tensor
            Distance matrix.
        """
        return storch.cdist(self.positions)

    def clear_cache(self) -> None:
        """Clear the cross-instance caches of all memoized methods."""
        if hasattr(self.distances, "clear"):
            self.distances.clear(self)

    def checks(self) -> None | NoReturn:
        """
        Check all variables for consistency.

        Raises
        ------
        RuntimeError
            Wrong device or shape errors.
        """

        # check tensor type inputs
        dimension_check(self.numbers, min_ndim=1, max_ndim=2)
        dimension_check(self.positions, min_ndim=2, max_ndim=3)
        dimension_check(self.charge, min_ndim=0, max_ndim=1)

        allowed_dtypes = (torch.long, torch.int16, torch.int32, torch.int64)
        if self.numbers.dtype not in allowed_dtypes:
            raise DtypeError(
                "Dtype of atomic numbers must be one of the following to allow "
                f" indexing: '{', '.join([str(x) for x in allowed_dtypes])}', "
                f"but is '{self.numbers.dtype}'"
            )

        # check if all tensors are on the same device
        for s in self.__slots__:
            if s.startswith("__"):
                continue

            attr = getattr(self, s)
            if isinstance(attr, Tensor):
                if attr.device != self.device:
                    raise DeviceError("All tensors must be on the same device!")

        if self.numbers.shape != self.positions.shape[:-1]:
            raise RuntimeError(
                f"Shape of positions ({self.positions.shape[:-1]}) is not "
                f"consistent with atomic numbers ({self.numbers.shape})."
            )

    def sum_formula(self) -> str:
        """
        Calculate the sum formula of the molecule.

        Returns
        -------
        str
            Sum formula.
        """
        from ..data import pse

        formula = ""

        unique, counts = torch.unique(self.numbers, return_counts=True)
        for u, c in zip(unique, counts):
            formula += f"{pse.Z2S[int(u)]}"
            if c > 1:
                formula += f"{c}"

        return formula

    def __str__(self) -> str:  # pragma: no cover
        return f"{self.__class__.__name__}({self.numbers.tolist()})"

    def __repr__(self) -> str:  # pragma: no cover
        return str(self)
