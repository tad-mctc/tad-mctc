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
I/O Checks: Molecule
====================

This module contains various checkers for the molecule that is read/written.

In particular,the following checks can be done:
- coldfusion_check (interatomic distances)
- content_checks (atomic numbers)
- deflatable_check (clash between padding and coordinates)

Note
----
The check `deflatable_check` attempts to catch cases, in which the padding
value (default: 0) is the same as a triple of atomic positions, which would
obscure the distinction between padding and actual atomic positions in batched
calculations. This primarily occurs for single atoms, which are usually placed
at the origin.
The behavior of this check is best controlled through keyword arguments of the
respective readers. The available keyword arguments are:
- padding_value (`float | int`, default: 0): Value for padding used in check
- raise_padding_exception (`bool`, default: False): Raise an exception (or just
a warning)
- raise_padding_warning (`bool`, default: True): Raise a warning
- shift_for_last (`bool`, default: False): Automatically shift all positions by
a constant if a clash is detected
- shift_value (`float | int`, default: 1.0): Constant for shift.

For more details and examples, check `test/test_io/test_deflatable.py`.
"""

from __future__ import annotations

import torch

from ... import storch
from ...batch import deflate, real_pairs
from ...data import pse
from ...exceptions import MoleculeError, MoleculeWarning
from ...typing import DD, IO, Any, NoReturn, Tensor

__all__ = [
    "coldfusion_check",
    "content_checks",
    "deflatable_check",
    "dimension_check",
]


def coldfusion_check(
    numbers: Tensor,
    positions: Tensor,
    threshold: Tensor | float | int | None = None,
) -> bool | NoReturn:
    """
    Check if interatomic distances are large enough (no fusion of atoms).

    Parameters
    ----------
    numbers : Tensor
        A 1D tensor containing atomic numbers or symbols.
    positions : Tensor
        A 2D tensor of shape (n_atoms, 3) containing atomic positions.
    threshold : Tensor | float | int | None, optional
        Threshold for acceptable interatomic distances. Defaults to `None`,
        which resolves to `torch.tensor(torch.finfo(dtype).eps ** 0.75, **dd)`.

    Returns
    -------
    bool
        True of atoms are not too close.

    Raises
    ------
    MoleculeError
        Interatomic distances are too close.
    """
    dd: DD = {"device": positions.device, "dtype": positions.dtype}

    mask = real_pairs(numbers, mask_diagonal=True)
    distances = torch.where(
        mask,
        storch.cdist(positions, positions),
        torch.tensor(1e100, **dd),
    )

    if threshold is None:
        threshold = torch.tensor(torch.finfo(dd["dtype"]).eps ** 0.75, **dd)

    # Check if any distance below the threshold is found
    if torch.any((distances < threshold) & mask):
        raise MoleculeError("Too close interatomic distances found")

    return True


def content_checks(
    numbers: Tensor,
    positions: Tensor,
    max_element: int = pse.MAX_ELEMENT,
    allow_batched: bool = True,
) -> bool | NoReturn:
    """
    Check the content of the numbers and positions tensors.

    This function should be asserted as it returns `True` on success and raises
    an error on failure.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers for all atoms in the system of shape ``(..., nat)``.
    positions : Tensor
        Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
    max_element : int, optional
        Maximum atomic number allowed. Defaults to
        :data:`tad_mctc.data.pse.MAX_ELEMENT`.
    allow_batched : bool, optional
        Allow batched tensors. Defaults to ``True``.

    Returns
    -------
    bool
        ``True`` if content is correct.

    Raises
    ------
    MoleculeError
        Atomic number too large or too small.
    """
    if numbers.max() > max_element:
        raise MoleculeError(f"Atomic number larger than {max_element} found.")

    if allow_batched is False:
        if numbers.min() < 1:
            raise MoleculeError(
                "Atomic number smaller than 1 found. This may indicate "
                "residual padding. Remove before writing to file."
            )

    assert coldfusion_check(numbers, positions)

    return True


def deflatable_check(
    positions: Tensor, fileobj: IO[Any] | None = None, **kwargs: Any
) -> bool | NoReturn:
    """
    Check for the last coordinate being at the origin as this might clash with
    padding.

    This function should be asserted as it returns ``True`` on success and
    raises an error on failure.

    Parameters
    ----------
    positions : Tensor
        A 2D tensor of shape ``(nat, 3)`` containing atomic positions.
    fileobj : IO[Any] | None, optional
        The file-like object from which is read (only for printing).

    Returns
    -------
    bool
        True if content is correct.

    Raises
    ------
    MoleculeError
        Padding clashes with coordinates. Requires the keyword argument
        ``raise_padding_exception=True``.
    """
    # collect the padding value
    pad = kwargs.pop("padding_value", 0)

    # do not deflate the coordinate axis (all z-coordinates could be zero)
    dpos = deflate(positions, value=pad, axis=1)
    if dpos.shape != positions.shape:
        msg = (
            f"The position tensor from '{fileobj}' cannot handle the padding "
            f"value '{pad}'. This commonly occurs for zero-padding if the last "
            "atom is in the origin."
        )

        # raise exception
        if kwargs.pop("raise_padding_exception", False):
            raise MoleculeError(msg)

        # shift all atoms
        if kwargs.pop("shift_for_last", False):
            positions += kwargs.pop("shift_value", 1.0)
            return True

        # issue warning
        if kwargs.pop("raise_padding_warning", True):
            # pylint: disable=import-outside-toplevel
            from warnings import warn

            warn(msg, MoleculeWarning)

    return True


def dimension_check(
    x: Any,
    min_ndim: int = -1,
    max_ndim: int = 9999,
) -> bool | NoReturn:
    """
    Check if the number of dimensions of a tensor is within a certain range.

    Parameters
    ----------
    x : Any
        The tensor to check.
    min_ndim : int, optional
        Minimum number of dimensions for the tensor. Defaults to ``-1``.
    max_ndim : int, optional
        Maximum number of dimensions for the tensor. Defaults to ``9999``.

    Returns
    -------
    None | NoReturn
        Returns ``None`` if the tensor has the correct number of dimensions.

    Raises
    ------
    TypeError
        If the input is not a tensor.
    RuntimeError
        If the number of dimensions is not within the specified range.

    Examples
    --------
    >>> import torch
    >>> from tad_mctc.io.checks.molecule import dimension_check
    >>> x = torch.tensor([1, 2, 3])
    >>> dimension_check(x, min_ndim=1, max_ndim=1)
    True
    >>> dimension_check(x, min_ndim=2, max_ndim=2)
    Traceback (most recent call last):
    ...
    RuntimeError: The tensor should not fall below '2' dimensions.
    """
    if not isinstance(x, Tensor):
        raise TypeError(f"Variable is not a tensor but '{type(x)}'.")

    if x.ndim < min_ndim:
        raise RuntimeError(
            f"The tensor should not fall below {min_ndim} dimensions."
        )
    if x.ndim > max_ndim:
        raise RuntimeError(
            f"The tensor should not exceed '{max_ndim}' dimensions."
        )

    return True
