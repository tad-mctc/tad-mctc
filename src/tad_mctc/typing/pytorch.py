# This file is part of tad-mctc.
#
# SPDX-Identifier: LGPL-3.0
# Copyright (C) 2023 Marvin Friede
#
# tad-mctc is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# tad_mctc is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with tad-mctc. If not, see <https://www.gnu.org/licenses/>.
"""
Typing: PyTorch
===============

This module contains PyTorch-related type annotations.

Most importantly, the `TensorLike` base class is defined, which brings
tensor-like behavior (`.to` and `.type` methods) to classes.
"""
from __future__ import annotations

import torch
from torch import Tensor

from ..exceptions import DtypeError
from .builtin import Any, NoReturn, TypedDict
from .compat import Self


class Molecule(TypedDict):
    """Representation of fundamental molecular structure (atom types and postions)."""

    numbers: Tensor
    """Tensor of atomic numbers"""

    positions: Tensor
    """Tensor of 3D coordinates of shape (n, 3)"""


class DD(TypedDict):
    """Collection of torch.device and torch.dtype."""

    device: torch.device | None
    """Device on which a tensor lives."""

    dtype: torch.dtype
    """Floating point precision of a tensor."""


def get_default_device() -> torch.device:
    """
    Default device for tensors.

    Returns
    -------
    torch.device
        PyTorch `device` type.
    """
    return torch.tensor(1.0).device


def get_default_dtype() -> torch.dtype:
    """
    Default data type for floating point tensors.

    Returns
    -------
    torch.dtype
        PyTorch `dtype` type.
    """
    return torch.tensor(1.0).dtype


class TensorLike:
    """
    Provide `device` and `dtype` as well as `to()` and `type()` for other
    classes.

    The selection of `torch.Tensor` variables to change within the class is
    handled by searching `__slots__`. Hence, if one wants to use this
    functionality the subclass of `TensorLike` must specify `__slots__`.
    """

    __device: torch.device
    """The device on which the class object resides."""

    __dtype: torch.dtype
    """Floating point dtype used by class object."""

    __dd: DD
    """Shortcut for device and dtype."""

    __slots__ = ["__device", "__dtype", "__dd"]

    def __init__(
        self, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        self.__device = device if device is not None else get_default_device()
        self.__dtype = dtype if dtype is not None else get_default_dtype()
        self.__dd = {"device": self.device, "dtype": self.dtype}

    @property
    def device(self) -> torch.device:
        """The device on which the class object resides."""
        return self.__device

    @device.setter
    def device(self, *_: Any) -> NoReturn:
        """
        Instruct users to use the ".to" method if wanting to change device.

        Returns
        -------
        NoReturn
            Always raises an `AttributeError`.

        Raises
        ------
        AttributeError
            Setter is called.
        """
        raise AttributeError("Move object to device using the `.to` method")

    @property
    def dtype(self) -> torch.dtype:
        """Floating point dtype used by class object."""
        return self.__dtype

    @dtype.setter
    def dtype(self, *_: Any) -> NoReturn:
        """
        Instruct users to use the `.type` method if wanting to change dtype.

        Returns
        -------
        NoReturn
            Always raises an `AttributeError`.

        Raises
        ------
        AttributeError
            Setter is called.
        """
        raise AttributeError("Change object to dtype using the `.type` method")

    @property
    def dd(self) -> DD:
        """Shortcut for device and dtype."""
        return self.__dd

    @dd.setter
    def dd(self, *_: Any) -> NoReturn:
        """
        Instruct users to use the `.type` and `.to` methods to change.

        Returns
        -------
        NoReturn
            Always raises an `AttributeError`.

        Raises
        ------
        AttributeError
            Setter is called.
        """
        raise AttributeError(
            "Change object to dtype and device using the `.type` and `.to` "
            "methods, respectively."
        )

    def type(self, dtype: torch.dtype) -> Self:
        """
        Returns a copy of the `TensorLike` instance with specified floating
        point type.
        This method creates and returns a new copy of the `TensorLike` instance
        with the specified dtype.

        Parameters
        ----------
        dtype : torch.dtype
            Floating point type.

        Returns
        -------
        TensorLike
            A copy of the `TensorLike` instance with the specified dtype.

        Notes
        -----
        If the `TensorLike` instance has already the desired dtype `self` will
        be returned.
        """
        if self.dtype == dtype:
            return self

        if len(self.__slots__) == 0:
            raise RuntimeError(
                f"The `type` method requires setting `__slots__` in the "
                f"'{self.__class__.__name__}' class."
            )

        if dtype not in self.allowed_dtypes:
            raise DtypeError(
                f"Only '{self.allowed_dtypes}' allowed (received '{dtype}')."
            )

        args = {}
        for s in self.__slots__:
            if not s.startswith("__"):
                attr = getattr(self, s)
                if isinstance(attr, Tensor) or issubclass(type(attr), TensorLike):
                    if attr.dtype in self.allowed_dtypes:
                        attr = attr.type(dtype)  # type: ignore
                args[s] = attr

        return self.__class__(**args, dtype=dtype)

    def to(self, device: torch.device) -> Self:
        """
        Returns a copy of the `TensorLike` instance on the specified device.

        This method creates and returns a new copy of the `TensorLike` instance
        on the specified device "``device``".

        Parameters
        ----------
        device : torch.device
            Device to which all associated tensors should be moved.

        Returns
        -------
        TensorLike
            A copy of the `TensorLike` instance placed on the specified device.

        Notes
        -----
        If the `TensorLike` instance is already on the desired device `self`
        will be returned.
        """
        if self.device == device:
            return self

        if len(self.__slots__) == 0:
            raise RuntimeError(
                f"The `to` method requires setting `__slots__` in the "
                f"'{self.__class__.__name__}' class."
            )

        args = {}
        for s in self.__slots__:
            if not s.startswith("__"):
                attr = getattr(self, s)
                if isinstance(attr, Tensor) or issubclass(type(attr), TensorLike):
                    attr = attr.to(device=device)  # type: ignore
                args[s] = attr

        return self.__class__(**args, device=device)

    @property
    def allowed_dtypes(self) -> tuple[torch.dtype, ...]:
        """
        Specification of dtypes that the TensorLike object can take. Defaults
        to float types and must be overridden by subclass if float are not
        allowed. The IndexHelper is an example that should only allow integers.

        Returns
        -------
        tuple[torch.dtype, ...]
            Collection of allowed dtypes the TensorLike object can take.
        """
        return (torch.float16, torch.float32, torch.float64)
