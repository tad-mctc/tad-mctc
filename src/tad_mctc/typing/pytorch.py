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
Typing: PyTorch
===============

This module contains PyTorch-related type annotations.

Most importantly, the `TensorLike` base class is defined, which brings
tensor-like behavior (`.to` and `.type` methods) to classes.
"""

from __future__ import annotations

from typing import Any, NoReturn, Protocol, TypedDict

import torch
from torch import Tensor

from ..exceptions import DtypeError
from .compat import CountingFunction, Self

__all__ = [
    "CNFunction",
    "CNGradFunction",
    "DD",
    "MockTensor",
    "Molecule",
    "get_default_device",
    "get_default_dtype",
    "Tensor",
    "TensorLike",
]


class DD(TypedDict):
    """Collection of torch.device and torch.dtype."""

    device: torch.device | None
    """Device on which a tensor lives."""

    dtype: torch.dtype
    """Floating point precision of a tensor."""


class Molecule(TypedDict):
    """Representation of fundamental molecular structure (atom types and postions)."""

    numbers: Tensor
    """Tensor of atomic numbers"""

    positions: Tensor
    """Tensor of 3D coordinates of shape (n, 3)"""


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


class MockTensor(Tensor):
    """
    Custom Tensor class with overridable device property.

    This can be used for testing different devices on systems, which only have
    a single device or no CUDA support.
    """

    @property
    def device(self) -> Any:
        """Overridable device property."""
        return self._device

    @device.setter
    def device(self, value: Any) -> None:  # type: ignore
        self._device = value


class TensorLike:
    """
    Provide ``device`` and ``dtype`` as well as :meth:`torch.Tensor.to`
    and :meth:`torch.Tensor.type` for other classes.

    The selection of :class:`torch.Tensor` variables to change within the
    class is handled by searching ``__slots__``. Hence, if one wants to use
    this functionality the subclass of :class:`.TensorLike` must specify
    ``__slots__``.
    """

    __device: torch.device
    """The device on which the class object resides."""

    __dtype: torch.dtype
    """Floating point dtype used by class object."""

    __slots__ = ["__device", "__dtype"]

    def __init__(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        self.__device = device if device is not None else get_default_device()
        self.__dtype = dtype if dtype is not None else get_default_dtype()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        if not hasattr(cls, "__slots__"):
            raise TypeError(
                f"Subclasses of {cls.__name__} must define `__slots__` "
                "make use of its functionality."
            )

        super().__init_subclass__(**kwargs)

    def _clone_tensorlike(
        self,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Self:
        """Helper function to clone with new device and dtype."""
        device = device if device is not None else self.device
        dtype = dtype if dtype is not None else self.dtype

        # Create an empty instance bypassing __init__
        new_obj = self.__class__.__new__(self.__class__)

        # Copy the tensor attributes from __slots__
        for slot in self.__slots__:
            # Skip private attributes managed directly (like __device, __dtype)
            if slot.startswith("__"):
                continue

            attr = getattr(self, slot)
            if not (
                isinstance(attr, Tensor) or issubclass(type(attr), TensorLike)
            ):
                setattr(new_obj, slot, attr)
                continue

            if attr.device == device and attr.dtype == dtype:
                setattr(new_obj, slot, attr)
                continue

            # Skip if the new dtype is not in the list of allowed dtypes
            if hasattr(attr, "allowed_dtypes"):
                if dtype not in attr.allowed_dtypes:  # type: ignore
                    setattr(new_obj, slot, attr)
                    continue

            attr = attr.to(device=device, dtype=dtype)
            setattr(new_obj, slot, attr)

        # Manually set device and dtype
        new_obj.override_device(device)
        new_obj.override_dtype(dtype)

        # Copy over any additional attributes
        # (if needed, e.g. ones that are not in __slots__)
        extra_attrs = getattr(self, "__dict__", None)
        if extra_attrs is not None:
            for key, value in extra_attrs.items():
                setattr(new_obj, key, value)

        return new_obj

    @property
    def device(self) -> torch.device:
        """The device on which the class object resides."""
        return self.__device

    @device.setter
    def device(self, *_: Any) -> NoReturn:
        """
        Instruct users to use the :meth:`to` method if wanting to change device.

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

    def override_device(self, device: torch.device) -> None:
        """
        Override the device of the class object.

        .. warning::

            This does not change the device of the underlying tensors. It only
            changes the device of the class object. Use with caution.

        Parameters
        ----------
        device : :class:`torch.device`
            Device to override the current device.

        """
        self.__device = device

    ###########################################################################

    @property
    def dtype(self) -> torch.dtype:
        """Floating point dtype used by class object."""
        return self.__dtype

    @dtype.setter
    def dtype(self, *_: Any) -> NoReturn:
        """
        Instruct users to use the :meth:`type` method if wanting to change
        dtype.

        Returns
        -------
        NoReturn
            Always raises an ``AttributeError``.

        Raises
        ------
        AttributeError
            Setter is called.
        """
        raise AttributeError("Change object to dtype using the `.type` method")

    def override_dtype(self, dtype: torch.dtype) -> None:
        """
        Override the dtype of the class object.

        .. warning::

            This does not change the dtype of the underlying tensors. It only
            changes the dtype of the class object. Use with caution.

        Parameters
        ----------
        dtype : :class:`torch.dtype`
            Floating point dtype to override the current dtype.
        """
        self.__dtype = dtype

    ###########################################################################

    @property
    def dd(self) -> DD:
        """Shortcut for device and dtype."""
        return {"device": self.device, "dtype": self.dtype}

    @dd.setter
    def dd(self, *_: Any) -> NoReturn:
        """
        Instruct users to use the :meth:`type` and :meth:`to` methods to change.

        Returns
        -------
        NoReturn
            Always raises an ``AttributeError``.

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
        Returns a copy of the :class:`.TensorLike` instance with specified
        floating point type.
        This method creates and returns a new copy of the :class:`.TensorLike`
        instance with the specified dtype.

        Parameters
        ----------
        dtype : :class:`torch.dtype`
            Floating point type.

        Returns
        -------
        TensorLike
            A copy of the :class:`.TensorLike` instance with the specified
            dtype.

        Notes
        -----
        If the :class:`.TensorLike` instance has already the desired dtype
        ``Self`` will be returned.
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

        return self._clone_tensorlike(device=None, dtype=dtype)

    def to(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Self:
        """
        Returns a copy of the :class:`.TensorLike` instance on the specified
        device.

        This method creates and returns a new copy of the :class:`.TensorLike`
        instance on the specified device "``device``".

        Parameters
        ----------
        device : :class:`torch.device`
            Device to which all associated tensors should be moved.

        Returns
        -------
        TensorLike
            A copy of the :class:`.TensorLike` instance placed on the specified
            device.

        Notes
        -----
        If the :class:`.TensorLike` instance is already on the desired device
        ``self`` will be returned.
        """
        if device is None and dtype is None:
            return self

        # Default to current device and dtype if not specified
        device = device if device is not None else self.device
        dtype = dtype if dtype is not None else self.dtype

        # If both device and dtype are the same, return self
        if self.device == device and self.dtype == dtype:
            return self

        if len(self.__slots__) == 0:
            raise RuntimeError(
                f"The `to` method requires setting `__slots__` in the "
                f"'{self.__class__.__name__}' class."
            )

        if dtype not in self.allowed_dtypes:
            raise DtypeError(
                f"Only '{self.allowed_dtypes}' allowed (received '{dtype}')."
            )

        return self._clone_tensorlike(device=device, dtype=dtype)

    def cpu(self) -> Self:
        """
        Returns a copy of the :class:`.TensorLike` instance on the CPU.

        This method creates and returns a new copy of the :class:`.TensorLike`
        instance on the CPU.

        Returns
        -------
        TensorLike
            A copy of the :class:`.TensorLike` instance placed on the CPU.
        """
        return self.to(torch.device("cpu"))

    @property
    def allowed_dtypes(self) -> tuple[torch.dtype, ...]:
        """
        Specification of dtypes that the :class:`.TensorLike` object can take.
        Defaults to float types and must be overridden by subclass if float are
        not allowed. The IndexHelper is an example that should only allow
        integers.

        Returns
        -------
        tuple[torch.dtype, ...]
            Collection of allowed dtypes the :class:`.TensorLike` object can
            take.
        """
        return (torch.float16, torch.float32, torch.float64)


class CNFunction(Protocol):
    """
    Type annotation for coordination number function.
    """

    def __call__(
        self,
        numbers: Tensor,
        positions: Tensor,
        *,
        counting_function: CountingFunction | None = None,
        rcov: Tensor | None = None,
        en: Tensor | None = None,
        cutoff: Tensor | None = None,
        kcn: float = 7.5,
        **kwargs: Any,
    ) -> Tensor:
        """
        Calculate the coordination number of each atom in the system.
        """
        ...


class CNGradFunction(Protocol):
    """
    Type annotation for coordination number function.
    """

    def __call__(
        self,
        numbers: Tensor,
        positions: Tensor,
        *,
        dcounting_function: CountingFunction | None = None,
        rcov: Tensor | None = None,
        en: Tensor | None = None,
        cutoff: Tensor | None = None,
        kcn: float = 7.5,
        **kwargs: Any,
    ) -> Tensor:
        """
        Calculate the coordination number gradient of each atom in the system.
        """
        ...
