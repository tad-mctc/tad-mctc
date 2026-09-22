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
I/O Read: Finalize (internal)
=============================

The steps every structure reader shares around its format-specific parsing:
resolving the float/integer tensor options before parsing, and validating
the parsed geometry afterwards. Not part of the public API; each reader
imports these directly.
"""

from __future__ import annotations

from typing import IO, Any

import torch

from ...typing import DD, Tensor, get_default_dtype
from ..checks import content_checks, deflatable_check, shape_checks

__all__: list[str] = []


def resolve_dd(
    device: torch.device | None,
    dtype: torch.dtype | None,
    dtype_int: torch.dtype,
) -> tuple[DD, DD]:
    """
    Tensor options for a reader's float and integer tensors.

    Parameters
    ----------
    device : :class:`torch.device` | None
        Device to store the tensors on.
    dtype : :class:`torch.dtype` | None
        Floating point data type. ``None`` resolves to PyTorch's default.
    dtype_int : :class:`torch.dtype`
        Integer data type.

    Returns
    -------
    tuple[DD, DD]
        Options for float tensors and for integer tensors.
    """
    dd: DD = {
        "device": device,
        "dtype": dtype if dtype is not None else get_default_dtype(),
    }
    ddi: DD = {"device": device, "dtype": dtype_int}
    return dd, ddi


def finalize_geometry(
    numbers: Tensor,
    positions: Tensor,
    fileobj: IO[Any],
    **kwargs: Any,
) -> Tensor:
    """
    Validate a single parsed geometry (shapes, atomic numbers, optionally
    interatomic distances, and clashes with the padding value).

    The checks raise on failure, so they also run under ``python -O``.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers of shape ``(nat,)``.
    positions : Tensor
        Cartesian coordinates of shape ``(nat, 3)``.
    fileobj : IO[Any]
        The file-like object that was read (only for error messages).
    **kwargs : Any
        The reader's keyword arguments. ``check_coldfusion`` and
        ``coldfusion_cutoff`` are forwarded to
        :func:`~tad_mctc.io.checks.content_checks`; the padding options
        (``padding_value``, ``shift_for_last``, ...) to
        :func:`~tad_mctc.io.checks.deflatable_check`.

    Returns
    -------
    Tensor
        The positions. With ``shift_for_last=True``, a clash with the
        padding value shifts them in place.
    """
    shape_checks(numbers, positions, allow_batched=False)
    content_checks(
        numbers,
        positions,
        allow_batched=False,
        check_coldfusion=kwargs.get("check_coldfusion", False),
        coldfusion_cutoff=kwargs.get("coldfusion_cutoff", 2.0),
    )
    deflatable_check(positions, fileobj, **kwargs)
    return positions
