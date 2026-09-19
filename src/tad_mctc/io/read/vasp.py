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
I/O Read: VASP
==============

Reader for VASP ``POSCAR``/``CONTCAR`` geometry files.
See https://www.vasp.at/wiki/index.php/POSCAR. Mirrors mctc-lib's
``mctc_io_read_vasp`` (``src/mctc/io/read/vasp.f90``).

Unlike every other reader in this package, a POSCAR always carries a unit
cell, so ``read_poscar_fileobj`` returns a 3-tuple (numbers, positions,
lattice) instead of the usual (numbers, positions) pair.
"""

from __future__ import annotations

import itertools
from typing import IO, Any, Iterator

import torch

from ...convert import symbol_to_number
from ...exceptions import EmptyFileError, FormatErrorVASP
from ...typing import DD, Tensor, get_default_dtype
from ...units import length
from ..checks import content_checks, deflatable_check, shape_checks
from .frompath import create_path_reader_lattice

__all__ = ["read_poscar"]


def _next_line(lines: Iterator[str], fileobj: IO[Any], context: str) -> str:
    """Pop the next raw line, translating exhaustion into `FormatErrorVASP`."""
    try:
        return next(lines)
    except StopIteration as e:
        raise FormatErrorVASP(
            f"Unexpected end of POSCAR file '{fileobj}' while reading "
            f"{context}."
        ) from e


def _parse_floats(
    line: str, n: int, fileobj: IO[Any], context: str
) -> list[float]:
    """Parse the first `n` whitespace-separated tokens of `line` as floats."""
    parts = line.split()
    try:
        if len(parts) < n:
            raise ValueError("not enough values")
        return [float(x) for x in parts[:n]]
    except ValueError as e:
        raise FormatErrorVASP(
            f"Cannot read {context} from '{fileobj}': expected {n} real "
            f"value(s) in line {line!r}."
        ) from e


def read_poscar_fileobj(
    fileobj: IO[Any],
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    dtype_int: torch.dtype = torch.long,
    **kwargs: Any,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Reads a VASP POSCAR/CONTCAR file and returns atomic numbers, positions,
    and lattice vectors as tensors. Both VASP5+ (element symbols on their
    own line) and pre-VASP5 (element symbols on the comment line) formats
    are supported, as are ``Selective dynamics`` and ``Direct``/fractional
    coordinates.

    Parameters
    ----------
    fileobj : IO[Any]
        The file-like object to read from.
    device : :class:`torch.device` | None, optional
        Device to store the tensor on. Defaults to `None`.
    dtype : :class:`torch.dtype` | None, optional
        Floating point data type of the tensor. Defaults to `None`.
    dtype_int : torch.dtype, optional
        Integer data type of the tensor. Defaults to `torch.long`.

    Returns
    -------
    (Tensor, Tensor, Tensor)
        Tensors of atomic numbers, positions, and lattice vectors (as rows).
        Positions and lattice are given in atomic units (bohr).

    Raises
    ------
    EmptyFileError
        The file contains no lines at all.
    FormatErrorVASP
        The file does not conform with the expected POSCAR format.
    """
    dd: DD = {
        "device": device,
        "dtype": dtype if dtype is not None else get_default_dtype(),
    }
    ddi: DD = {"device": device, "dtype": dtype_int}

    raw_lines = fileobj.readlines()
    if not raw_lines:
        raise EmptyFileError(f"File '{fileobj}' is empty.")

    lines = iter(raw_lines)

    comment = _next_line(lines, fileobj, "comment line").strip()
    scale_line = _next_line(lines, fileobj, "scaling factor")
    (scale,) = _parse_floats(scale_line, 1, fileobj, "scaling factor")

    lattice_rows = [
        _parse_floats(
            _next_line(lines, fileobj, "lattice vectors"),
            3,
            fileobj,
            "lattice vectors",
        )
        for _ in range(3)
    ]
    lattice = torch.tensor(lattice_rows, **dd) * scale * length.AA2AU

    line = _next_line(lines, fileobj, "element symbols or atom counts")
    tokens = line.split()
    is_counts_line = bool(tokens) and all(t.isdigit() for t in tokens)

    if is_counts_line:
        # pre-VASP5: element symbols were given on the comment line
        symbols = comment.split()
        counts_tokens = tokens
    else:
        symbols = tokens
        counts_line = _next_line(lines, fileobj, "atom counts")
        counts_tokens = counts_line.split()

    if not symbols or len(counts_tokens) != len(symbols):
        raise FormatErrorVASP(
            f"Number of element symbols ({len(symbols)}) does not match "
            f"the number of atom counts ({len(counts_tokens)}) in "
            f"'{fileobj}'."
        )

    try:
        counts = [int(c) for c in counts_tokens]
    except ValueError as e:
        raise FormatErrorVASP(
            f"Cannot read atom counts from '{fileobj}': {counts_tokens}."
        ) from e

    numbers_per_symbol = []
    for s in symbols:
        number = symbol_to_number(s)
        if number is None:
            raise FormatErrorVASP(
                f"Unknown element symbol {s!r} in '{fileobj}'."
            )
        numbers_per_symbol.append(number)

    line = _next_line(lines, fileobj, "selective dynamics or coordinate type")
    if line.strip()[:1] in ("s", "S"):
        line = _next_line(lines, fileobj, "coordinate type")
    cartesian = line.strip()[:1] in ("c", "C", "k", "K")

    natoms = sum(counts)
    coords_rows = [
        _parse_floats(
            _next_line(lines, fileobj, "atomic coordinates"),
            3,
            fileobj,
            "atomic coordinates",
        )
        for _ in range(natoms)
    ]
    coords = torch.tensor(coords_rows, **dd)

    if cartesian:
        positions = coords * scale * length.AA2AU
    else:
        positions = coords @ lattice

    numbers = torch.tensor(
        list(
            itertools.chain.from_iterable(
                itertools.repeat(z, c)
                for z, c in zip(numbers_per_symbol, counts)
            )
        ),
        **ddi,
    )

    assert shape_checks(numbers, positions, allow_batched=False)
    assert content_checks(
        numbers,
        positions,
        allow_batched=False,
        check_coldfusion=kwargs.get("check_coldfusion", False),
        coldfusion_cutoff=kwargs.get("coldfusion_cutoff", 2.0),
    )
    assert deflatable_check(positions, fileobj, **kwargs)

    return numbers, positions, lattice


read_poscar = create_path_reader_lattice(read_poscar_fileobj)
