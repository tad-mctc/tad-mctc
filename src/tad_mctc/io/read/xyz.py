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
I/O Read: XYZ
=============

Reader for standard XYZ files.
See https://en.wikipedia.org/wiki/XYZ_file_format.

Also supports the Extended XYZ dialect (comment line carrying a
``key=value`` header), mirroring mctc-lib's ``parse_extxyz_header``/
``parse_properties``/``read_extxyz_atom`` (``src/mctc/io/read/xyz.f90``):
a comment line is only treated as an Extended XYZ header if it contains a
``Properties`` key (e.g. ``Properties=species:S:1:pos:R:3``); ``Lattice``
(a 9- or 3-value list, bohr after conversion) and ``pbc`` (3 booleans,
``pbc`` overriding the all-``True`` default that ``Lattice`` alone
implies) are both optional and, together, are how periodicity is
declared. A ``pbc`` marking a periodic axis needs a ``Lattice``, whereas
mctc-lib accepts it and keeps a zero cell. Unlike every other case in this reader, mctc-lib's own
``read_xyz`` handles exactly one frame -- multi-frame batching is this
package's own pre-existing extension, so a periodic frame is only
supported when the whole file is a single frame; a multi-frame file with
periodicity in any frame raises :class:`FormatErrorXYZ` rather than
inventing a per-frame-batched lattice/periodic convention.
"""

from __future__ import annotations

import io as _io
import itertools
from dataclasses import dataclass
from typing import IO, Any

import numpy as np
import torch

from ...batch import pack
from ...convert import symbol_to_number
from ...exceptions import EmptyFileError, FormatErrorXYZ
from ...typing import DD, Tensor
from ...units import length
from ..structure import Structure
from ._finalize import finalize_geometry, resolve_dd
from .frompath import create_path_reader


def _resolve_xyz_symbol(symbol: str) -> int:
    """
    Resolve one XYZ first-column token to an atomic number the way
    mctc-lib's ``read_xyz`` does: the quirky ``symbol_to_number`` lookup
    (digit-stripping, decorations, D/T) first, falling back to parsing the
    token as a raw atomic number if that fails (mctc-lib's
    ``test_valid5_xyz``, e.g. "8"/"1" instead of "O"/"H").
    """
    number = symbol_to_number(symbol)
    if number is not None:
        return number
    try:
        return int(symbol)
    except ValueError:
        raise FormatErrorXYZ(
            f"Could not map symbol {symbol!r} to an atomic number."
        ) from None


__all__ = ["read_xyz", "read_xyz_qm9"]


def _is_space(char: str) -> bool:
    return char == " " or char == "\t"


def _parse_header_pairs(line: str) -> tuple[dict[str, str], bool]:
    """
    Split an XYZ comment line into whatever ``key -> value`` pairs it
    happens to contain, mirroring mctc-lib's ``get_header_value`` state
    machine: a bare (unquoted) key, then ``=``, then a value that is
    either a quoted string (``"..."``/``'...'``, spaces allowed), a
    bracketed list (``[...]``/``{...}``, matching only its own bracket
    character, quote-aware so a comma inside a quoted element doesn't
    split it), or a bare space-delimited token. A key with no ``=`` (a
    bare flag) is skipped. The first occurrence of a repeated key wins,
    matching mctc-lib's early-return-on-match scan. Unlike mctc-lib, keys
    themselves are assumed unquoted -- realistic Extended XYZ headers
    never quote a key name.

    Returns ``(pairs, fully_parsed)``. The overwhelming majority of
    comment lines are *not* Extended XYZ headers at all (e.g. a plain
    "Energy = -1.234" comment) -- mctc-lib itself only raises a real error
    for malformed header syntax once it has confirmed a ``Properties`` key
    is present (see ``parse_extxyz_header``: any error while scanning for
    ``Properties`` itself is swallowed and treated as "not extended", but
    mctc-lib re-scans the whole line separately for each of ``Lattice``/
    ``pbc``/``comment`` afterwards, so a malformed fragment anywhere *past*
    an already-found ``Properties`` still surfaces as a real error). This
    scanner never raises itself: the first unparseable fragment (an
    unterminated quote/bracket, or ``key=`` with nothing after it) simply
    ends the scan, and ``fully_parsed`` reports whether that happened --
    the caller treats "found Properties, but not fully_parsed" as the
    same real error mctc-lib's own re-scan would hit.
    """
    pairs: dict[str, str] = {}
    n = len(line.rstrip())
    i = 0

    while i < n:
        while i < n and _is_space(line[i]):
            i += 1
        if i >= n:
            # unreachable: `n` excludes trailing whitespace already, so
            # this loop can never run off the end looking for more of it.
            break  # pragma: no cover

        first = i
        while i < n and not (_is_space(line[i]) or line[i] == "="):
            i += 1
        key = line[first:i]
        if not key:
            return pairs, False

        while i < n and _is_space(line[i]):
            i += 1
        if i >= n or line[i] != "=":
            # bare flag with no value -- skip to the next whitespace
            while i < n and not _is_space(line[i]):
                i += 1
            continue
        i += 1
        while i < n and _is_space(line[i]):
            i += 1
        if i >= n:
            return pairs, False

        char = line[i]
        if char in "\"'":
            quote = char
            i += 1
            first = i
            while i < n and line[i] != quote:
                if line[i] == "\\" and i < n - 1:
                    i += 1
                i += 1
            if i >= n:
                return pairs, False
            value = line[first:i]
            i += 1
        elif char in "[{":
            open_char = char
            close_char = "]" if open_char == "[" else "}"
            first = i
            depth = 0
            quote = ""
            while i < n:
                if quote:
                    if line[i] == "\\" and i < n - 1:
                        i += 2
                        continue
                    if line[i] == quote:
                        quote = ""
                elif line[i] in "\"'":
                    quote = line[i]
                elif line[i] == open_char:
                    depth += 1
                elif line[i] == close_char:
                    depth -= 1
                    if depth == 0:
                        break
                i += 1
            if i >= n or depth != 0:
                return pairs, False
            value = line[first : i + 1]
            i += 1
        else:
            first = i
            while i < n and not _is_space(line[i]):
                i += 1
            value = line[first:i]

        pairs.setdefault(key, value)

    return pairs, True


def _parse_properties_spec(
    properties: str, fileobj: IO[Any]
) -> tuple[int, int, int, int]:
    """
    Parse a ``Properties=name:kind:count:...`` specification into 1-indexed
    ``(species_col, z_col, pos_col, ncols)``, mirroring mctc-lib's
    ``parse_properties``: only ``species`` (kind ``S``, count 1), ``Z``
    (kind ``I``, count 1) and ``pos`` (kind ``R``, count 3) are recognised;
    any other name is accepted but its columns are skipped over, not
    stored (matching mctc-lib's ``case default: continue``). At least one
    of ``species``/``Z`` and exactly one ``pos`` are required.
    """
    parts = properties.split(":")
    if not parts or len(parts) % 3 != 0:
        raise FormatErrorXYZ(
            f"Invalid Properties specification in '{fileobj}'."
        )

    species_col = z_col = pos_col = 0
    col = 1
    for i in range(0, len(parts), 3):
        name, kind, count_str = parts[i], parts[i + 1], parts[i + 2]
        try:
            count = int(count_str)
        except ValueError:
            raise FormatErrorXYZ(
                f"Invalid Properties specification in '{fileobj}'."
            ) from None
        if count < 1 or len(kind) != 1:
            raise FormatErrorXYZ(
                f"Invalid Properties specification in '{fileobj}'."
            )

        if name == "species":
            if kind != "S" or count != 1:
                raise FormatErrorXYZ(
                    f"Invalid Properties specification in '{fileobj}'."
                )
            species_col = col
        elif name == "Z":
            if kind != "I" or count != 1:
                raise FormatErrorXYZ(
                    f"Invalid Properties specification in '{fileobj}'."
                )
            z_col = col
        elif name == "pos":
            if kind != "R" or count != 3:
                raise FormatErrorXYZ(
                    f"Invalid Properties specification in '{fileobj}'."
                )
            pos_col = col

        col += count

    ncols = col - 1
    if pos_col == 0 or (species_col == 0 and z_col == 0):
        raise FormatErrorXYZ(
            f"Invalid Properties specification in '{fileobj}': a 'pos' "
            "and either a 'species' or 'Z' entry are required."
        )

    return species_col, z_col, pos_col, ncols


def _strip_list_delims(value: str) -> str:
    for ch in "[]{},":
        value = value.replace(ch, " ")
    return value


def _parse_lattice_value(value: str, dd: DD, fileobj: IO[Any]) -> Tensor:
    """
    Parse a ``Lattice="..."`` value into a ``(3, 3)`` lattice tensor
    (bohr), mirroring mctc-lib's ``parse_lattice``: 9 values fill the
    matrix directly, 3 values fill only the diagonal. Consecutive triples
    in the flat list are whole lattice vectors -- a plain row-major
    ``reshape(3, 3)`` already puts them in this project's "rows are
    lattice vectors" convention, no transpose needed (mctc-lib's own
    ``reshape(vals, [3, 3])`` fills column-major into a columns-are-
    vectors array, which groups the same way).
    """
    try:
        values = [float(v) for v in _strip_list_delims(value).split()]
    except ValueError:
        raise FormatErrorXYZ(
            f"Invalid Lattice specification in '{fileobj}'."
        ) from None

    if len(values) == 9:
        lattice = torch.tensor(values, **dd).reshape(3, 3)
    elif len(values) == 3:
        lattice = torch.diag(torch.tensor(values, **dd))
    else:
        raise FormatErrorXYZ(
            f"Invalid Lattice specification in '{fileobj}': expected 3 or "
            f"9 values, got {len(values)}."
        )

    return lattice * length.AA2AU


def _parse_pbc_value(value: str, fileobj: IO[Any]) -> list[bool]:
    """Parse a ``pbc="..."`` value into 3 booleans, mirroring mctc-lib's
    ``parse_pbc`` (Fortran list-directed logical input: a token starting
    with ``t``/``T`` is true, ``f``/``F`` is false)."""
    tokens = _strip_list_delims(value).split()
    if len(tokens) != 3:
        raise FormatErrorXYZ(
            f"Invalid pbc specification in '{fileobj}': expected 3 "
            f"values, got {len(tokens)}."
        )

    result = []
    for token in tokens:
        stripped = token.lstrip(".")
        if stripped[:1] in ("t", "T"):
            result.append(True)
        elif stripped[:1] in ("f", "F"):
            result.append(False)
        else:
            raise FormatErrorXYZ(f"Invalid pbc value {token!r} in '{fileobj}'.")
    return result


@dataclass
class _ExtxyzHeader:
    species_col: int
    z_col: int
    pos_col: int
    ncols: int
    lattice: Tensor | None
    periodic: list[bool] | None


def _parse_extxyz_header(
    line: str, dd: DD, fileobj: IO[Any]
) -> _ExtxyzHeader | None:
    """
    Parse an XYZ comment line as an Extended XYZ header. Returns ``None``
    if the line has no ``Properties`` key -- mctc-lib switches to Extended
    XYZ mode on ``Properties`` alone, so e.g. a ``Lattice=`` given without
    ``Properties`` is not Extended XYZ and its lattice is never even
    looked at (matching mctc-lib's ``parse_extxyz_header``, which returns
    immediately if ``Properties`` is absent).
    """
    pairs, fully_parsed = _parse_header_pairs(line)
    if "Properties" not in pairs:
        return None
    if not fully_parsed:
        raise FormatErrorXYZ(
            f"Could not parse Extended XYZ header in '{fileobj}'."
        )

    species_col, z_col, pos_col, ncols = _parse_properties_spec(
        pairs["Properties"], fileobj
    )

    lattice = None
    periodic = None
    if "Lattice" in pairs:
        lattice = _parse_lattice_value(pairs["Lattice"], dd, fileobj)
        periodic = [True, True, True]
    if "pbc" in pairs:
        periodic = _parse_pbc_value(pairs["pbc"], fileobj)

    # mctc-lib keeps such a mask with a zero cell, but a periodic axis
    # without its lattice vector cannot be evaluated. An all-false `pbc`
    # (as ASE writes for a molecule) needs no lattice.
    if lattice is None and periodic is not None and any(periodic):
        raise FormatErrorXYZ(
            f"'{fileobj}' marks periodic axes with 'pbc' but gives no "
            "'Lattice' for them."
        )

    return _ExtxyzHeader(
        species_col=species_col,
        z_col=z_col,
        pos_col=pos_col,
        ncols=ncols,
        lattice=lattice,
        periodic=periodic,
    )


def _parse_extxyz_atom(
    line: str, header: _ExtxyzHeader, fileobj: IO[Any]
) -> tuple[int, float, float, float]:
    """Parse one Extended XYZ atom line according to the column layout
    from :func:`_parse_properties_spec`, mirroring mctc-lib's
    ``read_extxyz_atom``: exactly ``header.ncols`` whitespace-separated
    tokens must be present, no more, no fewer."""
    tokens = line.split()
    if len(tokens) != header.ncols:
        raise FormatErrorXYZ(
            f"Could not parse atom data from Extended XYZ file "
            f"'{fileobj}': expected {header.ncols} columns, got "
            f"{len(tokens)}."
        )

    symbol: str | None = None
    atomic_number: int | None = None
    x = y = z = 0.0
    for col, token in enumerate(tokens, start=1):
        if col == header.species_col:
            symbol = token
        elif col == header.z_col:
            try:
                atomic_number = int(token)
            except ValueError:
                raise FormatErrorXYZ(
                    f"Could not parse atom data from Extended XYZ file "
                    f"'{fileobj}': expected an integer 'Z', got {token!r}."
                ) from None
        elif header.pos_col <= col < header.pos_col + 3:
            try:
                value = float(token)
            except ValueError:
                raise FormatErrorXYZ(
                    f"Could not parse atom data from Extended XYZ file "
                    f"'{fileobj}': expected a real value, got {token!r}."
                ) from None
            offset = col - header.pos_col
            if offset == 0:
                x = value
            elif offset == 1:
                y = value
            else:
                z = value

    if header.species_col:
        assert symbol is not None
        number = symbol_to_number(symbol)
        if number is None:
            raise FormatErrorXYZ(
                f"Cannot map symbol {symbol!r} to an atomic number in "
                f"'{fileobj}'."
            )
    else:
        if atomic_number is None or atomic_number <= 0:
            raise FormatErrorXYZ(
                f"Cannot map atomic number to a symbol in '{fileobj}'."
            )
        number = atomic_number

    return number, x, y, z


def _parse_atom_block(
    fileobj: IO[Any], natoms: int
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """
    Parse ``natoms`` consecutive ``symbol x y z ...`` lines from ``fileobj``
    into atomic numbers and an ``(natoms, 3)`` float64 coordinate array.

    Reads the whole block as one string and hands it to ``numpy.loadtxt``
    instead of looping over ``fileobj.readline()``/``str.split()``/
    ``float()``/``dict[symbol]`` once per atom: profiling a 852k-atom file
    showed those per-atom calls dominating read time (readline 851971x,
    split 851971x, 3x float() each), all of it single-element Python-level
    overhead that ``numpy.loadtxt``'s C parser and a symbol->number lookup
    vectorised over only the *distinct* symbols amortise across the whole
    block instead. Only the first four whitespace-separated columns are
    used (``usecols``), matching the old loop's ``line[:4]``: a trailing
    column (e.g. a comment or velocity in an extended xyz variant) is
    ignored, not an error.
    """
    lines = itertools.islice(fileobj, natoms)
    block = "".join(lines)

    if natoms == 0 or not block.strip():
        return np.empty(0, dtype=np.int64), np.empty((0, 3), dtype=np.float64)

    data = np.loadtxt(
        _io.StringIO(block),
        dtype={
            "names": ("symbol", "x", "y", "z"),
            "formats": ("U8", "f8", "f8", "f8"),
        },
        usecols=(0, 1, 2, 3),
        ndmin=1,
    )

    coords = np.stack([data["x"], data["y"], data["z"]], axis=-1)

    # The symbol resolution only runs over the (usually far smaller) set
    # of distinct symbols, not once per atom.
    unique_symbols, inverse = np.unique(data["symbol"], return_inverse=True)
    z_per_unique = np.array(
        [_resolve_xyz_symbol(str(s)) for s in unique_symbols],
        dtype=np.int64,
    )
    numbers = z_per_unique[inverse]

    return numbers, coords


def read_xyz_fileobj(
    fileobj: IO[Any],
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    dtype_int: torch.dtype = torch.long,
    **kwargs: Any,
) -> Structure:
    """
    Reads an XYZ file into a structure. Multiple frames are batched
    together. If the file has exactly one frame and its comment line is an
    Extended XYZ header declaring periodicity (a ``Lattice`` and/or ``pbc``
    key), the structure also carries a lattice and periodicity mask (see
    the module docstring). Positions are converted to atomic units (bohrs).

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
    Structure
        (Possibly batched) atomic numbers and positions (shape
        ``(batch_size, nat, 3)`` for several frames, bohr). A single-frame
        Extended XYZ file declaring periodicity also carries the lattice
        (bohr) and the periodicity mask.

    Raises
    ------
    FormatErrorXYZ
        The file does not conform with the expected format, a ``pbc``
        marks a periodic axis without a ``Lattice``, or more than one
        frame declares periodicity (unsupported: mctc-lib itself only
        ever handles a single frame, so there is no established
        multi-frame lattice/periodic convention to follow here).
    """
    natoms_line: str

    dd, ddi = resolve_dd(device, dtype, dtype_int)

    numbers_images: list[Tensor] = []
    positions_images: list[Tensor] = []
    frame_lattice: Tensor | None = None
    frame_periodic: Tensor | None = None
    n_periodic_frames = 0

    while True:
        # Stripping here also covers additional trailing new lines; otherwise,
        # a blank line would be interpreted as the start of a new image. This
        # is only a legitimate end-of-file marker once at least one image was
        # already read -- a blank line before any image means the file never
        # contained a usable first frame (covers both a genuinely empty file
        # and one whose very first, non-comment line is blank).
        natoms_line = fileobj.readline().strip()
        if not natoms_line:
            if not numbers_images:
                raise EmptyFileError(f"No valid data found in '{fileobj}'.")
            break

        # Check if line is a number (must strip newline character before)
        if not natoms_line.isdigit():
            raise FormatErrorXYZ(
                "The first line in an xyz file should be the number of atoms "
                f"in the structure, but is {repr(natoms_line)}."
            )
        natoms = int(natoms_line)

        comment_line = fileobj.readline()
        header = _parse_extxyz_header(comment_line, dd, fileobj)

        if header is None:
            numbers_np, coords = _parse_atom_block(fileobj, natoms)
        else:
            numbers_list: list[int] = []
            coords_list: list[list[float]] = []
            for _ in range(natoms):
                atom_line = fileobj.readline()
                if not atom_line:
                    raise FormatErrorXYZ(
                        f"Could not read geometry from xyz file '{fileobj}'."
                    )
                number, x, y, z = _parse_extxyz_atom(atom_line, header, fileobj)
                numbers_list.append(number)
                coords_list.append([x, y, z])
            numbers_np = np.array(numbers_list, dtype=np.int64)
            coords = np.array(coords_list, dtype=np.float64)

            if header.periodic is not None and any(header.periodic):
                n_periodic_frames += 1
                # the header parser only allows a periodic axis with a lattice
                frame_lattice = header.lattice
                frame_periodic = torch.tensor(
                    header.periodic, dtype=torch.bool, device=device
                )

        if numbers_np.shape[0] != natoms:
            raise FormatErrorXYZ(
                f"Number of atoms in the geometry block ({numbers_np.shape[0]}) "
                f"does not match the declared number of atoms ({natoms})."
            )

        numbers = torch.tensor(numbers_np, **ddi)
        positions = torch.tensor(coords, **dd) * length.AA2AU

        positions = finalize_geometry(numbers, positions, fileobj, **kwargs)

        numbers_images.append(numbers)
        positions_images.append(positions)

    if n_periodic_frames > 0 and len(numbers_images) > 1:
        raise FormatErrorXYZ(
            f"'{fileobj}' has {len(numbers_images)} frames and at least "
            "one declares Extended XYZ periodicity; periodic multi-frame "
            "XYZ files are not supported."
        )

    # if only one image, return its tensors directly
    if len(numbers_images) == 1:
        n = numbers_images[0]
        p = positions_images[0]

        if n_periodic_frames == 1:
            return Structure(
                numbers=n,
                positions=p,
                lattice=frame_lattice,
                periodic=frame_periodic,
            )

        if kwargs.get("batch_agnostic", False):
            return Structure(numbers=n.unsqueeze(0), positions=p.unsqueeze(0))

        return Structure(numbers=n, positions=p)

    return Structure(
        numbers=pack(numbers_images), positions=pack(positions_images)
    )


read_xyz = create_path_reader(read_xyz_fileobj)


def read_xyz_qm9_fileobj(
    fileobj: IO[Any],
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    dtype_int: torch.dtype = torch.long,
    **kwargs: Any,
) -> Structure:
    """
    Reads the XYZ file of the QM9 dataset, which does not conform with the
    standard format, and returns atomic numbers and positions as tensors.
    Handles only a single structure.
    Positions are converted to atomic units (bohrs).

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
    Structure
        Atomic numbers and positions (shape ``(nat, 3)``, bohr).
    """
    line: list[str]
    natoms_line: str

    dd, ddi = resolve_dd(device, dtype, dtype_int)

    natoms_line = fileobj.readline()
    natoms = int(natoms_line.strip())

    # Skip comment line
    fileobj.readline()

    symbols = []
    coords = []
    for _ in range(natoms):
        line = fileobj.readline().split()
        symbols.append(line[0])
        coords.append([float(x.replace("*^", "e")) for x in line[1:4]])

    numbers = torch.tensor(
        [_resolve_xyz_symbol(symbol) for symbol in symbols], **ddi
    )
    positions = torch.tensor(coords, **dd) * length.AA2AU

    positions = finalize_geometry(numbers, positions, fileobj, **kwargs)

    return Structure(numbers=numbers, positions=positions)


read_xyz_qm9 = create_path_reader(read_xyz_qm9_fileobj)
