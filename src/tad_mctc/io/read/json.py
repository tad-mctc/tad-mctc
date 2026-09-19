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
I/O Read: General JSON
=======================

Sniff-and-dispatch reader for a generic ``.json`` file. Mirrors
mctc-lib's ``mctc_io_read_json`` (``src/mctc/io/read/json.F90``): unlike
every other reader here, this isn't a distinct format with its own
grammar -- it parses the JSON once, inspects a handful of top-level keys,
and routes to whichever of the three JSON-based readers (qcschema,
pymatgen, cjson) actually matches, in mctc-lib's own precedence order:

- ``schema_name``/``schema_version`` present -> qcschema
- ``@module``/``@class`` present -> pymatgen
- ``chemicalJson``/``chemical json`` present -> cjson
- otherwise -> qcschema, mctc-lib's own default

The first two branches both resolve to qcschema, but the explicit check
still matters: it is what decides the outcome for a (pathological) object
that carries identifying keys for more than one schema, e.g. both
``schema_version`` and ``@module``.

Only the *presence* of these keys is checked here; each delegate then
validates its own schema in full (so, e.g., a file with ``@module`` but
the wrong value still raises pymatgen's own error, not a generic one).
The already-parsed JSON object is passed directly to each delegate's
``*_from_dict`` helper so the same text is never parsed twice.
"""

from __future__ import annotations

import json
from typing import IO, Any

import torch

from ...typing import DD, get_default_dtype
from .cjson import read_cjson_from_dict
from .frompath import JSONResult, create_path_reader_json
from .pymatgen import read_pymatgen_from_dict
from .qcschema import read_qcschema_from_dict

__all__ = ["read_json"]


def read_json_fileobj(
    fileobj: IO[Any],
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    dtype_int: torch.dtype = torch.long,
    **kwargs: Any,
) -> JSONResult:
    """
    Reads a generic JSON file, sniffs which of the qcschema/pymatgen/
    cjson schemas it matches, and delegates to that reader. See the
    module docstring for the exact sniffing rule.

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
    (Tensor, Tensor) | (Tensor, Tensor, Tensor, Tensor) | CJSONResult
        Whichever shape the matched delegate returns -- see
        :func:`tad_mctc.io.read.qcschema.read_qcschema_fileobj`,
        :func:`tad_mctc.io.read.pymatgen.read_pymatgen_fileobj`, or
        :func:`tad_mctc.io.read.cjson.read_cjson_fileobj`.

    Raises
    ------
    ValueError
        The file is not valid JSON (``json.JSONDecodeError``, a
        ``ValueError`` subclass).
    """
    dd: DD = {
        "device": device,
        "dtype": dtype if dtype is not None else get_default_dtype(),
    }
    ddi: DD = {"device": device, "dtype": dtype_int}

    data = json.load(fileobj)

    # QCSchema JSON uses "schema_name" and "schema_version" keys -- checked
    # first to match mctc-lib's own precedence (json.F90) for the case
    # where an object carries keys identifying more than one schema
    if isinstance(data, dict) and (
        "schema_name" in data or "schema_version" in data
    ):
        return read_qcschema_from_dict(data, fileobj, dd, ddi, **kwargs)

    if isinstance(data, dict) and ("@module" in data or "@class" in data):
        return read_pymatgen_from_dict(
            data, fileobj, dd, ddi, device=device, **kwargs
        )
    if isinstance(data, dict) and (
        "chemicalJson" in data or "chemical json" in data
    ):
        return read_cjson_from_dict(
            data, fileobj, dd, ddi, device=device, **kwargs
        )

    return read_qcschema_from_dict(data, fileobj, dd, ddi, **kwargs)


read_json = create_path_reader_json(read_json_fileobj)
