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
Module containing the version string.
"""
from __future__ import annotations

import torch


def version_tuple(version_string: str) -> tuple[int, ...]:
    """
    Convert a version string (with possible additional version specifications)
    to a tuple following semantic versioning.

    Parameters
    ----------
    version_string : str
        Version string to convert.

    Returns
    -------
    tuple[int, ...]
        Semantic version number as tuple.
    """
    main_version_part = version_string.split("-")[0].split("+")[0].split("_")[0]

    s = main_version_part.split(".")
    if 3 > len(s):
        raise RuntimeError(
            "Version specification does not seem to follow the semantic "
            f"versioning scheme of MAJOR.MINOR.PATCH ({s})."
        )

    version_numbers = [int(part) for part in s[:3]]
    return tuple(version_numbers)


__version__ = "0.0.6"
__tversion__ = version_tuple(torch.__version__)
