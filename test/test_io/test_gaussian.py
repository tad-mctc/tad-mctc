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
Test the Gaussian external (.ein) file reader, mirroring mctc-lib's
``mctc_io_read_gaussian`` test cases. Coordinates in this format are
already given in bohr (no Angstrom conversion applies), and the header's
embedded charge/spin are parsed (so a malformed header still raises) but
not propagated -- unlike every other geometry field, this repository
threads charge/uhf only via the ``.CHRG``/``.UHF`` sidecar mechanism.
"""

import tempfile
from pathlib import Path

import pytest
import torch

from tad_mctc.exceptions import FormatErrorGaussian
from tad_mctc.io import read
from tad_mctc.typing import DD


def _write(content: str) -> tuple[tempfile.TemporaryDirectory[str], Path]:
    tmpdir = tempfile.TemporaryDirectory()
    filepath = Path(tmpdir.name) / "mol.ein"
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    return tmpdir, filepath


_VALID1 = (
    "         4         1         0         1\n"
    "         7      0.000000000000      0.000000000000     -0.114091591161      0.000000000000 \n"
    "         1     -1.817280998039      0.000000000000      0.528409372569      0.000000000000 \n"
    "         1      0.908640499019     -1.573811509290      0.528409372569      0.000000000000 \n"
    "         1      0.908640499019      1.573811509290      0.528409372569      0.000000000000 \n"
    " 1 2 1.000 3 1.000 4 1.000\n"
    " 2 1 1.000\n"
    " 3 1 1.000\n"
    " 4 1 1.000\n"
)


def test_read_valid() -> None:
    """mctc-lib's own ``valid1-ein`` fixture (ammonia); coordinates are
    already in bohr, and the trailing bond-connectivity lines are ignored."""
    dd: DD = {"device": None, "dtype": torch.double}

    tmpdir, filepath = _write(_VALID1)
    with tmpdir:
        numbers, positions = read.read_gaussian(filepath, **dd)

    ref_numbers = torch.tensor([7, 1, 1, 1])
    ref_positions = torch.tensor(
        [
            [0.000000000000, 0.000000000000, -0.114091591161],
            [-1.817280998039, 0.000000000000, 0.528409372569],
            [0.908640499019, -1.573811509290, 0.528409372569],
            [0.908640499019, 1.573811509290, 0.528409372569],
        ],
        **dd,
    )

    assert (ref_numbers == numbers).all()
    assert pytest.approx(ref_positions.cpu()) == positions.cpu()


def test_read_fail_notfound() -> None:
    with pytest.raises(FileNotFoundError):
        read.read_gaussian("not found")


@pytest.mark.parametrize(
    "content",
    [
        # non-numeric header fields (charge/spin columns)
        (
            "         4         1      zero       one\n"
            "         7      0.000000000000      0.000000000000     -0.114091591161      0.000000000000 \n"
            "         1     -1.817280998039      0.000000000000      0.528409372569      0.000000000000 \n"
            "         1      0.908640499019     -1.573811509290      0.528409372569      0.000000000000 \n"
            "         1      0.908640499019      1.573811509290      0.528409372569      0.000000000000 \n"
        ),
        # negative/zero atom count
        (
            "        -4         1         0         1\n"
            "         7      0.000000000000      0.000000000000     -0.114091591161      0.000000000000 \n"
        ),
        # malformed coordinate value
        (
            "         4         1         0         1\n"
            "         7      0.000000000000      0.000000000000     -0.114091591161      0.000000000000 \n"
            "         1     -1.817280998039      0.000000000000      0.528409372569      0.000000000000 \n"
            "         1      0.908640499019     -1.573811509290      0.528409372569      0.000000000000 \n"
            "         1      abcd.efgh-jklm      1.573811509290      0.528409372569      0.000000000000 \n"
        ),
        # invalid (non-positive) atomic number
        (
            "         4         1         0         1\n"
            "         7      0.000000000000      0.000000000000     -0.114091591161      0.000000000000 \n"
            "         1     -1.817280998039      0.000000000000      0.528409372569      0.000000000000 \n"
            "         1      0.908640499019     -1.573811509290      0.528409372569      0.000000000000 \n"
            "        -1      0.908640499019      1.573811509290      0.528409372569      0.000000000000 \n"
        ),
        # fewer atom lines than the header's atom count declares
        (
            "         4         1         0         1\n"
            "         7      0.000000000000      0.000000000000     -0.114091591161      0.000000000000 \n"
            "         1     -1.817280998039      0.000000000000      0.528409372569      0.000000000000 \n"
        ),
    ],
    ids=[
        "bad-header",
        "non-positive-atom-count",
        "malformed-coordinate",
        "invalid-atomic-number",
        "atom-count-mismatch",
    ],
)
def test_read_fail_format(content: str) -> None:
    tmpdir, filepath = _write(content)
    with tmpdir:
        with pytest.raises(FormatErrorGaussian):
            read.read_gaussian(filepath)
