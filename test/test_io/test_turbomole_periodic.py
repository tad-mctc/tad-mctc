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
Test periodic (``$periodic``/``$lattice``/``$cell``) support in the
Turbomole coord reader, mirroring mctc-lib's ``mctc_io_read_turbomole``.
"""

import math
import tempfile
from pathlib import Path

import pytest
import torch

from tad_mctc.exceptions import FormatErrorTM
from tad_mctc.io import read
from tad_mctc.typing import DD
from tad_mctc.units import length

from ..conftest import DEVICE


def _write(content: str) -> tuple[tempfile.TemporaryDirectory[str], Path]:
    tmpdir = tempfile.TemporaryDirectory()
    filepath = Path(tmpdir.name) / "coord"
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    return tmpdir, filepath


def _cell_to_lattice(
    a: float, b: float, c: float, alpha: float, beta: float, gamma: float
) -> torch.Tensor:
    """Independent re-implementation of mctc-lib's ``cell_to_dlat`` (angles
    in degrees) used only to build reference values for the tests."""
    alp, bet, gam = math.radians(alpha), math.radians(beta), math.radians(gamma)
    vol2 = (
        1.0
        - math.cos(alp) ** 2
        - math.cos(bet) ** 2
        - math.cos(gam) ** 2
        + 2.0 * math.cos(alp) * math.cos(bet) * math.cos(gam)
    )
    dvol = math.sqrt(abs(vol2)) * a * b * c
    if vol2 < 0.0:
        dvol = -dvol

    v1 = [a, 0.0, 0.0]
    v2 = [b * math.cos(gam), b * math.sin(gam), 0.0]
    v3 = [
        c * math.cos(bet),
        c * (math.cos(alp) - math.cos(bet) * math.cos(gam)) / math.sin(gam),
        dvol / (a * b * math.sin(gam)),
    ]
    return torch.tensor([v1, v2, v3])


################################################################################
# non-periodic regression: unchanged 2-tuple return
################################################################################


def test_read_non_periodic_still_2_tuple() -> None:
    content = "$coord\n0.0 0.0 0.0 h\n$end\n"
    tmpdir, filepath = _write(content)
    with tmpdir:
        result = read.read_turbomole(filepath)

    assert len(result) == 2


def test_read_explicit_periodic_zero_still_2_tuple() -> None:
    content = "$coord\n0.0 0.0 0.0 h\n$periodic 0\n$end\n"
    tmpdir, filepath = _write(content)
    with tmpdir:
        result = read.read_turbomole(filepath)

    assert len(result) == 2


################################################################################
# $cell (cell parameters, requiring the trig conversion)
################################################################################


def test_read_periodic_cell_triclinic_frac() -> None:
    """Rock-salt MgO primitive cell (mctc-lib's own ``test_coord`` fixture):
    fractional coordinates, a rhombohedral cell given via ``$cell`` with no
    unit modifier (bohr by default)."""
    dd: DD = {"device": None, "dtype": torch.double}

    content = (
        "$coord frac\n"
        "    0.00000000000000      0.00000000000000      0.00000000000000      mg\n"
        "    0.50000000000000      0.50000000000000      0.50000000000000      o\n"
        "$periodic 3\n"
        "$cell\n"
        " 5.798338236 5.798338236 5.798338236 60. 60. 60.\n"
        "$end\n"
    )
    tmpdir, filepath = _write(content)
    with tmpdir:
        numbers, positions, lattice, periodic = read.read_turbomole(  # type: ignore[misc]
            filepath, **dd
        )

    ref_numbers = torch.tensor([12, 8])
    ref_lattice = _cell_to_lattice(
        5.798338236, 5.798338236, 5.798338236, 60.0, 60.0, 60.0
    ).to(**dd)
    frac = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], **dd)
    ref_positions = frac @ ref_lattice

    assert (ref_numbers == numbers).all()
    assert (periodic == torch.tensor([True, True, True])).all()
    assert pytest.approx(ref_lattice.cpu()) == lattice.cpu()
    assert pytest.approx(ref_positions.cpu()) == positions.cpu()


def test_read_periodic_cell_angs() -> None:
    dd: DD = {"device": None, "dtype": torch.double}

    content = (
        "$coord\n"
        "0.0 0.0 0.0 si\n"
        "$periodic 3\n"
        "$cell angs\n"
        " 5.43 5.43 5.43 90. 90. 90.\n"
        "$end\n"
    )
    tmpdir, filepath = _write(content)
    with tmpdir:
        numbers, positions, lattice, periodic = read.read_turbomole(  # type: ignore[misc]
            filepath, **dd
        )

    ref_lattice = (
        _cell_to_lattice(5.43, 5.43, 5.43, 90.0, 90.0, 90.0).to(**dd)
        * length.AA2AU
    )

    assert (numbers == torch.tensor([14])).all()
    assert (periodic == torch.tensor([True, True, True])).all()
    assert pytest.approx(ref_lattice.cpu()) == lattice.cpu()
    # cartesian coordinate at the origin stays at the origin regardless of
    # the cell -- only fractional coordinates go through the lattice
    assert pytest.approx(torch.zeros(1, 3, **dd).cpu()) == positions.cpu()


def test_read_periodic_cell_2d_slab() -> None:
    """``$periodic 2``: a and b lengths plus the gamma angle only."""
    dd: DD = {"device": None, "dtype": torch.double}

    content = (
        "$coord frac\n"
        "0.0 0.0 0.0 c\n"
        "$periodic 2\n"
        "$cell\n"
        " 4.0 4.0 90.\n"
        "$end\n"
    )
    tmpdir, filepath = _write(content)
    with tmpdir:
        numbers, positions, lattice, periodic = read.read_turbomole(  # type: ignore[misc]
            filepath, **dd
        )

    ref_lattice = _cell_to_lattice(4.0, 4.0, 1.0, 90.0, 90.0, 90.0).to(**dd)

    assert (numbers == torch.tensor([6])).all()
    assert (periodic == torch.tensor([True, True, False])).all()
    assert pytest.approx(ref_lattice.cpu()) == lattice.cpu()


def test_read_periodic_cell_1d_wire() -> None:
    """``$periodic 1``: a single length only."""
    dd: DD = {"device": None, "dtype": torch.double}

    content = (
        "$coord frac\n"
        "0.0 0.0 0.0 c\n"
        "$periodic 1\n"
        "$cell\n"
        " 3.0\n"
        "$end\n"
    )
    tmpdir, filepath = _write(content)
    with tmpdir:
        numbers, positions, lattice, periodic = read.read_turbomole(  # type: ignore[misc]
            filepath, **dd
        )

    ref_lattice = _cell_to_lattice(3.0, 1.0, 1.0, 90.0, 90.0, 90.0).to(**dd)

    assert (numbers == torch.tensor([6])).all()
    assert (periodic == torch.tensor([True, False, False])).all()
    assert pytest.approx(ref_lattice.cpu()) == lattice.cpu()


def test_read_valid2_cell_1d_bohr_real() -> None:
    """mctc-lib's own ``valid2-coord`` fixture: a real 32-atom carbon
    system, ``$cell`` (no unit suffix, i.e. bohr) for ``$periodic 1``, with
    an unknown ``$eht`` group interspersed between ``$periodic`` and
    ``$cell`` that must be skipped."""
    content = (
        "$coord\n"
        "    1.36794785746435     13.45808943446053      8.83754983226359      c\n"
        "    3.69183290816438     13.13552229161569     10.16652201690950      c\n"
        "    1.36792668081267     10.38660504434782     13.04411926632965      c\n"
        "    3.69180534781206     11.55414582295511     12.33193380846742      c\n"
        "    1.36791549262702      3.53066844289674     10.38660588677206      c\n"
        "    1.36792046664920      7.73723910626293     13.45809224934817      c\n"
        "    3.69181279359489      6.40826717723392     13.13552570942280      c\n"
        "    1.36792009865062      3.11669712338516      7.73723850632628      c\n"
        "    3.69181515738094      3.43926499914873      6.40826580885474      c\n"
        "    3.69178443989294      4.24285720771059     11.55415026712869      c\n"
        "    1.36790824853106      6.18818490375705      3.53066863732142      c\n"
        "    3.69178194163078      5.02063901427657      4.24285736953327      c\n"
        "    1.36794124909207     13.04411858182861      6.18818324080182      c\n"
        "    1.36792249732236      8.83755133592807      3.11669686076913      c\n"
        "    3.69182456413952     10.16652118921143      3.43926084011816      c\n"
        "    3.69181444966104     12.33193631088573      5.02063847821044      c\n"
        "    6.01572566324028     13.45790756713123      8.83752222635545      c\n"
        "    8.33965926123256     13.13576644753615     10.16660228658307      c\n"
        "    6.01574747573805     10.38654070512969     13.04391961251944      c\n"
        "    8.33964066450677     11.55427002850905     12.33211653730939      c\n"
        "    6.01574728097580      3.53087013230607     10.38654217813321      c\n"
        "    6.01568913853645      7.73726406411719     13.45790864082374      c\n"
        "    8.33963586549168      6.40818371470975     13.13576911116618      c\n"
        "    6.01568179676984      3.11688332536281      7.73726611148835      c\n"
        "    8.33963704688671      3.43902559351770      6.40818390180453      c\n"
        "    8.33962496288127      4.24267007149867     11.55427031066552      c\n"
        "    6.01573464280675      6.18824653544318      3.53086861480278      c\n"
        "    8.33961857277245      5.02052001792996      4.24267413625204      c\n"
        "    6.01575677304189     13.04392044501564      6.18824448603611      c\n"
        "    6.01568344836224      8.83752193432504      3.11688171781516      c\n"
        "    8.33964228963694     10.16660428027860      3.43902155668011      c\n"
        "    8.33965118613331     12.33211762632282      5.02051902430387      c\n"
        "$periodic 1\n"
        "$eht charge=0 unpaired=0\n"
        "$cell\n"
        " 9.29556285275863798006\n"
        "$end\n"
    )
    tmpdir, filepath = _write(content)
    with tmpdir:
        numbers, positions, _, periodic = read.read_turbomole(  # type: ignore[misc]
            filepath
        )

    assert positions.shape[0] == 32
    assert len(torch.unique(numbers)) == 1
    assert (periodic == torch.tensor([True, False, False])).all()


def test_read_valid3_cell_2d_angs() -> None:
    """mctc-lib's own ``valid3-coord`` fixture: ``$cell angs`` for
    ``$periodic 2`` (a, b, gamma), with ``$eht`` skipped between ``$coord``
    and ``$periodic``."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    content = (
        "$coord\n"
        "   -0.12918412100093      0.06210659750976     -2.13384498734326  c\n"
        "    0.12856915667443     -0.07403227791901      4.02358027265954  c\n"
        "   -0.12317720857511      2.75170732207802     -2.13345350602279  c\n"
        "    2.44816466162280      1.28612566399214      4.02317048854901  c\n"
        "$eht unpaired=0 charge=0\n"
        "$periodic 2\n"
        "$cell  angs\n"
        "    2.4809835980     2.4811430162   120.2612191150\n"
        "$end\n"
    )
    tmpdir, filepath = _write(content)
    with tmpdir:
        numbers, _, lattice, periodic = read.read_turbomole(  # type: ignore[misc]
            filepath, **dd
        )

    # mctc-lib's turbomole.f90 (periodic == 2 branch) only multiplies the
    # two *read* lengths (latvec(1)*conv, latvec(2)*conv) by the angstrom
    # conversion factor; the dummy third length used to fill out the 3x3
    # cell is left as the literal 1.0_wp, unconverted
    ref_lattice = _cell_to_lattice(
        2.4809835980 * length.AA2AU,
        2.4811430162 * length.AA2AU,
        1.0,
        90.0,
        90.0,
        120.2612191150,
    ).to(**dd)

    assert numbers.shape[0] == 4
    assert len(torch.unique(numbers)) == 1
    assert (periodic == torch.tensor([True, True, False])).all()
    assert pytest.approx(ref_lattice.cpu()) == lattice.cpu()


def test_read_valid4_cell_before_coord() -> None:
    """mctc-lib's own ``valid4-coord`` fixture: ``$cell`` given *before*
    ``$coord`` (order independence), full 6-parameter rhombohedral cell for
    ``$periodic 3``."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    content = (
        "$cell\n"
        "  4.766080896955 4.766080896955 4.766080896955 60. 60. 60.\n"
        "$coord\n"
        "    0.00000000000000      0.00000000000000      0.00000000000000      c\n"
        "    2.38304045219106      1.39084904447079      0.97287218605834      c\n"
        "$periodic 3\n"
        "$end\n"
    )
    tmpdir, filepath = _write(content)
    with tmpdir:
        numbers, _, lattice, periodic = read.read_turbomole(  # type: ignore[misc]
            filepath, **dd
        )

    ref_lattice = _cell_to_lattice(
        4.766080896955, 4.766080896955, 4.766080896955, 60.0, 60.0, 60.0
    ).to(**dd)

    assert numbers.shape[0] == 2
    assert len(torch.unique(numbers)) == 1
    assert (periodic == torch.tensor([True, True, True])).all()
    assert pytest.approx(ref_lattice.cpu()) == lattice.cpu()


def test_read_valid6_cell_hexagonal() -> None:
    """mctc-lib's own ``valid6-coord`` fixture: a hexagonal cell
    (gamma=120) given before ``$coord``, with four distinct elements."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    content = (
        "$cell\n"
        " 9.09903133 9.09903130512 30.4604956 90.0 90.0 120.000000127\n"
        "$coord\n"
        "   -0.57949455800000      0.06835893310000     -7.51993484000000      ca\n"
        "   -0.57949455800000      0.06835893310000      7.71031294000000      mg\n"
        "   -0.57949455800000      0.06835893310000     -0.10280417200000      c\n"
        "    1.73848367000000     -0.20507679900000     -0.08757392470000      o\n"
        "$periodic 3\n"
        "$end\n"
    )
    tmpdir, filepath = _write(content)
    with tmpdir:
        numbers, _, lattice, periodic = read.read_turbomole(  # type: ignore[misc]
            filepath, **dd
        )

    ref_lattice = _cell_to_lattice(
        9.09903133, 9.09903130512, 30.4604956, 90.0, 90.0, 120.000000127
    ).to(**dd)

    assert numbers.shape[0] == 4
    assert len(torch.unique(numbers)) == 4
    assert (periodic == torch.tensor([True, True, True])).all()
    assert pytest.approx(ref_lattice.cpu()) == lattice.cpu()


def test_read_valid10_cell_before_periodic_before_coord() -> None:
    """mctc-lib's own ``valid10-coord`` fixture: ``$cell`` before
    ``$periodic`` before ``$coord``, plus a trailing ``$user-defined bonds``
    group that must be skipped."""
    content = (
        "$cell\n"
        " 8.00000006 \n"
        "$periodic 1\n"
        "$coord\n"
        "   -2.00000001000000      3.50586945000000      0.00000000000000      b\n"
        "   -2.00000001000000      2.98408124000000      2.61870552000000      n\n"
        "   -2.00000001000000      1.49889804000000     -1.76123460000000      n\n"
        "   -2.00000001000000      5.56753332000000     -0.69908457400000      h\n"
        "   -2.00000001000000      0.45532164600000      3.47617645000000      b\n"
        "   -2.00000001000000     -1.02986157000000     -0.90376369200000      b\n"
        "   -2.00000001000000     -1.55164977000000      1.71494186000000      n\n"
        "   -2.00000001000000      0.02991464770000      5.61117202000000      h\n"
        "   -2.00000001000000     -2.66611845000000     -2.33967477000000      h\n"
        "   -2.00000001000000      4.53085539000000      3.97609015000000      h\n"
        "   -2.00000001000000     -3.50056641000000      2.37579525000000      h\n"
        "   -2.00000001000000      1.90104056000000     -3.77947261000000      h\n"
        "    2.00000001000000      1.55164977000000     -1.71494183000000      b\n"
        "    2.00000001000000      1.02986156000000      0.90376368700000      n\n"
        "    2.00000001000000     -0.45532163900000     -3.47617644000000      n\n"
        "    2.00000001000000      3.61331364000000     -2.41402641000000      h\n"
        "    2.00000001000000     -1.49889804000000      1.76123461000000      b\n"
        "    2.00000001000000     -2.98408125000000     -2.61870553000000      b\n"
        "    2.00000001000000     -3.50586946000000      0.00000002473480      n\n"
        "    2.00000001000000     -1.92430504000000      3.89623019000000      h\n"
        "    2.00000001000000     -4.62033813000000     -4.05461660000000      h\n"
        "    2.00000001000000      2.57663570000000      2.26114832000000      h\n"
        "    2.00000001000000     -5.45478609000000      0.66085341700000      h\n"
        "    2.00000001000000     -0.05317912580000     -5.49441445000000      h\n"
        "$user-defined bonds\n"
        "$end\n"
    )
    tmpdir, filepath = _write(content)
    with tmpdir:
        numbers, _, _, periodic = read.read_turbomole(filepath)  # type: ignore[misc]

    assert numbers.shape[0] == 24
    assert len(torch.unique(numbers)) == 3
    assert int(periodic.sum()) == 1


################################################################################
# $lattice (explicit vectors)
################################################################################


def test_read_periodic_lattice_cartesian() -> None:
    dd: DD = {"device": None, "dtype": torch.double}

    content = (
        "$coord\n"
        "0.0 0.0 0.0 c\n"
        "$periodic 3\n"
        "$lattice angs\n"
        "3.57 0.0 0.0\n"
        "0.0 3.57 0.0\n"
        "0.0 0.0 3.57\n"
        "$end\n"
    )
    tmpdir, filepath = _write(content)
    with tmpdir:
        numbers, positions, lattice, periodic = read.read_turbomole(  # type: ignore[misc]
            filepath, **dd
        )

    ref_lattice = torch.eye(3, **dd) * 3.57 * length.AA2AU

    assert (numbers == torch.tensor([6])).all()
    assert (periodic == torch.tensor([True, True, True])).all()
    assert pytest.approx(ref_lattice.cpu()) == lattice.cpu()
    assert pytest.approx(torch.zeros(1, 3, **dd).cpu()) == positions.cpu()


def test_read_periodic_lattice_frac() -> None:
    dd: DD = {"device": None, "dtype": torch.double}

    content = (
        "$coord frac\n"
        "0.25 0.25 0.25 c\n"
        "$periodic 3\n"
        "$lattice bohr\n"
        "4.0 0.0 0.0\n"
        "0.0 4.0 0.0\n"
        "0.0 0.0 4.0\n"
        "$end\n"
    )
    tmpdir, filepath = _write(content)
    with tmpdir:
        numbers, positions, lattice, periodic = read.read_turbomole(  # type: ignore[misc]
            filepath, **dd
        )

    ref_lattice = torch.eye(3, **dd) * 4.0
    ref_positions = torch.tensor([[1.0, 1.0, 1.0]], **dd)

    assert pytest.approx(ref_lattice.cpu()) == lattice.cpu()
    assert pytest.approx(ref_positions.cpu()) == positions.cpu()


def test_read_valid5_lattice_triangular_angs() -> None:
    """mctc-lib's own ``valid5-coord`` fixture: a non-diagonal (lower
    triangular) ``$lattice angs`` matrix, fractional coordinates, and an
    intervening ``$user-defined bonds`` group that must be skipped."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    content = (
        "$coord frac\n"
        "    0.25000000000000      0.25000000000000      0.25000000000000      f\n"
        "    0.75000000000000      0.75000000000000      0.75000000000000      f\n"
        "    0.00000000000000      0.00000000000000      0.00000000000000      ca\n"
        "$user-defined bonds\n"
        "$lattice angs\n"
        "       3.153833580475253       1.115048555743951       1.931320751454818\n"
        "       0.000000000000000       3.345145667231851       1.931320751454818\n"
        "       0.000000000000000       0.000000000000000       3.862641502909638\n"
        "$periodic 3\n"
        "$end\n"
    )
    tmpdir, filepath = _write(content)
    with tmpdir:
        numbers, positions, lattice, periodic = read.read_turbomole(  # type: ignore[misc]
            filepath, **dd
        )

    ref_lattice = (
        torch.tensor(
            [
                [3.153833580475253, 1.115048555743951, 1.931320751454818],
                [0.000000000000000, 3.345145667231851, 1.931320751454818],
                [0.000000000000000, 0.000000000000000, 3.862641502909638],
            ],
            **dd,
        )
        * length.AA2AU
    )
    frac = torch.tensor(
        [[0.25, 0.25, 0.25], [0.75, 0.75, 0.75], [0.0, 0.0, 0.0]], **dd
    )
    ref_positions = frac @ ref_lattice

    assert numbers.shape[0] == 3
    assert len(torch.unique(numbers)) == 2
    assert (periodic == torch.tensor([True, True, True])).all()
    assert pytest.approx(ref_lattice.cpu()) == lattice.cpu()
    assert pytest.approx(ref_positions.cpu()) == positions.cpu()


def test_read_valid11_lattice_2d_two_vectors() -> None:
    """mctc-lib's own ``valid11-coord`` fixture: ``$lattice`` for
    ``$periodic 2`` needs only 2 (not 3) lattice vectors."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    content = (
        "$coord frac\n"
        "    0.00000000000000      0.00000000000000      0.00000000000000      mg\n"
        "    0.50000000000000      0.00000000000000      0.00000000000000      o\n"
        "    0.00000000000000      0.50000000000000      0.00000000000000      o\n"
        "    0.00000000000000      0.00000000000000      3.97881835572287      o\n"
        "    0.50000000000000      0.50000000000000      0.00000000000000      mg\n"
        "    0.50000000000000      0.00000000000000      3.97881835572287      mg\n"
        "    0.00000000000000      0.50000000000000      3.97881835572287      mg\n"
        "    0.50000000000000      0.50000000000000      3.97881835572287      o\n"
        "$periodic 2\n"
        "$lattice\n"
        " 5.626898880882 -5.626898880882\n"
        " 5.626898880882  5.626898880882\n"
        "$end\n"
    )
    tmpdir, filepath = _write(content)
    with tmpdir:
        numbers, _, lattice, periodic = read.read_turbomole(  # type: ignore[misc]
            filepath, **dd
        )

    ref_lattice = torch.zeros(3, 3, **dd)
    ref_lattice[:2, :2] = torch.tensor(
        [[5.626898880882, -5.626898880882], [5.626898880882, 5.626898880882]],
        **dd,
    )

    assert numbers.shape[0] == 8
    assert len(torch.unique(numbers)) == 2
    assert (periodic == torch.tensor([True, True, False])).all()
    assert pytest.approx(ref_lattice.cpu()) == lattice.cpu()


################################################################################
# error handling
################################################################################


@pytest.mark.parametrize(
    "content",
    [
        # $cell and $lattice both present
        (
            "$coord\n0.0 0.0 0.0 c\n$periodic 3\n$cell\n"
            "1. 1. 1. 90. 90. 90.\n$lattice\n1 0 0\n0 1 0\n0 0 1\n$end\n"
        ),
        # cell without periodic
        ("$coord\n0.0 0.0 0.0 c\n$cell\n1. 1. 1. 90. 90. 90.\n$end\n"),
        # lattice without periodic
        ("$coord\n0.0 0.0 0.0 c\n$lattice\n1 0 0\n0 1 0\n0 0 1\n$end\n"),
        # periodic > 0 without cell or lattice
        ("$coord\n0.0 0.0 0.0 c\n$periodic 3\n$end\n"),
        # fractional coordinates on a molecular (non-periodic) system
        ("$coord frac\n0.0 0.0 0.0 c\n$end\n"),
        # duplicated $periodic group
        (
            "$coord\n0.0 0.0 0.0 c\n$periodic 3\n$cell\n1. 1. 1. 90. 90. 90.\n"
            "$periodic 3\n$end\n"
        ),
        # duplicated $cell group
        (
            "$coord\n0.0 0.0 0.0 c\n$periodic 3\n$cell\n1. 1. 1. 90. 90. 90.\n"
            "$cell\n1. 1. 1. 90. 90. 90.\n$end\n"
        ),
        # number of $lattice vectors does not match periodicity
        ("$coord\n0.0 0.0 0.0 c\n$periodic 3\n$lattice\n1 0 0\n0 1 0\n$end\n"),
        # unreadable periodicity value
        ("$coord\n0.0 0.0 0.0 c\n$periodic x\n$end\n"),
        # unreadable cell parameters
        (
            "$coord\n0.0 0.0 0.0 c\n$periodic 3\n$cell\n"
            "a b c 90. 90. 90.\n$end\n"
        ),
        # mctc-lib's invalid6-coord: explicit "$periodic 0" combined with
        # fractional coordinates (distinct code path from the implicit,
        # no-$periodic-tag-at-all "frac-without-periodic" case above)
        ("$coord frac\n0.0 0.0 0.0 c\n$periodic 0\n$end\n"),
        # mctc-lib's invalid9-coord: periodicity value out of the allowed
        # 0-3 range (distinct from the unparsable "bad-periodicity-value")
        (
            "$cell\n4.766080896955 4.766080896955 4.766080896955 60. 60. 60.\n"
            "$coord\n0.0 0.0 0.0 c\n2.38304045219106 1.39084904447079 "
            "0.97287218605834 c\n$periodic 4\n$end\n"
        ),
        # mctc-lib's invalid12-coord: duplicated $coord data group
        (
            "$coord\n-0.12918412100093 0.06210659750976 -2.13384498734326 c\n"
            "0.12856915667443 -0.07403227791901 4.02358027265954 c\n"
            "$eht unpaired=0 charge=0\n$periodic 2\n$cell angs\n"
            "2.4809835980 2.4811430162 120.2612191150\n"
            "$coord\n-0.12317720857511 2.75170732207802 -2.13345350602279 c\n"
            "2.44816466162280 1.28612566399214 4.02317048854901 c\n$end\n"
        ),
        # mctc-lib's invalid13-coord: duplicated $cell group, one before and
        # one after $periodic (order-independence of the duplicate check)
        (
            "$coord\n-0.12918412100093 0.06210659750976 -2.13384498734326 c\n"
            "0.12856915667443 -0.07403227791901 4.02358027265954 c\n"
            "-0.12317720857511 2.75170732207802 -2.13345350602279 c\n"
            "2.44816466162280 1.28612566399214 4.02317048854901 c\n"
            "$cell angs\n2.4809835980 2.4811430162 120.2612191150\n"
            "$eht unpaired=0 charge=0\n$periodic 2\n$cell angs\n"
            "2.4809835980 2.4811430162 120.2612191150\n$end\n"
        ),
        # mctc-lib's invalid16-coord: duplicated $lattice group
        (
            "$lattice angs\n"
            "3.153833580475253 1.115048555743951 1.931320751454818\n"
            "0.000000000000000 3.345145667231851 1.931320751454818\n"
            "0.000000000000000 0.000000000000000 3.862641502909638\n"
            "$coord frac\n0.25 0.25 0.25 f\n0.75 0.75 0.75 f\n0.0 0.0 0.0 ca\n"
            "$user-defined bonds\n$lattice angs\n"
            "3.153833580475253 1.115048555743951 1.931320751454818\n"
            "0.000000000000000 3.345145667231851 1.931320751454818\n"
            "0.000000000000000 0.000000000000000 3.862641502909638\n"
            "$periodic 3\n$end\n"
        ),
        # mctc-lib's invalid17-cell: unparseable $cell for $periodic 1
        ("$coord\n0.0 0.0 0.0 h\n$periodic 1\n$cell\nnot-a-cell\n$end\n"),
    ],
    ids=[
        "conflicting-cell-lattice",
        "cell-without-periodic",
        "lattice-without-periodic",
        "periodic-without-data",
        "frac-without-periodic",
        "duplicated-periodic",
        "duplicated-cell",
        "lattice-vector-count-mismatch",
        "bad-periodicity-value",
        "bad-cell-parameters",
        "explicit-periodic-zero-with-frac",
        "periodicity-out-of-range",
        "duplicated-coord",
        "duplicated-cell-across-periodic",
        "duplicated-lattice",
        "unparseable-cell-1d",
    ],
)
def test_read_periodic_fail(content: str) -> None:
    tmpdir, filepath = _write(content)
    with tmpdir:
        with pytest.raises(FormatErrorTM):
            read.read_turbomole(filepath)
