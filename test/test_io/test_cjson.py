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
Test the Chemical JSON (cjson) file reader, mirroring mctc-lib's
``mctc_io_read_cjson`` test cases (fixture content taken directly from
mctc-lib's own ``test_read_cjson.f90``).
"""

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest
import torch

from tad_mctc.exceptions import FormatErrorCJSON
from tad_mctc.io import read
from tad_mctc.typing import DD
from tad_mctc.units import length

_VALID1_ETHANE: dict[str, Any] = {
    "chemical json": 0,
    "name": "ethane",
    "atoms": {
        "elements": {"number": [1, 6, 1, 1, 6, 1, 1, 1]},
        "coords": {
            "3d": [
                1.185080,
                -0.003838,
                0.987524,
                0.751621,
                -0.022441,
                -0.020839,
                1.166929,
                0.833015,
                -0.569312,
                1.115519,
                -0.932892,
                -0.514525,
                -0.751587,
                0.022496,
                0.020891,
                -1.166882,
                -0.833372,
                0.568699,
                -1.115691,
                0.932608,
                0.515082,
                -1.184988,
                0.004424,
                -0.987522,
            ]
        },
    },
    "bonds": {
        "connections": {
            "index": [0, 1, 1, 2, 1, 3, 1, 4, 4, 5, 4, 6, 4, 7],
        },
        "order": [1, 1, 1, 1, 1, 1, 1],
    },
}

_VALID2_RUTILE: dict[str, Any] = {
    "chemical json": 0,
    "name": "TiO2 rutile",
    "unit cell": {
        "a": 2.95812,
        "b": 4.59373,
        "c": 4.59373,
        "alpha": 90.0,
        "beta": 90.0,
        "gamma": 90.0,
    },
    "atoms": {
        "elements": {"number": [22, 22, 8, 8, 8, 8]},
        "coords": {
            "3d fractional": [
                0.0,
                0.0,
                0.0,
                0.5,
                0.5,
                0.5,
                0.0,
                0.3053,
                0.3053,
                0.0,
                0.6947,
                0.6947,
                0.5,
                0.1947,
                0.8053,
                0.5,
                0.8053,
                0.1947,
            ]
        },
    },
}


_VALID4_LARGE_MOLECULE: dict[str, Any] = {
    "chemical json": 1,
    "atoms": {
        "elements": {
            "number": [
                6,
                7,
                6,
                7,
                6,
                6,
                6,
                8,
                7,
                6,
                8,
                7,
                6,
                6,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
            ]
        },
        "coords": {
            "3d": [
                1.0731997649702911e00,
                4.8899989290949721e-02,
                -7.5699983421776973e-02,
                2.5136994495022558e00,
                1.2599997240612813e-02,
                -7.5799983399877077e-02,
                3.3519992659154081e00,
                1.0958997599990143e00,
                -7.5299983509376570e-02,
                4.6189989884436962e00,
                7.3029984006504256e-01,
                -7.5499983465576764e-02,
                4.5790989971817559e00,
                -6.3139986172404194e-01,
                -7.5299983509376570e-02,
                3.3012992770186567e00,
                -1.1025997585317211e00,
                -7.5199983531276451e-02,
                2.9806993472297307e00,
                -2.4868994553714288e00,
                -7.3799983837875047e-02,
                1.8252996002611557e00,
                -2.9003993648153492e00,
                -7.5799983399877077e-02,
                4.1143990989505834e00,
                -3.3042992763616597e00,
                -6.9399984801470568e-02,
                5.4516988060832432e00,
                -2.8561993744951040e00,
                -7.2399984144473614e-02,
                6.3892986007497967e00,
                -3.6596991985294207e00,
                -7.2299984166373524e-02,
                5.6623987599401575e00,
                -1.4767996765823013e00,
                -7.4899983596976152e-02,
                7.0094984649266268e00,
                -9.3649979490745228e-01,
                -7.5199983531276451e-02,
                3.9205991413925863e00,
                -4.7408989617477202e00,
                -6.1599986509662634e-02,
                7.3399983925474632e-01,
                1.0878997617510062e00,
                -7.4999983575076257e-02,
                7.1239984398512435e-01,
                -4.5699989991746470e-01,
                8.2339981967623732e-01,
                7.1239984398512435e-01,
                -4.5579990018026340e-01,
                -9.7549978636649193e-01,
                2.9929993445360430e00,
                2.1175995362477531e00,
                -7.4799983618876062e-02,
                7.7652982994071955e00,
                -1.7262996219420552e00,
                -7.5899983377977168e-02,
                7.1485984344638682e00,
                -3.2179992952612718e-01,
                8.1969982048653345e-01,
                7.1479984345952676e00,
                -3.2079992974512617e-01,
                -9.6949978768048573e-01,
                2.8649993725679135e00,
                -5.0231988999243073e00,
                -5.8299987232359275e-02,
                4.4022990359007768e00,
                -5.1591988701404459e00,
                8.2839981858124223e-01,
                4.4001990363606742e00,
                -5.1692988679285561e00,
                -9.4779979243276369e-01,
            ]
        },
    },
    "bonds": {
        "connections": {
            "index": [
                0,
                1,
                1,
                2,
                2,
                3,
                3,
                4,
                1,
                5,
                4,
                5,
                5,
                6,
                6,
                7,
                6,
                8,
                8,
                9,
                9,
                10,
                4,
                11,
                9,
                11,
                11,
                12,
                8,
                13,
                0,
                14,
                0,
                15,
                0,
                16,
                2,
                17,
                12,
                18,
                12,
                19,
                12,
                20,
                13,
                21,
                13,
                22,
                13,
                23,
            ]
        },
        "order": [
            1,
            4,
            4,
            4,
            1,
            4,
            1,
            2,
            1,
            1,
            2,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
        ],
    },
}
"""mctc-lib's ``valid4``: a larger molecule with non-uniform bond orders
(single/double/aromatic) -- exercises that bond orders are picked up
positionally rather than just uniformly."""


def _write(data: Any) -> tuple[tempfile.TemporaryDirectory[str], Path]:
    tmpdir = tempfile.TemporaryDirectory()
    filepath = Path(tmpdir.name) / "mol.cjson"
    filepath.write_text(json.dumps(data), encoding="utf-8")
    return tmpdir, filepath


def test_read_molecule_with_bonds() -> None:
    """mctc-lib's ``valid1``: a molecule (no unit cell) with bonds."""
    dd: DD = {"device": None, "dtype": torch.double}

    tmpdir, filepath = _write(_VALID1_ETHANE)
    with tmpdir:
        result = read.read_cjson(filepath, **dd)

    numbers, positions, lattice, periodic, bonds, bond_orders = result
    assert lattice is None
    assert periodic is None
    assert bonds is not None
    assert bond_orders is not None

    assert numbers.shape == (8,)
    assert bonds.shape == (7, 2)
    assert bond_orders.shape == (7,)
    assert (bonds[0] == torch.tensor([0, 1])).all()
    assert (bond_orders == torch.ones(7, **dd)).all()

    ref_first = (
        torch.tensor([1.185080, -0.003838, 0.987524], **dd) * length.AA2AU
    )
    assert pytest.approx(ref_first.cpu()) == positions[0].cpu()


def test_read_periodic_fractional() -> None:
    """mctc-lib's ``valid2``: a periodic structure via cell parameters and
    fractional coordinates, no bonds."""
    dd: DD = {"device": None, "dtype": torch.double}

    tmpdir, filepath = _write(_VALID2_RUTILE)
    with tmpdir:
        result = read.read_cjson(filepath, **dd)

    numbers, positions, lattice, periodic, bonds, bond_orders = result
    assert bonds is None
    assert bond_orders is None
    assert lattice is not None
    assert periodic is not None

    assert numbers.shape == (6,)
    assert (periodic == torch.tensor([True, True, True])).all()
    assert lattice.shape == (3, 3)
    assert positions.shape == (6, 3)

    # regression test for a former bug (fixed here, reported upstream to
    # mctc-lib) where fractional coordinates were *also* multiplied by
    # AA2AU before the lattice transform, stretching every periodic bond
    # by that factor -- checked against the real rutile Ti-O bond lengths
    # (~1.949/~1.980 Angstrom experimentally)
    dists = torch.cdist(positions, positions)
    dists.fill_diagonal_(float("inf"))
    ti_o_bohr = dists[:2].min(dim=1).values
    ti_o_angstrom = ti_o_bohr / length.AA2AU
    assert pytest.approx(ti_o_angstrom.sort().values.cpu(), abs=1e-3) == [
        1.946,
        1.983,
    ]


def test_read_periodic_fractional_no_double_aatoau_scaling() -> None:
    """Fractional coordinates are dimensionless -- only the (already bohr)
    lattice carries a unit, so the Angstrom->bohr conversion must apply to
    cartesian coordinates only, not be applied to the raw fractional array
    a second time before the lattice transform (see the module docstring;
    formerly a bug ported from mctc-lib, now fixed here and reported
    upstream)."""
    dd: DD = {"device": None, "dtype": torch.double}

    data = {
        "chemical json": 0,
        "unit cell": {
            "a": 1.0,
            "b": 1.0,
            "c": 1.0,
            "alpha": 90.0,
            "beta": 90.0,
            "gamma": 90.0,
        },
        "atoms": {
            "elements": {"number": [6]},
            "coords": {"3d fractional": [1.0, 0.0, 0.0]},
        },
    }
    tmpdir, filepath = _write(data)
    with tmpdir:
        result = read.read_cjson(filepath, **dd)

    _, positions, lattice, _, _, _ = result
    assert lattice is not None

    # cellpar length 1.0 Angstrom -> bohr for the lattice vector itself;
    # the (dimensionless) fractional coordinate 1.0 just selects that
    # lattice vector, with no additional Angstrom->bohr scaling.
    ref_positions = lattice[0].unsqueeze(0)
    assert pytest.approx(ref_positions.cpu()) == positions.cpu()


def test_read_molecule_bonds_without_order() -> None:
    """mctc-lib's ``valid3``: identical to ``valid1`` except the ``bonds``
    block has no ``order`` key at all -- mctc-lib (and this port) default
    every bond order to 1 in that case."""
    dd: DD = {"device": None, "dtype": torch.double}

    data = json.loads(json.dumps(_VALID1_ETHANE))
    del data["bonds"]["order"]

    tmpdir, filepath = _write(data)
    with tmpdir:
        result = read.read_cjson(filepath, **dd)

    numbers, _, _, _, bonds, bond_orders = result
    assert numbers.shape == (8,)
    assert bonds is not None
    assert bond_orders is not None
    assert bonds.shape == (7, 2)
    assert (bonds[0] == torch.tensor([0, 1])).all()
    assert (bond_orders == torch.ones(7, **dd)).all()


def test_read_molecule_varied_bond_orders() -> None:
    """mctc-lib's ``valid4``: a larger molecule whose bond orders are not
    all 1 (single/double/aromatic) -- pins that bond orders are read
    positionally, not just defaulted."""
    dd: DD = {"device": None, "dtype": torch.double}

    tmpdir, filepath = _write(_VALID4_LARGE_MOLECULE)
    with tmpdir:
        result = read.read_cjson(filepath, **dd)

    numbers, positions, lattice, periodic, bonds, bond_orders = result
    assert lattice is None
    assert periodic is None
    assert bonds is not None
    assert bond_orders is not None

    assert numbers.shape == (24,)
    assert positions.shape == (24, 3)
    assert bonds.shape == (25, 2)
    assert bond_orders.shape == (25,)

    ref_orders = torch.tensor(
        [
            1,
            4,
            4,
            4,
            1,
            4,
            1,
            2,
            1,
            1,
            2,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
        ],
        **dd,
    )
    assert pytest.approx(ref_orders.cpu()) == bond_orders.cpu()
    assert (bonds[1] == torch.tensor([1, 2])).all()


def test_read_molecule_chemicaljson_alias() -> None:
    """mctc-lib's ``valid5``: the ``chemicalJson`` (no space) alias for the
    schema-version key, exercised here on a *successful* parse rather than
    only via the invalid-value/invalid-type tests -- ``formalCharges`` is
    parsed for the (dropped) total charge but otherwise unused here."""
    dd: DD = {"device": None, "dtype": torch.double}

    data: dict[str, Any] = {
        "chemicalJson": 1,
        "atoms": {
            "elements": {"number": [8, 1]},
            "coords": {
                "3d": [
                    1.2358341722502633e00,
                    -9.1774253284895344e-02,
                    -6.7936144993384059e-02,
                    1.5475582000473165e00,
                    5.7192830956765273e-01,
                    5.5691301045614838e-01,
                ]
            },
            "formalCharges": [-1, 0],
        },
    }
    tmpdir, filepath = _write(data)
    with tmpdir:
        result = read.read_cjson(filepath, **dd)

    numbers, positions, lattice, periodic, bonds, bond_orders = result
    assert lattice is None
    assert periodic is None
    assert bonds is None
    assert bond_orders is None
    assert numbers.shape == (2,)
    assert positions.shape == (2, 3)
    assert (numbers == torch.tensor([8, 1])).all()


def test_read_periodic_camelcase_aliases_match_spaced_keys() -> None:
    """None of mctc-lib's own tests ever exercise the ``unitCell``/
    ``3dFractional`` camelCase aliases (every periodic fixture uses the
    spaced form) -- a real gap in both suites' coverage of
    ``_get_alias``. Renaming ``valid2`` (rutile) to the camelCase keys
    must give the identical structure as the spaced-key original."""
    dd: DD = {"device": None, "dtype": torch.double}

    spaced = json.loads(json.dumps(_VALID2_RUTILE))
    camel = json.loads(json.dumps(_VALID2_RUTILE))
    camel["unitCell"] = camel.pop("unit cell")
    camel["atoms"]["coords"]["3dFractional"] = camel["atoms"]["coords"].pop(
        "3d fractional"
    )

    tmpdir1, filepath1 = _write(spaced)
    tmpdir2, filepath2 = _write(camel)
    with tmpdir1, tmpdir2:
        result_spaced = read.read_cjson(filepath1, **dd)
        result_camel = read.read_cjson(filepath2, **dd)

    numbers_s, positions_s, lattice_s, periodic_s, _, _ = result_spaced
    numbers_c, positions_c, lattice_c, periodic_c, _, _ = result_camel

    assert lattice_s is not None and lattice_c is not None
    assert (numbers_s == numbers_c).all()
    assert pytest.approx(lattice_s.cpu()) == lattice_c.cpu()
    assert pytest.approx(positions_s.cpu()) == positions_c.cpu()
    assert periodic_s is not None and periodic_c is not None
    assert (periodic_s == periodic_c).all()


def test_read_fail_notfound() -> None:
    with pytest.raises(FileNotFoundError):
        read.read_cjson("not found")


def test_read_fail_malformed_json() -> None:
    tmpdir = tempfile.TemporaryDirectory()
    filepath = Path(tmpdir.name) / "mol.cjson"
    filepath.write_text("{not valid json", encoding="utf-8")
    with tmpdir:
        with pytest.raises(FormatErrorCJSON):
            read.read_cjson(filepath)


def test_read_fail_invalid_root_data() -> None:
    """mctc-lib's ``invalid-root-data``: the root JSON value is an array
    (``[{...}]``), not an object."""
    tmpdir, filepath = _write([_VALID2_RUTILE])
    with tmpdir:
        with pytest.raises(FormatErrorCJSON):
            read.read_cjson(filepath)


def _mutated(**overrides: Any) -> dict[str, Any]:
    data = json.loads(json.dumps(_VALID1_ETHANE))
    for key, value in overrides.items():
        if value is _DELETE:
            data.pop(key)
        else:
            data[key] = value
    return data


_DELETE = object()


@pytest.mark.parametrize(
    "data",
    [
        _mutated(**{"chemical json": _DELETE}),
        _mutated(**{"chemical json": 2}),
        _mutated(**{"chemical json": "0"}),
        _mutated(
            atoms={
                "elements": {"number": [1, 6, 1]},  # 3 atoms, 8 coords
                "coords": _VALID1_ETHANE["atoms"]["coords"],
            }
        ),
        _mutated(atoms={"elements": {"number": [1, 6]}, "coords": {}}),
        _mutated(
            atoms={
                "elements": {"number": [1, 999]},
                "coords": {"3d": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]},
            }
        ),
        _mutated(
            atoms={
                "elements": {"number": [1, -6]},
                "coords": {"3d": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]},
            }
        ),
        _mutated(bonds=[]),
        _mutated(
            bonds={
                "connections": {
                    "index": [0, 1, 1, 2, 1, 3, 1, 4, 4, 5, 4, 6, 4, 7]
                },
                "order": [1, 1, 1, 1, 1, 1],  # 6 orders, 7 bonds
            }
        ),
        # mctc-lib test_invalid_element_type1: elements.number holds
        # element symbols instead of atomic numbers.
        _mutated(
            atoms={
                "elements": {
                    "number": ["H", "C", "H", "H", "C", "H", "H", "H"]
                },
                "coords": _VALID1_ETHANE["atoms"]["coords"],
            }
        ),
        # mctc-lib test_invalid_element_type2: "elements" itself is an
        # array instead of an object.
        _mutated(
            atoms={
                "elements": [1, 6, 1, 1, 6, 1, 1, 1],
                "coords": _VALID1_ETHANE["atoms"]["coords"],
            }
        ),
        # mctc-lib test_numbers_coords_mismatch2 / test_invalid_element_number:
        # both fixtures misspell "number" as "numbers" under "elements" --
        # same code path (atomic numbers not found), ported as one case.
        _mutated(
            atoms={
                "elements": {"numbers": [1, 6, 1, 1, 6, 1, 1, 1]},
                "coords": _VALID1_ETHANE["atoms"]["coords"],
            }
        ),
        # mctc-lib test_invalid_coordinate_type1: coords."3d" is an object,
        # not an array.
        _mutated(
            atoms={
                "elements": {"number": [1, 6, 1, 1, 6, 1, 1, 1]},
                "coords": {"3d": {}},
            }
        ),
        # mctc-lib test_invalid_coordinate_type2, ported verbatim: the
        # unit cell has a typo'd "gama" key instead of "gamma", which
        # (matching mctc-lib's own check order) raises on the *missing
        # cell parameter* before coordinate parsing is ever reached --
        # not on the "3d fractional": {} that the fixture also carries.
        _mutated(
            **{
                "unit cell": {
                    "a": 2.95812,
                    "b": 4.59373,
                    "c": 4.59373,
                    "alpha": 90.0,
                    "beta": 90.0,
                    "gama": 90.0,
                }
            }
        ),
        # mctc-lib test_invalid_atoms_type, ported verbatim: the unit cell
        # parameter "a" is a string, which (matching mctc-lib's own check
        # order, unit cell before atoms) raises before "atoms": [] is ever
        # inspected -- not the atoms-wrong-type check its name advertises.
        {
            "chemical json": 1,
            "unit cell": {
                "a": "xyz",
                "b": 4.59373,
                "c": 4.59373,
                "alpha": 90.0,
                "beta": 90.0,
                "gamma": 90.0,
            },
            "atoms": [],
        },
        # Isolated variant of the above: "atoms" is an array with no unit
        # cell in the way, actually exercising `isinstance(atoms, dict)`
        # (no fixture in mctc-lib's own suite reaches this branch).
        _mutated(atoms=[]),
        # Isolated variant of test_invalid_coordinate_type2: a
        # well-formed unit cell (all 6 keys present and numeric) paired
        # with "3d fractional": {}, actually exercising the fractional
        # not-an-array check (no fixture in mctc-lib's own suite reaches
        # it either, since its ported fixture above is masked by the
        # "gama" typo).
        _mutated(
            **{
                "unit cell": {
                    "a": 2.95812,
                    "b": 4.59373,
                    "c": 4.59373,
                    "alpha": 90.0,
                    "beta": 90.0,
                    "gamma": 90.0,
                }
            },
            atoms={
                "elements": {"number": [22, 22, 8, 8, 8, 8]},
                "coords": {"3d fractional": {}},
            },
        ),
    ],
    ids=[
        "missing-schema",
        "invalid-schema-value",
        "invalid-schema-type",
        "numbers-coords-mismatch",
        "missing-coords",
        "invalid-element-number-too-large",
        "invalid-element-number-negative",
        "bonds-wrong-type",
        "mismatch-bonds",
        "invalid-element-type-string",
        "elements-not-a-dict",
        "element-number-key-typo",
        "coords-3d-not-array",
        "unit-cell-missing-gamma",
        "unit-cell-param-not-numeric",
        "atoms-not-a-dict",
        "fractional-coords-not-array",
    ],
)
def test_read_fail_format(data: dict[str, Any]) -> None:
    tmpdir, filepath = _write(data)
    with tmpdir:
        with pytest.raises(FormatErrorCJSON):
            read.read_cjson(filepath)


def test_read_fail_invalid_unit_cell_type() -> None:
    data = json.loads(json.dumps(_VALID2_RUTILE))
    data["unit cell"] = []

    tmpdir, filepath = _write(data)
    with tmpdir:
        with pytest.raises(FormatErrorCJSON):
            read.read_cjson(filepath)
