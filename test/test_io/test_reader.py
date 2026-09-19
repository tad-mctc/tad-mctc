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
Test the general file reader.
"""

import tempfile
from pathlib import Path

import pytest
import torch

from tad_mctc.io import read
from tad_mctc.io.structure import Structure
from tad_mctc.typing import DD
from tad_mctc.units import length

from ..conftest import DEVICE


def test_fail_crystal() -> None:
    # "crystal" (the CRYSTAL quantum-chemistry code's format) is a distinct,
    # still-unimplemented format that happens to share the VASP dispatch
    # branch's neighbourhood; content is irrelevant since dispatch raises
    # before the file is parsed.
    p = Path(__file__).parent.resolve() / "files" / "mol.xyz"
    with pytest.raises(NotImplementedError):
        read.read(p, ftype="crystal")


def test_fail_unknown() -> None:
    p = Path(__file__).parent.resolve() / "files" / "mol.xyz"
    with pytest.raises(ValueError):
        read.read(p, ftype="something")


def test_fail_notfound() -> None:
    p = Path(__file__).parent.resolve() / "files" / "notfound"
    with pytest.raises(FileNotFoundError):
        read.read(p)


def test_read_gaussian_dispatch() -> None:
    """The general dispatcher recognizes the ``.ein`` extension."""
    p = Path(__file__).parent.resolve() / "files" / "mol.ein"

    result = read.read(p)
    assert len(result) == 2
    numbers, positions = result

    assert (numbers == torch.tensor([7, 1, 1, 1], device=DEVICE)).all()
    assert positions.shape == (4, 3)


def test_read_qcjson_dispatch() -> None:
    """The general dispatcher recognizes the ``.qcjson`` extension
    (mctc-lib's actual QCSchema extension) and routes it straight to
    qcschema, not through the ``.json`` sniffer."""
    content = (
        '{"schema_version": 1, "molecule": {"symbols": ["O", "H"], '
        '"geometry": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]}}'
    )

    with tempfile.TemporaryDirectory() as tmpdirname:
        filepath = Path(tmpdirname) / "mol.qcjson"
        filepath.write_text(content, encoding="utf-8")

        result = read.read(filepath)

    assert len(result) == 2
    numbers, positions = result
    assert (numbers == torch.tensor([8, 1], device=DEVICE)).all()
    assert positions.shape == (2, 3)


def test_read_qchem_dispatch() -> None:
    """The general dispatcher recognizes the ``.qchem`` extension."""
    content = "$molecule\n0 1\nO 0.0 0.0 0.0\nH 1.0 0.0 0.0\n$end\n"

    with tempfile.TemporaryDirectory() as tmpdirname:
        filepath = Path(tmpdirname) / "mol.qchem"
        filepath.write_text(content, encoding="utf-8")

        result = read.read(filepath)

    assert len(result) == 2
    numbers, positions = result

    assert (numbers == torch.tensor([8, 1], device=DEVICE)).all()
    assert positions.shape == (2, 3)


def test_read_pdb_dispatch() -> None:
    """The general dispatcher recognizes the ``.pdb`` extension."""
    content = (
        "ATOM      1  N   GLY Z   1      -0.821  -2.072  16.609  1.00  9.93           N\n"
        "ATOM      2  CA  GLY Z   1      -1.705  -2.345  15.487  1.00  7.38           C\n"
        "END\n"
    )

    with tempfile.TemporaryDirectory() as tmpdirname:
        filepath = Path(tmpdirname) / "mol.pdb"
        filepath.write_text(content, encoding="utf-8")

        result = read.read(filepath)

    assert len(result) == 2
    numbers, positions = result

    assert (numbers == torch.tensor([7, 6], device=DEVICE)).all()
    assert positions.shape == (2, 3)


def test_read_genformat_dispatch() -> None:
    """The general dispatcher recognizes the ``.gen`` extension."""
    content = "2 C\nO H\n1 1 0.0 0.0 0.0\n2 2 1.0 0.0 0.0\n"

    with tempfile.TemporaryDirectory() as tmpdirname:
        filepath = Path(tmpdirname) / "mol.gen"
        filepath.write_text(content, encoding="utf-8")

        result = read.read(filepath)

    assert len(result) == 2
    numbers, positions = result

    assert (numbers == torch.tensor([8, 1], device=DEVICE)).all()
    assert positions.shape == (2, 3)


def test_read_pymatgen_dispatch() -> None:
    """The general dispatcher recognizes the ``.pmgjson`` extension."""
    content = (
        '{"@module": "pymatgen.core.structure", "@class": "Molecule", '
        '"sites": [{"label": "O", "xyz": [0.0, 0.0, 0.0]}, '
        '{"label": "H", "xyz": [1.0, 0.0, 0.0]}]}'
    )

    with tempfile.TemporaryDirectory() as tmpdirname:
        filepath = Path(tmpdirname) / "mol.pmgjson"
        filepath.write_text(content, encoding="utf-8")

        result = read.read(filepath)

    assert len(result) == 2
    numbers, positions = result

    assert (numbers == torch.tensor([8, 1], device=DEVICE)).all()
    assert positions.shape == (2, 3)


def test_read_cjson_dispatch() -> None:
    """The general dispatcher recognizes the ``.cjson`` extension and
    returns the 6-tuple carrying lattice/periodic and bonds/bond_orders
    (all ``None`` here, since this fixture has neither)."""
    content = (
        '{"chemicalJson": 1, "atoms": {"elements": {"number": [8, 1]}, '
        '"coords": {"3d": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]}}}'
    )

    with tempfile.TemporaryDirectory() as tmpdirname:
        filepath = Path(tmpdirname) / "mol.cjson"
        filepath.write_text(content, encoding="utf-8")

        result = read.read(filepath)

    assert len(result) == 6
    numbers, positions, lattice, periodic, bonds, bond_orders = result

    assert (numbers == torch.tensor([8, 1], device=DEVICE)).all()
    assert positions.shape == (2, 3)
    assert lattice is None
    assert periodic is None
    assert bonds is None
    assert bond_orders is None


def test_read_json_dispatch_sniffs_cjson() -> None:
    """A bare ``.json`` extension goes through the sniff-and-dispatch
    reader, not straight to qcschema -- a ``chemicalJson`` key routes it
    to cjson and its 6-tuple return shape, distinct from ``.cjson``
    itself only in how the format was inferred."""
    content = (
        '{"chemicalJson": 1, "atoms": {"elements": {"number": [8, 1]}, '
        '"coords": {"3d": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]}}}'
    )

    with tempfile.TemporaryDirectory() as tmpdirname:
        filepath = Path(tmpdirname) / "mol.json"
        filepath.write_text(content, encoding="utf-8")

        result = read.read(filepath)

    assert len(result) == 6
    numbers, positions, *_ = result
    assert (numbers == torch.tensor([8, 1], device=DEVICE)).all()
    assert positions.shape == (2, 3)


def test_read_structure_cjson_bonds_and_lattice() -> None:
    """``read_structure`` threads both bond connectivity and a periodic
    lattice from a cjson file into the returned ``Structure`` at once --
    the combination no other format in this dispatcher produces."""
    content = (
        '{"chemicalJson": 1, '
        '"unitCell": {"a": 5.0, "b": 5.0, "c": 5.0, '
        '"alpha": 90.0, "beta": 90.0, "gamma": 90.0}, '
        '"atoms": {"elements": {"number": [8, 1]}, '
        '"coords": {"3d": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]}}, '
        '"bonds": {"connections": {"index": [0, 1]}, "order": [1]}}'
    )

    with tempfile.TemporaryDirectory() as tmpdirname:
        filepath = Path(tmpdirname) / "mol.cjson"
        filepath.write_text(content, encoding="utf-8")

        structure = read.read_structure(filepath)

    assert isinstance(structure, Structure)
    assert structure.lattice is not None
    assert structure.periodic is not None
    assert structure.bonds is not None
    assert structure.bond_orders is not None
    assert structure.bonds.shape == (1, 2)


_V2000_ETHANE_FRAGMENT = (
    "\n\n\n"
    "  2  1  0  0  0  0            999 V2000\n"
    "    0.0000    0.0000    0.0000 N   0  0  0  0  0  0  0  0  0  0  0  0\n"
    "    1.0000    0.0000    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0\n"
    "  1  2  1  0  0  0  0\n"
    "M  END\n"
)


def test_read_molfile_dispatch() -> None:
    """The general dispatcher recognizes the ``.mol`` extension and
    returns the 6-tuple, with bonds/bond_orders set and lattice/periodic
    ``None`` (molfiles are never periodic)."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        filepath = Path(tmpdirname) / "mol.mol"
        filepath.write_text(_V2000_ETHANE_FRAGMENT, encoding="utf-8")

        result = read.read(filepath)

    assert len(result) == 6
    numbers, positions, lattice, periodic, bonds, bond_orders = result

    assert (numbers == torch.tensor([7, 1], device=DEVICE)).all()
    assert positions.shape == (2, 3)
    assert lattice is None
    assert periodic is None
    assert bonds is not None and bonds.shape == (1, 2)
    assert bond_orders is not None


def test_read_sdf_dispatch() -> None:
    """The general dispatcher recognizes the ``.sdf`` extension."""
    content = _V2000_ETHANE_FRAGMENT + ">  <note>\nsome value\n\n$$$$\n"

    with tempfile.TemporaryDirectory() as tmpdirname:
        filepath = Path(tmpdirname) / "mol.sdf"
        filepath.write_text(content, encoding="utf-8")

        result = read.read(filepath)

    assert len(result) == 6
    numbers, positions, _, _, bonds, _ = result

    assert (numbers == torch.tensor([7, 1], device=DEVICE)).all()
    assert positions.shape == (2, 3)
    assert bonds is not None and bonds.shape == (1, 2)


def test_read_structure_molfile_bonds() -> None:
    """``read_structure`` threads bond connectivity from a molfile into
    the returned ``Structure``, with ``lattice``/``periodic`` left unset."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        filepath = Path(tmpdirname) / "mol.mol"
        filepath.write_text(_V2000_ETHANE_FRAGMENT, encoding="utf-8")

        structure = read.read_structure(filepath)

    assert isinstance(structure, Structure)
    assert structure.lattice is None
    assert structure.periodic is None
    assert structure.bonds is not None
    assert structure.bonds.shape == (1, 2)


################################################################################


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("file", ["mol.xyz", "mol.json", "coord", "qm9.xyz"])
def test_types(dtype: torch.dtype, file: str) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    p = Path(__file__).parent.resolve() / "files" / file

    ref_numbers = torch.tensor([8, 1, 1], device=DEVICE)
    ref_positions = torch.tensor(
        [
            [+0.00000000000000, +0.00000000000000, -0.74288549752983],
            [-1.43472674945442, +0.00000000000000, +0.37144274876492],
            [+1.43472674945442, +0.00000000000000, +0.37144274876492],
        ],
        **dd,
    )

    ftype = None if "qm9" not in file else "qm9"
    numbers, positions = read.read(p, ftype=ftype, **dd)  # type: ignore[misc]

    assert (ref_numbers == numbers).all()
    assert pytest.approx(ref_positions.cpu()) == positions.cpu()


################################################################################


def _poscar_reference() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ref_numbers = torch.tensor([6, 6], device=DEVICE)
    ref_lattice = (
        torch.tensor(
            [
                [0.5, 0.5, 0.0],
                [0.0, 0.5, 0.5],
                [0.5, 0.0, 0.5],
            ]
        )
        * 3.7
        * length.AA2AU
    )
    frac = torch.tensor([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]])
    ref_positions = frac @ ref_lattice
    return ref_numbers, ref_positions, ref_lattice


@pytest.mark.parametrize("ftype", [None, "poscar", "vasp"])
def test_read_poscar_dispatch_by_filename(ftype: str | None) -> None:
    """The general dispatcher recognizes a bare ``POSCAR`` filename (no
    extension to infer from) and returns a 3-tuple including the lattice."""
    p = Path(__file__).parent.resolve() / "files" / "POSCAR"

    result = read.read(p, ftype=ftype)
    assert len(result) == 3
    numbers, positions, lattice = result

    ref_numbers, ref_positions, ref_lattice = _poscar_reference()
    assert (ref_numbers == numbers).all()
    assert pytest.approx(ref_lattice.cpu()) == lattice.cpu()
    assert pytest.approx(ref_positions.cpu()) == positions.cpu()


@pytest.mark.parametrize("suffix", ["poscar", "vasp", "contcar"])
def test_read_poscar_dispatch_by_extension(suffix: str) -> None:
    """The general dispatcher infers the POSCAR reader from a
    ``.poscar``/``.vasp``/``.contcar`` extension."""
    content = (Path(__file__).parent.resolve() / "files" / "POSCAR").read_text(
        encoding="utf-8"
    )

    with tempfile.TemporaryDirectory() as tmpdirname:
        filepath = Path(tmpdirname) / f"structure.{suffix}"
        filepath.write_text(content, encoding="utf-8")

        result = read.read(filepath)

    assert len(result) == 3
    numbers, positions, lattice = result

    ref_numbers, ref_positions, ref_lattice = _poscar_reference()
    assert (ref_numbers == numbers).all()
    assert pytest.approx(ref_lattice.cpu()) == lattice.cpu()
    assert pytest.approx(ref_positions.cpu()) == positions.cpu()


def test_read_structure_poscar_lattice() -> None:
    """``read_structure`` threads the lattice from a periodic file (VASP
    POSCAR) into the returned ``Structure``."""
    p = Path(__file__).parent.resolve() / "files" / "POSCAR"

    structure = read.read_structure(p)

    assert isinstance(structure, Structure)
    assert structure.lattice is not None
    assert structure.lattice.shape == (3, 3)


def test_read_structure_non_periodic_lattice_is_none() -> None:
    """``read_structure`` leaves ``lattice`` unset for a non-periodic file,
    i.e., the POSCAR-only 3-tuple return does not leak into other formats."""
    p = Path(__file__).parent.resolve() / "files" / "mol.xyz"

    structure = read.read_structure(p)

    assert isinstance(structure, Structure)
    assert structure.lattice is None


################################################################################


@pytest.mark.parametrize("ftype", [None, "aims"])
def test_read_aims_dispatch_by_filename(ftype: str | None) -> None:
    """The general dispatcher recognizes the bare ``geometry.in`` filename
    (mctc-lib's own detection rule; ``.in`` alone is not distinctive
    enough to infer from)."""
    content = "atom 0.0 0.0 0.0 C\natom 1.0 0.0 0.0 H\n"

    with tempfile.TemporaryDirectory() as tmpdirname:
        filepath = Path(tmpdirname) / "geometry.in"
        filepath.write_text(content, encoding="utf-8")

        result = read.read(filepath, ftype=ftype)

    assert len(result) == 2
    numbers, positions = result

    assert (numbers == torch.tensor([6, 1])).all()
    ref_positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]) * (
        length.AA2AU
    )
    assert pytest.approx(ref_positions.cpu()) == positions.cpu()


def test_read_structure_aims_periodic() -> None:
    """``read_structure`` threads lattice and periodicity mask from a
    periodic FHI-aims file into the returned ``Structure``."""
    content = (
        "lattice_vector 1.0 0.0 0.0\n"
        "lattice_vector 0.0 1.0 0.0\n"
        "lattice_vector 0.0 0.0 1.0\n"
        "atom 0.0 0.0 0.0 C\n"
    )

    with tempfile.TemporaryDirectory() as tmpdirname:
        filepath = Path(tmpdirname) / "geometry.in"
        filepath.write_text(content, encoding="utf-8")

        structure = read.read_structure(filepath)

    assert isinstance(structure, Structure)
    assert structure.lattice is not None
    assert structure.periodic is not None
    assert (structure.periodic == torch.tensor([True, True, True])).all()


################################################################################


def test_read_coord_dispatch_by_extension() -> None:
    """The general dispatcher recognizes a ``.coord`` *extension* (mctc-lib's
    ``get_filetype`` maps both the bare filename ``coord`` and the extension
    ``.coord``/``.tmol`` to its ``tmol`` filetype; only the bare-filename case
    was dispatch-tested here before, via ``test_types``)."""
    content = (Path(__file__).parent.resolve() / "files" / "coord").read_text(
        encoding="utf-8"
    )

    with tempfile.TemporaryDirectory() as tmpdirname:
        filepath = Path(tmpdirname) / "structure.coord"
        filepath.write_text(content, encoding="utf-8")

        result = read.read(filepath)

    assert len(result) == 2
    numbers, positions = result

    assert (numbers == torch.tensor([8, 1, 1], device=DEVICE)).all()
    assert positions.shape == (3, 3)


################################################################################


def test_read_structure_extxyz_periodic() -> None:
    """The general dispatcher should recognize the ``.extxyz`` extension
    and, via ``read_structure``, report the structure as periodic --
    mirroring mctc-lib's ``test_extxyz``."""
    content = (
        '2\nLattice="5 0 0 0 5 0 0 0 5" Properties=species:S:1:pos:R:3 '
        'pbc="T T T"\nH 0.0 0.0 0.0\nO 1.0 1.0 1.0\n'
    )

    with tempfile.TemporaryDirectory() as tmpdirname:
        filepath = Path(tmpdirname) / "mol.extxyz"
        filepath.write_text(content, encoding="utf-8")

        structure = read.read_structure(filepath)

    assert structure.periodic is not None
    assert bool(structure.periodic.all())


################################################################################


def test_read_structure_chrg_uhf_sidecars() -> None:
    """``read_structure`` picks up ``.CHRG``/``.UHF`` sidecar files from the
    same directory as the geometry file (mctc-lib's ``read_structure`` does
    the same, though ``test_read.f90`` itself has no dedicated case for it --
    this closes that gap directly against the dispatcher's documented
    behavior)."""
    content = (Path(__file__).parent.resolve() / "files" / "mol.xyz").read_text(
        encoding="utf-8"
    )

    with tempfile.TemporaryDirectory() as tmpdirname:
        filepath = Path(tmpdirname) / "mol.xyz"
        filepath.write_text(content, encoding="utf-8")
        (Path(tmpdirname) / ".CHRG").write_text("1\n", encoding="utf-8")
        (Path(tmpdirname) / ".UHF").write_text("2\n", encoding="utf-8")

        structure = read.read_structure(filepath)

    assert isinstance(structure, Structure)
    assert structure.charge is not None
    assert structure.charge.item() == pytest.approx(1.0)
    assert structure.uhf is not None
    assert structure.uhf.item() == 2
