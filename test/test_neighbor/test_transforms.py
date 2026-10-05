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
Test building a neighbour list *inside* a `torch.func` transform.

A list is index data, so under `jacrev` (and `grad`) the build runs on the
values under the wrappers and gives the list it gives outside. Under `vmap`
the size of the list depends on the data, so the build refuses with a
message that says what to do instead. Evaluating a list that was built
outside is covered by `test_ncoord/test_transforms.py`.
"""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

import pytest
import torch

from tad_mctc.autograd import jacrev_matches_finite_diff
from tad_mctc.io.structure import Structure
from tad_mctc.ncoord import cn_d3
from tad_mctc.neighbor import list as nblist
from tad_mctc.neighbor.list import NeighborList, build_neighborlist
from tad_mctc.tools.testing import requires_compile
from tad_mctc.typing import DD, Tensor

from ..conftest import DEVICE
from ..utils import (
    jacrev,
    load_structure,
)

DD_DOUBLE: DD = {"device": DEVICE, "dtype": torch.double}

CUTOFF = 25.0

VMAP_MESSAGE = "cannot be built inside `torch.func.vmap`"


@pytest.fixture(autouse=True)
def _no_layer_left_behind() -> Generator[None]:
    """A build sets the `torch.func` layers aside and puts them back; after
    any test, whether the build or the transform raised or not, none is
    left on the stack."""
    yield
    assert getattr(torch._C, "_functorch").peek_interpreter_stack() is None


def _molecule() -> Structure:
    return load_structure("mb16_43", "02", DD_DOUBLE)


def _cell() -> Structure:
    return load_structure("other", "periodic_cubic", DD_DOUBLE)


def _triclinic() -> Structure:
    return load_structure("other", "periodic_triclinic", DD_DOUBLE)


def _assert_same_list(built: NeighborList, expected: NeighborList) -> None:
    assert torch.equal(built.idx_i, expected.idx_i)
    assert torch.equal(built.idx_j, expected.idx_j)
    assert torch.equal(built.shift, expected.shift)
    assert torch.equal(built.mask, expected.mask)


def _check_list_built_inside_jacrev(structure: Structure) -> None:
    """The list built from the wrapped positions is the list built from
    the plain ones."""
    expected = build_neighborlist(structure, CUTOFF)
    seen: list[NeighborList] = []

    def f(positions: Tensor) -> Tensor:
        seen.append(
            build_neighborlist(structure.replace(positions=positions), CUTOFF)
        )
        return positions.sum()

    jacrev(f)(structure.positions)

    assert len(seen) == 1
    _assert_same_list(seen[0], expected)


def _check_gradient_with_list_built_inside_jacrev(
    structure: Structure,
) -> None:
    """The coordination number's gradient does not depend on whether the
    list is built inside or outside the transform."""
    nbl = build_neighborlist(structure, CUTOFF)

    def outside(positions: Tensor) -> Tensor:
        return cn_d3(structure.replace(positions=positions), pairs=nbl).sum()

    def inside(positions: Tensor) -> Tensor:
        target = structure.replace(positions=positions)
        built = build_neighborlist(target, CUTOFF)
        return cn_d3(target, pairs=built).sum()

    ref = jacrev(outside)(structure.positions)
    out = jacrev(inside)(structure.positions)

    assert torch.isfinite(out).all()
    assert out.abs().sum() > 0
    assert torch.allclose(out, ref, atol=1e-12, rtol=0)


########################################################################
# jacrev


def test_list_built_inside_jacrev_molecule() -> None:
    _check_list_built_inside_jacrev(_molecule())


def test_list_built_inside_jacrev_cell() -> None:
    _check_list_built_inside_jacrev(_cell())


def test_gradient_with_list_built_inside_jacrev_molecule() -> None:
    _check_gradient_with_list_built_inside_jacrev(_molecule())


def test_gradient_with_list_built_inside_jacrev_cell() -> None:
    _check_gradient_with_list_built_inside_jacrev(_cell())


def test_list_built_inside_jacrev_wrt_lattice() -> None:
    """The lattice is wrapped too, and is read by the build."""
    structure = _cell()
    assert structure.lattice is not None
    expected = build_neighborlist(structure, CUTOFF)
    seen: list[NeighborList] = []

    def f(lattice: Tensor) -> Tensor:
        seen.append(
            build_neighborlist(structure.replace(lattice=lattice), CUTOFF)
        )
        return lattice.sum()

    jacrev(f)(structure.lattice)

    assert len(seen) == 1
    _assert_same_list(seen[0], expected)


def test_list_built_inside_nested_jacrev() -> None:
    """Two grad-tracking layers are both set aside."""
    structure = _molecule()
    expected = build_neighborlist(structure, CUTOFF)
    seen: list[NeighborList] = []

    def f(positions: Tensor) -> Tensor:
        seen.append(
            build_neighborlist(structure.replace(positions=positions), CUTOFF)
        )
        return (positions**2).sum()

    jacrev(jacrev(f))(structure.positions)

    assert len(seen) == 1
    _assert_same_list(seen[0], expected)


def test_list_built_inside_grad() -> None:
    structure = _molecule()
    expected = build_neighborlist(structure, CUTOFF)
    seen: list[NeighborList] = []

    def f(positions: Tensor) -> Tensor:
        seen.append(
            build_neighborlist(structure.replace(positions=positions), CUTOFF)
        )
        return positions.sum()

    torch.func.grad(f)(structure.positions)

    assert len(seen) == 1
    _assert_same_list(seen[0], expected)


########################################################################
# vmap


def test_list_built_inside_vmap_raises() -> None:
    structure = _molecule()

    def f(positions: Tensor) -> Tensor:
        build_neighborlist(structure.replace(positions=positions), CUTOFF)
        return positions.sum()

    stacked = torch.stack([structure.positions, structure.positions])
    with pytest.raises(RuntimeError, match=VMAP_MESSAGE):
        torch.func.vmap(f)(stacked)


def test_list_built_inside_vmap_of_jacrev_raises() -> None:
    structure = _molecule()

    def f(positions: Tensor) -> Tensor:
        build_neighborlist(structure.replace(positions=positions), CUTOFF)
        return positions.sum()

    stacked = torch.stack([structure.positions, structure.positions])
    with pytest.raises(RuntimeError, match=VMAP_MESSAGE):
        torch.func.vmap(jacrev(f))(stacked)


def test_list_built_inside_jacrev_of_vmap_raises() -> None:
    structure = _molecule()
    stacked = torch.stack([structure.positions, structure.positions])

    def f(positions: Tensor) -> Tensor:
        build_neighborlist(structure.replace(positions=positions), CUTOFF)
        return positions.sum()

    def batched(positions: Tensor) -> Tensor:
        return torch.func.vmap(f)(positions).sum()

    with pytest.raises(RuntimeError, match=VMAP_MESSAGE):
        jacrev(batched)(stacked)


########################################################################
# compile


@requires_compile
@pytest.mark.usefixtures("reset_dynamo")
def test_list_built_inside_compiled_function() -> None:
    """The build is data-dependent, so Dynamo runs it eagerly (a graph
    break); it must still give the same list."""
    structure = _molecule()
    expected = build_neighborlist(structure, CUTOFF)

    def f(positions: Tensor) -> NeighborList:
        return build_neighborlist(
            structure.replace(positions=positions), CUTOFF
        )

    built = torch.compile(f)(structure.positions)

    _assert_same_list(built, expected)


@requires_compile
@pytest.mark.usefixtures("reset_dynamo")
def test_list_built_inside_fullgraph_compile_raises() -> None:
    """The build cannot be one graph; Dynamo must say so and not return
    anything."""
    structure = _molecule()

    def f(positions: Tensor) -> Tensor:
        return build_neighborlist(
            structure.replace(positions=positions), CUTOFF
        ).idx_i

    with pytest.raises(getattr(torch._dynamo, "exc").Unsupported):
        torch.compile(f, fullgraph=True)(structure.positions)


# Not `torch.compile(jacrev(f))`: PyTorch (2.4 and 2.10 alike) returns an
# all-zero gradient for it whenever `f` contains a graph break of any kind,
# build or not. `grad` is unaffected.
@requires_compile
@pytest.mark.usefixtures("reset_dynamo")
def test_list_built_inside_compiled_grad() -> None:
    structure = _molecule()
    expected = build_neighborlist(structure, CUTOFF)

    def f(positions: Tensor) -> Tensor:
        built = build_neighborlist(
            structure.replace(positions=positions), CUTOFF
        )
        assert torch.equal(built.idx_i, expected.idx_i)
        return (positions**2).sum()

    out = torch.compile(torch.func.grad(f))(structure.positions)
    assert torch.allclose(out, 2 * structure.positions)


@requires_compile
@pytest.mark.usefixtures("reset_dynamo")
def test_list_built_inside_jacrev_of_compiled_function() -> None:
    structure = _molecule()
    expected = build_neighborlist(structure, CUTOFF)

    def f(positions: Tensor) -> Tensor:
        built = build_neighborlist(
            structure.replace(positions=positions), CUTOFF
        )
        assert torch.equal(built.idx_i, expected.idx_i)
        return (positions**2).sum()

    out = jacrev(torch.compile(f))(structure.positions)
    assert torch.allclose(out, 2 * structure.positions)


########################################################################
# forward mode and Hessians
#
# `jacfwd` is `vmap` over `jvp`, but it batches only the tangents; the
# primal positions the build reads are not batched, so the build works.


def _check_list_built_inside(transform: str) -> None:
    structure = _molecule()
    expected = build_neighborlist(structure, CUTOFF)
    seen: list[NeighborList] = []

    def f(positions: Tensor) -> Tensor:
        seen.append(
            build_neighborlist(structure.replace(positions=positions), CUTOFF)
        )
        return (positions**2).sum()

    out = getattr(torch.func, transform)(f)(structure.positions)

    assert len(seen) >= 1
    for built in seen:
        _assert_same_list(built, expected)
    assert torch.isfinite(out).all()


def test_list_built_inside_jacfwd() -> None:
    _check_list_built_inside("jacfwd")


def test_list_built_inside_hessian() -> None:
    _check_list_built_inside("hessian")


def test_list_built_inside_jvp() -> None:
    structure = _molecule()
    expected = build_neighborlist(structure, CUTOFF)

    def f(positions: Tensor) -> Tensor:
        built = build_neighborlist(
            structure.replace(positions=positions), CUTOFF
        )
        _assert_same_list(built, expected)
        return (positions**2).sum()

    positions = structure.positions
    jvp: Any = torch.func.jvp
    _, tangent = jvp(f, (positions,), (torch.ones_like(positions),))

    assert torch.allclose(tangent, 2 * positions.sum())


########################################################################
# what crosses the boundary


def _check_only_integers_cross_the_boundary(structure: Structure) -> None:
    """The list is a constant to the transform it was built in. That is
    right for indices and masks only: a floating point tensor computed from
    the positions or the lattice would lose its gradient."""
    seen: list[NeighborList] = []

    def f(positions: Tensor) -> Tensor:
        seen.append(
            build_neighborlist(structure.replace(positions=positions), CUTOFF)
        )
        return positions.sum()

    jacrev(f)(structure.positions)

    built = seen[0]
    assert built.idx_i.dtype == torch.long
    assert built.idx_j.dtype == torch.long
    assert built.shift.dtype == torch.int16
    assert built.mask.dtype == torch.bool
    if built.periodic_axes is not None:
        assert built.periodic_axes.dtype == torch.bool

    for name in ("idx_i", "idx_j", "shift", "mask", "periodic_axes"):
        tensor = getattr(built, name)
        if tensor is not None:
            assert not tensor.is_floating_point()
            assert not getattr(
                torch._C, "_functorch"
            ).is_functorch_wrapped_tensor(tensor)


def test_only_integers_cross_the_boundary_molecule() -> None:
    _check_only_integers_cross_the_boundary(_molecule())


def test_only_integers_cross_the_boundary_cell() -> None:
    _check_only_integers_cross_the_boundary(_triclinic())


def test_list_does_not_alias_the_periodic_axes_it_was_built_from() -> None:
    structure = _cell()
    assert structure.periodic is not None
    seen: list[NeighborList] = []

    def f(positions: Tensor) -> Tensor:
        seen.append(
            build_neighborlist(structure.replace(positions=positions), CUTOFF)
        )
        return positions.sum()

    jacrev(f)(structure.positions)

    axes = seen[0].periodic_axes
    assert axes is not None
    assert axes.data_ptr() != structure.periodic.data_ptr()
    assert torch.equal(axes, structure.periodic)


########################################################################
# against finite differences, not against another build


def test_gradient_wrt_positions_matches_finite_differences_molecule() -> None:
    structure = _molecule()

    def f(positions: Tensor) -> Tensor:
        target = structure.replace(positions=positions)
        return cn_d3(target, pairs=build_neighborlist(target, CUTOFF))

    assert jacrev_matches_finite_diff(f, structure.positions)


def test_gradient_wrt_positions_matches_finite_differences_cell() -> None:
    structure = _triclinic()

    def f(positions: Tensor) -> Tensor:
        target = structure.replace(positions=positions)
        return cn_d3(target, pairs=build_neighborlist(target, CUTOFF))

    assert jacrev_matches_finite_diff(f, structure.positions)


def test_gradient_wrt_lattice_matches_finite_differences_cell() -> None:
    """The shifts of the list are integers; the Cartesian images are
    `shift @ lattice`, formed from the lattice the transform differentiates.
    A list that carried them as constants would fail here."""
    structure = _triclinic()
    assert structure.lattice is not None

    def f(lattice: Tensor) -> Tensor:
        target = structure.replace(lattice=lattice)
        return cn_d3(target, pairs=build_neighborlist(target, CUTOFF))

    # off the loaded cell, where an image sits exactly on the cutoff and
    # the CN is a step (see `test_ncoord/test_transforms.py`)
    lattice = 1.013 * structure.lattice

    jac = jacrev(f)(lattice)
    assert jac.abs().sum() > 0
    assert jacrev_matches_finite_diff(f, lattice)


########################################################################
# the layers come back


def test_layers_come_back_after_a_failed_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A build that raises inside the popped region must not leave the
    stack popped: a later transform has to work."""
    structure = _molecule()

    def broken(*args: object, **kwargs: object) -> object:
        raise ValueError("search failed")

    monkeypatch.setattr(nblist, "_build_single_neighborlists", broken)

    def f(positions: Tensor) -> Tensor:
        build_neighborlist(structure.replace(positions=positions), CUTOFF)
        return positions.sum()

    with pytest.raises(ValueError, match="search failed"):
        jacrev(jacrev(f))(structure.positions)
    assert getattr(torch._C, "_functorch").peek_interpreter_stack() is None

    monkeypatch.undo()

    out = jacrev(lambda p: (p**2).sum())(structure.positions)
    assert torch.allclose(out, 2 * structure.positions)


def test_vmap_error_leaves_the_stack_alone() -> None:
    """The `vmap` check comes before anything is popped."""
    structure = _molecule()
    depth: list[int] = []

    def f(positions: Tensor) -> Tensor:
        depth.append(
            getattr(torch._C, "_functorch").peek_interpreter_stack().level()
        )
        build_neighborlist(structure.replace(positions=positions), CUTOFF)
        return positions.sum()

    stacked = torch.stack([structure.positions, structure.positions])
    with pytest.raises(RuntimeError, match=VMAP_MESSAGE):
        torch.func.vmap(f)(stacked)

    assert depth == [1]


def test_private_functorch_symbols_exist() -> None:
    """What the build sets the layers aside with. If a PyTorch release moves
    them, this fails, not just the build."""
    assert nblist._CAN_POP_LAYERS  # pylint: disable=protected-access
    for name in (
        "peek_interpreter_stack",
        "pop_dynamic_layer_stack",
        "push_dynamic_layer_stack",
    ):
        assert callable(getattr(getattr(torch._C, "_functorch"), name))
