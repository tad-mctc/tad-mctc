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
Function transforms across the library: ``vmap``, ``jacrev``, ``jacfwd`` and
``torch.compile(fullgraph=True)``.

Every tensor-in/tensor-out function below is checked, in its own test, for

- ``vmap`` over a leading batch equals a Python loop,
- ``jacfwd`` equals ``jacrev`` (and ``jacrev`` equals finite differences
  where the function is smooth enough for that to be meaningful),
- ``torch.compile(fullgraph=True)`` equals eager, also with ``jacrev``
  applied on the outside of the compiled function.

Not covered because the *shape* of the output depends on input *values*, so
neither ``vmap`` nor ``fullgraph`` can represent them. Use the alternative:

- ``neighbor.images.count_image_rings_*``, ``build_periodic_shifts``,
  ``build_ghost_pool``: build the table once, eagerly, with
  ``build_shared_periodic_shifts`` and pass it to
  ``CNModel.with_precomputed_shifts``.
- ``batch.pack``/``unpack``/``deflate``, ``properties.sum_formula``,
  ``io.*``, ``convert.*_to_*`` string/NumPy converters: host-side
  bookkeeping, not tensor math.
- ``storch.linalg.eighb(..., sort_out=True, aux=False)``: the ghost sort uses
  NumPy and ``torch.where(mask)``; the default ``aux=True`` path is traceable.
"""

from __future__ import annotations

import itertools

import pytest
import torch
from torch.func import jacfwd, jacrev, vmap

from tad_mctc.batch import psort
from tad_mctc.batch.mask import zero_masked_pairs
from tad_mctc.convert import reshape_fortran, symmetrize
from tad_mctc.data import getters
from tad_mctc.io.structure import Structure
from tad_mctc.ncoord import (
    cn_d3,
    cn_d4,
    cn_eeq,
    cn_eeqbc,
    cn_gfn2,
    cut_coordination_number,
)
from tad_mctc.ncoord.count import erf_count, exp_count, gfn2_count
from tad_mctc.neighbor.images import (
    build_shared_periodic_shifts,
    wrap_to_central_cell,
)
from tad_mctc.properties import (
    bond_angles,
    center_of_mass,
    enn,
    guess_bond_length,
    guess_bond_order,
    inertia_moment,
    rot_consts,
)
from tad_mctc.properties.general import positions_rel_com
from tad_mctc.storch import (
    cdist,
    eighb,
    safe_divide,
    safe_pow,
    safe_reciprocal,
    safe_sqrt,
)
from tad_mctc.typing import DD, Callable, Tensor

from ..conftest import DEVICE
from ..utils import (
    DYNAMO_SUPPORTED,
    DYNAMO_UNSUPPORTED_REASON,
    run_compiled_or_skip,
)

dd: DD = {"device": DEVICE, "dtype": torch.double}


BATCH = 3
NAT = 5
NUMBERS = torch.tensor([6, 1, 1, 8, 1], device=DEVICE)


def _positions() -> Tensor:
    torch.manual_seed(7)
    return torch.randn(BATCH, NAT, 3, **dd) * 2.0


def _positive(*shape: int) -> Tensor:
    torch.manual_seed(8)
    return torch.rand(*shape, **dd) + 0.5


def _symmetric(n: int) -> Tensor:
    torch.manual_seed(9)
    x = torch.randn(BATCH, n, n, **dd)
    return x + x.mT + 10 * torch.eye(n, **dd)


def _spd(n: int) -> Tensor:
    torch.manual_seed(10)
    x = torch.randn(BATCH, n, n, **dd)
    return x @ x.mT / n + torch.eye(n, **dd)


def _lattice() -> Tensor:
    torch.manual_seed(11)
    lat = torch.eye(3, **dd) * 8.0 + torch.randn(BATCH, 3, 3, **dd) * 0.1
    return lat


def _assert_vmap(f: Callable[..., Tensor], *batched: Tensor) -> None:
    """``vmap(f)`` over the leading dimension equals a loop."""
    vmapped = vmap(f)(*batched)
    looped = torch.stack([f(*sample) for sample in zip(*batched)])
    assert torch.allclose(vmapped, looped, atol=1e-10, rtol=1e-10)


def _assert_jac(f: Callable[..., Tensor], *args: Tensor, argnums: int = 0) -> None:
    """``jacfwd`` equals ``jacrev``, in eager mode and under ``vmap``."""
    fwd = jacfwd(f, argnums=argnums)(*args)
    rev = jacrev(f, argnums=argnums)(*args)
    assert torch.allclose(fwd, rev, atol=1e-8, rtol=1e-8)

    vfwd = vmap(jacfwd(f, argnums=argnums))(*[a.unsqueeze(0) for a in args])
    vrev = vmap(jacrev(f, argnums=argnums))(*[a.unsqueeze(0) for a in args])
    assert torch.allclose(vfwd, vrev, atol=1e-8, rtol=1e-8)
    assert torch.allclose(vrev[0], rev, atol=1e-8, rtol=1e-8)


def _assert_fd(f: Callable[[Tensor], Tensor], x: Tensor) -> None:
    """``jacrev`` equals central finite differences."""
    jacobian = jacrev(f)(x)
    numeric = torch.zeros_like(jacobian)

    for idx in itertools.product(*(range(s) for s in x.shape)):
        shifted = x.clone()
        shifted[idx] += 1e-6
        plus = f(shifted)
        shifted[idx] -= 2e-6
        minus = f(shifted)
        numeric[(..., *idx)] = (plus - minus) / 2e-6

    assert torch.allclose(jacobian, numeric, atol=1e-5, rtol=1e-5)


def _assert_compile(f: Callable[..., Tensor], *args: Tensor) -> None:
    """Compiled value equals eager value, ``fullgraph=True``."""
    torch._dynamo.reset()
    compiled = run_compiled_or_skip(f, *args)
    assert torch.allclose(compiled, f(*args), atol=1e-10, rtol=1e-10)


def _assert_compile_jacrev(f: Callable[..., Tensor], *args: Tensor) -> None:
    """``torch.compile(jacrev(f))`` equals eager ``jacrev(f)``."""
    torch._dynamo.reset()
    compiled = run_compiled_or_skip(jacrev(f), *args)
    assert torch.allclose(compiled, jacrev(f)(*args), atol=1e-8, rtol=1e-8)


# ---------------------------------------------------------------------------
# storch: distances and safe elementwise operations
# ---------------------------------------------------------------------------


def test_cdist_transforms() -> None:
    x = _positions()

    def f(p: Tensor) -> Tensor:
        # the diagonal is `sqrt(eps)` noise; compare off-diagonal only
        d = cdist(p)
        return d * (1 - torch.eye(NAT, **dd))

    _assert_vmap(f, x)
    _assert_jac(f, x[0])
    _assert_fd(f, x[0])


@pytest.mark.skipif(not DYNAMO_SUPPORTED, reason=DYNAMO_UNSUPPORTED_REASON)
def test_cdist_compile() -> None:
    x = _positions()[0]

    def f(p: Tensor) -> Tensor:
        return cdist(p) * (1 - torch.eye(NAT, **dd))

    _assert_compile(f, x)
    _assert_compile_jacrev(f, x)


def test_safe_divide_transforms() -> None:
    a, b = _positive(BATCH, NAT), _positive(BATCH, NAT)
    _assert_vmap(safe_divide, a, b)
    _assert_jac(safe_divide, a[0], b[0], argnums=0)
    _assert_jac(safe_divide, a[0], b[0], argnums=1)


@pytest.mark.skipif(not DYNAMO_SUPPORTED, reason=DYNAMO_UNSUPPORTED_REASON)
def test_safe_divide_compile() -> None:
    a, b = _positive(BATCH, NAT), _positive(BATCH, NAT)
    _assert_compile(safe_divide, a[0], b[0])
    _assert_compile_jacrev(safe_divide, a[0], b[0])


def test_safe_reciprocal_transforms() -> None:
    x = _positive(BATCH, NAT)
    _assert_vmap(safe_reciprocal, x)
    _assert_jac(safe_reciprocal, x[0])


@pytest.mark.skipif(not DYNAMO_SUPPORTED, reason=DYNAMO_UNSUPPORTED_REASON)
def test_safe_reciprocal_compile() -> None:
    x = _positive(BATCH, NAT)[0]
    _assert_compile(safe_reciprocal, x)
    _assert_compile_jacrev(safe_reciprocal, x)


def test_safe_pow_transforms() -> None:
    x = _positive(BATCH, NAT)

    def f_float(a: Tensor) -> Tensor:
        return safe_pow(a, 1.5)

    def f_int_valued_float(a: Tensor) -> Tensor:
        return safe_pow(a, -2.0)

    def f_int(a: Tensor) -> Tensor:
        return safe_pow(a, 3)

    _assert_vmap(f_float, x)
    _assert_vmap(f_int_valued_float, x)
    _assert_vmap(f_int, x)
    _assert_jac(f_float, x[0])
    _assert_jac(f_int_valued_float, x[0])
    _assert_jac(f_int, x[0])


@pytest.mark.skipif(not DYNAMO_SUPPORTED, reason=DYNAMO_UNSUPPORTED_REASON)
def test_safe_pow_compile() -> None:
    x = _positive(BATCH, NAT)[0]

    def f_float(a: Tensor) -> Tensor:
        return safe_pow(a, 1.5)

    def f_int_valued_float(a: Tensor) -> Tensor:
        # `float.is_integer` is not traceable by older Dynamo versions
        return safe_pow(a, -2.0)

    _assert_compile(f_float, x)
    _assert_compile(f_int_valued_float, x)
    _assert_compile_jacrev(f_float, x)


def test_safe_sqrt_transforms() -> None:
    x = _positive(BATCH, NAT)
    _assert_vmap(safe_sqrt, x)
    _assert_jac(safe_sqrt, x[0])


@pytest.mark.skipif(not DYNAMO_SUPPORTED, reason=DYNAMO_UNSUPPORTED_REASON)
def test_safe_sqrt_compile() -> None:
    x = _positive(BATCH, NAT)[0]
    _assert_compile(safe_sqrt, x)
    _assert_compile_jacrev(safe_sqrt, x)


# ---------------------------------------------------------------------------
# storch.linalg.eighb: used to be a custom `autograd.Function` without a
# vmap rule or a `jvp`
# ---------------------------------------------------------------------------


def test_eighb_eigenvalues_transforms() -> None:
    a = _symmetric(NAT)

    def f(m: Tensor) -> Tensor:
        return eighb(m)[0]

    _assert_vmap(f, a)
    _assert_jac(f, a[0])
    _assert_fd(lambda m: f(0.5 * (m + m.mT)), a[0])


def test_eighb_eigenvectors_transforms() -> None:
    a = _symmetric(NAT)

    def f(m: Tensor) -> Tensor:
        # squared: invariant to the arbitrary sign of an eigenvector
        return eighb(m)[1] ** 2

    _assert_vmap(f, a)
    _assert_jac(f, a[0])
    _assert_fd(lambda m: f(0.5 * (m + m.mT)), a[0])


def test_eighb_second_derivatives_match_torch_eigh() -> None:
    """
    The Hessian through `eighb` equals the Hessian through
    ``torch.linalg.eigh`` where broadening is inactive (non-degenerate
    spectrum), for reverse-over-reverse and forward-over-reverse.
    """
    a = _symmetric(NAT)[0]

    def reference(m: Tensor) -> Tensor:
        w, v = torch.linalg.eigh(0.5 * (m + m.mT))
        return w[1] + (v[:, 1] ** 2).sum() * w[2]

    def f(m: Tensor) -> Tensor:
        w, v = eighb(0.5 * (m + m.mT))
        return w[1] + (v[:, 1] ** 2).sum() * w[2]

    expected = jacrev(jacrev(reference))(a)
    assert expected.abs().max() > 1e-3

    assert torch.allclose(jacrev(jacrev(f))(a), expected, atol=1e-8)
    assert torch.allclose(jacfwd(jacrev(f))(a), expected, atol=1e-8)
    assert torch.allclose(jacrev(jacfwd(f))(a), expected, atol=1e-8)


def test_eighb_lorentzian_transforms() -> None:
    a = _symmetric(NAT)

    def f(m: Tensor) -> Tensor:
        w, v = eighb(m, broadening_method="lorn")
        return torch.cat([w, (v**2).flatten()])

    _assert_vmap(f, a)
    _assert_jac(f, a[0])


def test_eighb_without_broadening_transforms() -> None:
    a = _symmetric(NAT)

    def f(m: Tensor) -> Tensor:
        return eighb(m, broadening_method=None)[0]

    _assert_vmap(f, a)
    _assert_jac(f, a[0])


@pytest.mark.parametrize("scheme", ["chol", "lowd"])
def test_eighb_generalised_transforms(scheme: str) -> None:
    a, b = _symmetric(NAT), _spd(NAT)

    def f(m: Tensor, s: Tensor) -> Tensor:
        # Cholesky and `eigh` read one triangle only, so their forward-mode
        # derivative of a non-symmetric input differs from the symmetrized
        # reverse-mode gradient by PyTorch's own convention; symmetrize here
        # to compare the same function.
        m, s = 0.5 * (m + m.mT), 0.5 * (s + s.mT)
        return eighb(m, s, scheme=scheme)[0]  # type: ignore[arg-type]

    _assert_vmap(f, a, b)
    _assert_jac(f, a[0], b[0], argnums=0)
    _assert_jac(f, a[0], b[0], argnums=1)


@pytest.mark.skipif(not DYNAMO_SUPPORTED, reason=DYNAMO_UNSUPPORTED_REASON)
def test_eighb_compile() -> None:
    a = _symmetric(NAT)[0]

    def f(m: Tensor) -> Tensor:
        return eighb(m)[0]

    def g(m: Tensor) -> Tensor:
        return eighb(m)[1] ** 2

    _assert_compile(f, a)
    _assert_compile(g, a)
    # `compile(jacrev(f))` is not covered: Dynamo rejects an autograd
    # `Function` that defines a `jvp` ("Unsupported custom jvp"), which
    # `eighb` needs for `jacfwd`. `jacrev(compile(f))` is not supported by
    # PyTorch either.


@pytest.mark.skipif(not DYNAMO_SUPPORTED, reason=DYNAMO_UNSUPPORTED_REASON)
def test_eighb_generalised_compile() -> None:
    a, b = _symmetric(NAT)[0], _spd(NAT)[0]

    def f(m: Tensor, s: Tensor) -> Tensor:
        return eighb(m, s)[0]

    _assert_compile(f, a, b)


def test_eighb_unknown_broadening_raises() -> None:
    with pytest.raises(ValueError, match="Unknown broadening method"):
        eighb(_symmetric(NAT)[0], broadening_method="nope")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# properties
# ---------------------------------------------------------------------------


def test_enn_transforms() -> None:
    p = _positions()

    def f(x: Tensor) -> Tensor:
        return enn(NUMBERS, x)

    _assert_vmap(f, p)
    _assert_jac(f, p[0])
    _assert_fd(f, p[0])


@pytest.mark.skipif(not DYNAMO_SUPPORTED, reason=DYNAMO_UNSUPPORTED_REASON)
def test_enn_compile() -> None:
    p = _positions()[0]

    def f(x: Tensor) -> Tensor:
        return enn(NUMBERS, x)

    _assert_compile(f, p)
    _assert_compile_jacrev(f, p)


def test_center_of_mass_transforms() -> None:
    m, p = _positive(BATCH, NAT), _positions()
    _assert_vmap(center_of_mass, m, p)
    _assert_jac(center_of_mass, m[0], p[0], argnums=1)


@pytest.mark.skipif(not DYNAMO_SUPPORTED, reason=DYNAMO_UNSUPPORTED_REASON)
def test_center_of_mass_compile() -> None:
    m, p = _positive(BATCH, NAT)[0], _positions()[0]
    _assert_compile(center_of_mass, m, p)
    _assert_compile_jacrev(lambda x: center_of_mass(m, x), p)


def test_positions_rel_com_transforms() -> None:
    m, p = _positive(BATCH, NAT), _positions()
    _assert_vmap(positions_rel_com, m, p)
    _assert_jac(positions_rel_com, m[0], p[0], argnums=1)


@pytest.mark.skipif(not DYNAMO_SUPPORTED, reason=DYNAMO_UNSUPPORTED_REASON)
def test_positions_rel_com_compile() -> None:
    m, p = _positive(BATCH, NAT)[0], _positions()[0]
    _assert_compile(positions_rel_com, m, p)


def test_inertia_moment_transforms() -> None:
    m, p = _positive(BATCH, NAT), _positions()
    _assert_vmap(inertia_moment, m, p)
    _assert_jac(inertia_moment, m[0], p[0], argnums=1)


@pytest.mark.skipif(not DYNAMO_SUPPORTED, reason=DYNAMO_UNSUPPORTED_REASON)
def test_inertia_moment_compile() -> None:
    m, p = _positive(BATCH, NAT)[0], _positions()[0]
    _assert_compile(inertia_moment, m, p)


def test_rot_consts_transforms() -> None:
    m, p = _positive(BATCH, NAT), _positions()
    _assert_vmap(rot_consts, m, p)
    _assert_jac(rot_consts, m[0], p[0], argnums=1)


@pytest.mark.skipif(not DYNAMO_SUPPORTED, reason=DYNAMO_UNSUPPORTED_REASON)
def test_rot_consts_compile() -> None:
    m, p = _positive(BATCH, NAT)[0], _positions()[0]
    _assert_compile(rot_consts, m, p)


def test_bond_angles_transforms() -> None:
    p = _positions()

    def f(x: Tensor) -> Tensor:
        return bond_angles(NUMBERS, x)

    _assert_vmap(f, p)
    _assert_jac(f, p[0])


@pytest.mark.skipif(not DYNAMO_SUPPORTED, reason=DYNAMO_UNSUPPORTED_REASON)
def test_bond_angles_compile() -> None:
    p = _positions()[0]

    def f(x: Tensor) -> Tensor:
        return bond_angles(NUMBERS, x)

    _assert_compile(f, p)


def test_guess_bond_length_transforms() -> None:
    cn = _positive(BATCH, NAT)

    def f(c: Tensor) -> Tensor:
        return guess_bond_length(NUMBERS, c)

    _assert_vmap(f, cn)
    _assert_jac(f, cn[0])


@pytest.mark.skipif(not DYNAMO_SUPPORTED, reason=DYNAMO_UNSUPPORTED_REASON)
def test_guess_bond_length_compile() -> None:
    cn = _positive(BATCH, NAT)[0]

    def f(c: Tensor) -> Tensor:
        return guess_bond_length(NUMBERS, c)

    _assert_compile(f, cn)
    _assert_compile_jacrev(f, cn)


def test_guess_bond_order_transforms() -> None:
    p, cn = _positions(), _positive(BATCH, NAT)

    def f(x: Tensor, c: Tensor) -> Tensor:
        return guess_bond_order(NUMBERS, x, c)

    _assert_vmap(f, p, cn)
    _assert_jac(f, p[0], cn[0], argnums=0)
    _assert_jac(f, p[0], cn[0], argnums=1)


@pytest.mark.skipif(not DYNAMO_SUPPORTED, reason=DYNAMO_UNSUPPORTED_REASON)
def test_guess_bond_order_compile() -> None:
    p, cn = _positions()[0], _positive(BATCH, NAT)[0]

    def f(x: Tensor, c: Tensor) -> Tensor:
        return guess_bond_order(NUMBERS, x, c)

    _assert_compile(f, p, cn)


# ---------------------------------------------------------------------------
# batch / convert / data
# ---------------------------------------------------------------------------


def test_zero_masked_pairs_transforms() -> None:
    t = torch.randn(BATCH, NAT, NAT, 3, **dd)

    def f(x: Tensor) -> Tensor:
        return zero_masked_pairs(NUMBERS, x)

    _assert_vmap(f, t)
    _assert_jac(f, t[0])


@pytest.mark.skipif(not DYNAMO_SUPPORTED, reason=DYNAMO_UNSUPPORTED_REASON)
def test_zero_masked_pairs_compile() -> None:
    t = torch.randn(BATCH, NAT, NAT, 3, **dd)[0]

    def f(x: Tensor) -> Tensor:
        return zero_masked_pairs(NUMBERS, x)

    _assert_compile(f, t)
    _assert_compile_jacrev(f, t)


def test_psort_transforms() -> None:
    x = _positive(BATCH, NAT)

    def f(t: Tensor) -> Tensor:
        return psort(t).values

    _assert_vmap(f, x)
    _assert_jac(f, x[0])


@pytest.mark.skipif(not DYNAMO_SUPPORTED, reason=DYNAMO_UNSUPPORTED_REASON)
def test_psort_compile() -> None:
    x = _positive(BATCH, NAT)[0]

    def f(t: Tensor) -> Tensor:
        return psort(t).values

    _assert_compile(f, x)


def test_reshape_fortran_transforms() -> None:
    x = _positive(BATCH, NAT, 3)

    def f(t: Tensor) -> Tensor:
        return reshape_fortran(t, (3, NAT))

    _assert_vmap(f, x)
    _assert_jac(f, x[0])


@pytest.mark.skipif(not DYNAMO_SUPPORTED, reason=DYNAMO_UNSUPPORTED_REASON)
def test_reshape_fortran_compile() -> None:
    x = _positive(BATCH, NAT, 3)[0]

    def f(t: Tensor) -> Tensor:
        return reshape_fortran(t, (3, NAT))

    _assert_compile(f, x)


def test_symmetrize_transforms() -> None:
    x = _positive(BATCH, NAT, NAT)

    def f(t: Tensor) -> Tensor:
        return symmetrize(t, force=True)

    _assert_vmap(f, x)
    _assert_jac(f, x[0])


@pytest.mark.skipif(not DYNAMO_SUPPORTED, reason=DYNAMO_UNSUPPORTED_REASON)
def test_symmetrize_compile() -> None:
    x = _positive(BATCH, NAT, NAT)[0]

    def f(t: Tensor) -> Tensor:
        return symmetrize(t, force=True)

    _assert_compile(f, x)


def test_get_vdw_pairwise_vmap() -> None:
    numbers = NUMBERS.repeat(BATCH, 1)
    numbers[1, 0] = 7

    vmapped = vmap(lambda n: getters.get_vdw_pairwise(n, device=DEVICE))(numbers)
    looped = torch.stack([getters.get_vdw_pairwise(n, device=DEVICE) for n in numbers])
    assert torch.equal(vmapped, looped)


@pytest.mark.skipif(not DYNAMO_SUPPORTED, reason=DYNAMO_UNSUPPORTED_REASON)
def test_get_vdw_pairwise_compile() -> None:
    """The table is read from disk once, at import, so a first call that
    happens inside a trace does not do file I/O."""

    def f(n: Tensor) -> Tensor:
        return getters.get_vdw_pairwise(n, device=DEVICE)

    _assert_compile(f, NUMBERS)


def test_get_vdw_pairwise_returns_a_copy() -> None:
    first = getters.get_vdw_pairwise(NUMBERS, device=DEVICE)
    first.zero_()
    assert (getters.get_vdw_pairwise(NUMBERS, device=DEVICE) != 0).any()


# ---------------------------------------------------------------------------
# ncoord
# ---------------------------------------------------------------------------


def test_exp_count_transforms() -> None:
    r, r0 = _positive(BATCH, NAT, NAT), _positive(BATCH, NAT, NAT)
    _assert_vmap(exp_count, r, r0)
    _assert_jac(exp_count, r[0], r0[0], argnums=0)
    _assert_jac(exp_count, r[0], r0[0], argnums=1)


@pytest.mark.skipif(not DYNAMO_SUPPORTED, reason=DYNAMO_UNSUPPORTED_REASON)
def test_exp_count_compile() -> None:
    r, r0 = _positive(BATCH, NAT, NAT)[0], _positive(BATCH, NAT, NAT)[0]
    _assert_compile(exp_count, r, r0)
    _assert_compile_jacrev(exp_count, r, r0)


def test_erf_count_transforms() -> None:
    r, r0 = _positive(BATCH, NAT, NAT), _positive(BATCH, NAT, NAT)
    _assert_vmap(erf_count, r, r0)
    _assert_jac(erf_count, r[0], r0[0], argnums=0)
    _assert_jac(erf_count, r[0], r0[0], argnums=1)


@pytest.mark.skipif(not DYNAMO_SUPPORTED, reason=DYNAMO_UNSUPPORTED_REASON)
def test_erf_count_compile() -> None:
    r, r0 = _positive(BATCH, NAT, NAT)[0], _positive(BATCH, NAT, NAT)[0]
    _assert_compile(erf_count, r, r0)
    _assert_compile_jacrev(erf_count, r, r0)


def test_gfn2_count_transforms() -> None:
    r, r0 = _positive(BATCH, NAT, NAT), _positive(BATCH, NAT, NAT)
    _assert_vmap(gfn2_count, r, r0)
    _assert_jac(gfn2_count, r[0], r0[0], argnums=0)
    _assert_jac(gfn2_count, r[0], r0[0], argnums=1)


@pytest.mark.skipif(not DYNAMO_SUPPORTED, reason=DYNAMO_UNSUPPORTED_REASON)
def test_gfn2_count_compile() -> None:
    r, r0 = _positive(BATCH, NAT, NAT)[0], _positive(BATCH, NAT, NAT)[0]
    _assert_compile(gfn2_count, r, r0)
    _assert_compile_jacrev(gfn2_count, r, r0)


def test_cut_coordination_number_transforms() -> None:
    cn = _positive(BATCH, NAT) * 4.0

    def f_float(c: Tensor) -> Tensor:
        return cut_coordination_number(c, 8.0)

    def f_tensor(c: Tensor, cn_max: Tensor) -> Tensor:
        return cut_coordination_number(c, cn_max)

    cn_max = torch.full((BATCH,), 8.0, **dd)

    _assert_vmap(f_float, cn)
    _assert_vmap(f_tensor, cn, cn_max)
    _assert_jac(f_float, cn[0])
    _assert_jac(f_tensor, cn[0], cn_max[0], argnums=0)
    _assert_jac(f_tensor, cn[0], cn_max[0], argnums=1)


@pytest.mark.skipif(not DYNAMO_SUPPORTED, reason=DYNAMO_UNSUPPORTED_REASON)
def test_cut_coordination_number_compile() -> None:
    cn = _positive(BATCH, NAT)[0] * 4.0
    cn_max = torch.tensor(8.0, **dd)

    def f_float(c: Tensor) -> Tensor:
        return cut_coordination_number(c, 8.0)

    def f_tensor(c: Tensor, m: Tensor) -> Tensor:
        return cut_coordination_number(c, m)

    _assert_compile(f_float, cn)
    _assert_compile(f_tensor, cn, cn_max)
    _assert_compile_jacrev(f_float, cn)


def test_cn_d3_transforms() -> None:
    p = _positions()

    def f(x: Tensor) -> Tensor:
        return cn_d3(Structure(numbers=NUMBERS, positions=x))

    _assert_vmap(f, p)
    _assert_jac(f, p[0])
    _assert_fd(f, p[0])


@pytest.mark.skipif(not DYNAMO_SUPPORTED, reason=DYNAMO_UNSUPPORTED_REASON)
def test_cn_d3_compile() -> None:
    p = _positions()[0]

    def f(x: Tensor) -> Tensor:
        return cn_d3(Structure(numbers=NUMBERS, positions=x))

    _assert_compile(f, p)
    _assert_compile_jacrev(f, p)


def test_cn_d4_transforms() -> None:
    p = _positions()

    def f(x: Tensor) -> Tensor:
        return cn_d4(Structure(numbers=NUMBERS, positions=x))

    _assert_vmap(f, p)
    _assert_jac(f, p[0])


@pytest.mark.skipif(not DYNAMO_SUPPORTED, reason=DYNAMO_UNSUPPORTED_REASON)
def test_cn_d4_compile() -> None:
    p = _positions()[0]

    def f(x: Tensor) -> Tensor:
        return cn_d4(Structure(numbers=NUMBERS, positions=x))

    _assert_compile(f, p)
    _assert_compile_jacrev(f, p)


def test_cn_gfn2_transforms() -> None:
    p = _positions()

    def f(x: Tensor) -> Tensor:
        return cn_gfn2(Structure(numbers=NUMBERS, positions=x))

    _assert_vmap(f, p)
    _assert_jac(f, p[0])


@pytest.mark.skipif(not DYNAMO_SUPPORTED, reason=DYNAMO_UNSUPPORTED_REASON)
def test_cn_gfn2_compile() -> None:
    p = _positions()[0]

    def f(x: Tensor) -> Tensor:
        return cn_gfn2(Structure(numbers=NUMBERS, positions=x))

    _assert_compile(f, p)
    _assert_compile_jacrev(f, p)


def test_cn_eeq_transforms() -> None:
    p = _positions()

    def f(x: Tensor) -> Tensor:
        return cn_eeq(Structure(numbers=NUMBERS, positions=x))

    _assert_vmap(f, p)
    _assert_jac(f, p[0])


@pytest.mark.skipif(not DYNAMO_SUPPORTED, reason=DYNAMO_UNSUPPORTED_REASON)
def test_cn_eeq_compile() -> None:
    p = _positions()[0]

    def f(x: Tensor) -> Tensor:
        return cn_eeq(Structure(numbers=NUMBERS, positions=x))

    _assert_compile(f, p)
    _assert_compile_jacrev(f, p)


def test_cn_eeqbc_transforms() -> None:
    p = _positions()

    def f(x: Tensor) -> Tensor:
        return cn_eeqbc(Structure(numbers=NUMBERS, positions=x))

    _assert_vmap(f, p)
    _assert_jac(f, p[0])


@pytest.mark.skipif(not DYNAMO_SUPPORTED, reason=DYNAMO_UNSUPPORTED_REASON)
def test_cn_eeqbc_compile() -> None:
    p = _positions()[0]

    def f(x: Tensor) -> Tensor:
        return cn_eeqbc(Structure(numbers=NUMBERS, positions=x))

    _assert_compile(f, p)


def test_cn_eeq_tensor_cn_max_transforms() -> None:
    p = _positions()
    cn_max = torch.full((BATCH,), 8.0, **dd)

    def f(x: Tensor, m: Tensor) -> Tensor:
        model = cn_eeq.replace(cn_max=m)
        return model(Structure(numbers=NUMBERS, positions=x))

    _assert_vmap(f, p, cn_max)
    _assert_jac(f, p[0], cn_max[0], argnums=1)


@pytest.mark.skipif(not DYNAMO_SUPPORTED, reason=DYNAMO_UNSUPPORTED_REASON)
def test_cn_eeq_tensor_cn_max_compile() -> None:
    p = _positions()[0]
    m = torch.tensor(8.0, **dd)

    def f(x: Tensor, cn_max: Tensor) -> Tensor:
        model = cn_eeq.replace(cn_max=cn_max)
        return model(Structure(numbers=NUMBERS, positions=x))

    # `dataclasses.replace` is skipped by older Dynamo versions
    _assert_compile(f, p, m)


@pytest.mark.skipif(not DYNAMO_SUPPORTED, reason=DYNAMO_UNSUPPORTED_REASON)
def test_cn_batched_compile() -> None:
    """A real leading batch dimension (not ``vmap``) under ``fullgraph``."""
    p = _positions()
    numbers = NUMBERS.repeat(BATCH, 1)

    def f(x: Tensor) -> Tensor:
        return cn_d3(Structure(numbers=numbers, positions=x))

    _assert_compile(f, p)
    _assert_compile_jacrev(f, p)


# ---------------------------------------------------------------------------
# periodic CN with a precomputed shift table, and the wrap
# ---------------------------------------------------------------------------

PERIODIC = torch.tensor([True, True, True], device=DEVICE)


def test_cn_d3_periodic_precomputed_transforms() -> None:
    p, lat = _positions(), _lattice()
    shifts = build_shared_periodic_shifts(lat, PERIODIC, 40.0)

    def f(x: Tensor, l: Tensor) -> Tensor:
        s = Structure(numbers=NUMBERS, positions=x, lattice=l, periodic=PERIODIC)
        return cn_d3.with_precomputed_shifts(s, shifts=shifts)

    _assert_vmap(f, p, lat)
    _assert_jac(f, p[0], lat[0], argnums=0)
    _assert_jac(f, p[0], lat[0], argnums=1)


@pytest.mark.skipif(not DYNAMO_SUPPORTED, reason=DYNAMO_UNSUPPORTED_REASON)
def test_cn_d3_periodic_precomputed_compile() -> None:
    p, lat = _positions(), _lattice()
    shifts = build_shared_periodic_shifts(lat, PERIODIC, 40.0)

    def f(x: Tensor, l: Tensor) -> Tensor:
        s = Structure(numbers=NUMBERS, positions=x, lattice=l, periodic=PERIODIC)
        return cn_d3.with_precomputed_shifts(s, shifts=shifts)

    _assert_compile(f, p[0], lat[0])
    _assert_compile_jacrev(f, p[0], lat[0])


def test_wrap_to_central_cell_transforms() -> None:
    p, lat = _positions(), _lattice()

    def f(x: Tensor, l: Tensor) -> Tensor:
        return wrap_to_central_cell(x, l, PERIODIC)[0]

    _assert_vmap(f, p, lat)
    _assert_jac(f, p[0], lat[0], argnums=0)


@pytest.mark.skipif(not DYNAMO_SUPPORTED, reason=DYNAMO_UNSUPPORTED_REASON)
def test_wrap_to_central_cell_compile() -> None:
    p, lat = _positions()[0], _lattice()[0]

    def f(x: Tensor, l: Tensor) -> Tensor:
        return wrap_to_central_cell(x, l, PERIODIC)[0]

    _assert_compile(f, p, lat)
