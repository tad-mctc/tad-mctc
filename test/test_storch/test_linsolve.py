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
Test the iterative linear solvers (forward solution and diagnostics).

Every solution is compared with a dense direct solve, and every status with
the true residual. Derivatives are tested in `test_linsolve_implicit.py`.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import pytest
import torch
from torch.func import vmap

from tad_mctc.exceptions import ConvergenceError, DeviceError, DtypeError
from tad_mctc.storch import linsolve
from tad_mctc.storch.linsolve import (
    STATUS_BREAKDOWN,
    STATUS_CONVERGED,
    STATUS_NONFINITE,
    STATUS_NOT_CONVERGED,
    STATUS_STAGNATED,
    LinearSolveResult,
    bicgstab,
    cg,
    gmres,
    require_convergence,
)
from tad_mctc.typing import DD, Tensor

from ..conftest import DEVICE

SOLVERS = {"cg": cg, "bicgstab": bicgstab, "gmres": gmres}

Solver = Callable[..., LinearSolveResult]


def matvec(x: Tensor, a: Tensor) -> Tensor:
    return torch.einsum("...ij,...j->...i", a, x)


def jacobi(r: Tensor, a: Tensor) -> Tensor:
    return r / a.diagonal(dim1=-2, dim2=-1)


def spd(n: int, dd: DD, batch: tuple[int, ...] = (), seed: int = 0) -> Tensor:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    s = torch.randn(
        *batch, n, n, generator=gen, dtype=torch.double, device="cpu"
    )
    a = s @ s.mT + n * torch.eye(n, dtype=torch.double, device="cpu")
    return a.to(**dd)


def general(
    n: int, dd: DD, batch: tuple[int, ...] = (), seed: int = 0
) -> Tensor:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    s = torch.randn(
        *batch, n, n, generator=gen, dtype=torch.double, device="cpu"
    )
    return (
        s + 2 * n**0.5 * torch.eye(n, dtype=torch.double, device="cpu")
    ).to(**dd)


def rhs(shape: tuple[int, ...], dd: DD, seed: int = 1) -> Tensor:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(
        *shape, generator=gen, dtype=torch.double, device="cpu"
    ).to(**dd)


def matrix(name: str, n: int, dd: DD, batch: tuple[int, ...] = ()) -> Tensor:
    return spd(n, dd, batch) if name == "cg" else general(n, dd, batch)


def check_result(result: LinearSolveResult, a: Tensor, b: Tensor) -> None:
    """The reported residual is the true residual of the solution."""
    true = torch.linalg.vector_norm(b - matvec(result.solution, a), dim=-1)
    assert (
        pytest.approx(true.cpu(), abs=1e-14, rel=1e-6) == result.residual.cpu()
    )
    converged = result.residual <= result.target
    assert (converged == result.converged).all()


@pytest.mark.parametrize("name", SOLVERS)
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_single(name: str, dtype: torch.dtype) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}
    a, b = matrix(name, 8, dd), rhs((8,), dd)

    result = SOLVERS[name](matvec, b, a)
    ref = torch.linalg.solve(a, b)

    tol = 1e-5 if dtype == torch.float else 1e-10
    assert pytest.approx(ref.cpu(), abs=tol) == result.solution.cpu()
    assert result.converged.item()
    assert result.status.dtype == torch.int64
    assert result.iterations.shape == ()
    assert 0 < result.iterations.item() <= 16
    check_result(result, a, b)


@pytest.mark.parametrize("name", SOLVERS)
def test_batch(name: str) -> None:
    """Independent systems with different iteration counts."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    a = matrix(name, 10, dd, batch=(2, 3))
    a[0, 1] = torch.eye(10, **dd) * 2.0  # converges in one iteration
    b = rhs((2, 3, 10), dd)

    result = SOLVERS[name](matvec, b, a)
    ref = torch.linalg.solve(a, b)

    assert pytest.approx(ref.cpu(), abs=1e-10) == result.solution.cpu()
    assert result.status.shape == (2, 3)
    assert result.converged.all()
    assert result.iterations[0, 1] == 1
    assert (result.iterations[0, 1] < result.iterations[1, 1]).all()
    check_result(result, a, b)


@pytest.mark.parametrize("name", SOLVERS)
def test_batch_matches_loop(name: str) -> None:
    """Batched results equal the results of the single systems."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    a, b = matrix(name, 6, dd, batch=(4,)), rhs((4, 6), dd)

    result = SOLVERS[name](matvec, b, a)
    for i in range(4):
        single = SOLVERS[name](matvec, b[i], a[i])
        assert pytest.approx(single.solution.cpu(), abs=1e-13) == (
            result.solution[i].cpu()
        )
        assert single.iterations == result.iterations[i]


@pytest.mark.parametrize("name", SOLVERS)
def test_multiple_rhs(name: str) -> None:
    """Several right-hand sides of one matrix as a batch dimension."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    a, b = matrix(name, 7, dd), rhs((3, 7), dd)

    result = SOLVERS[name](matvec, b, a)  # `a` broadcasts over the rhs
    ref = torch.linalg.solve(a, b.mT).mT

    assert pytest.approx(ref.cpu(), abs=1e-10) == result.solution.cpu()
    assert result.status.shape == (3,)


@pytest.mark.parametrize("name", SOLVERS)
def test_vdims(name: str) -> None:
    """A matrix-valued unknown (`vdims=2`), e.g. a Fock-like state."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    a = matrix(name, 12, dd)
    b = rhs((2, 3, 4), dd)

    def op(x: Tensor, a: Tensor) -> Tensor:
        return matvec(x.flatten(-2), a).unflatten(-1, (3, 4))

    result = SOLVERS[name](op, b, a, vdims=2)
    ref = torch.linalg.solve(a, b.flatten(-2).mT).mT.unflatten(-1, (3, 4))

    assert pytest.approx(ref.cpu(), abs=1e-10) == result.solution.cpu()
    assert result.status.shape == (2,)
    assert result.converged.all()


@pytest.mark.parametrize("name", SOLVERS)
def test_zero_rhs(name: str) -> None:
    """A zero right-hand side converges without iterating, without NaN."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    a, b = matrix(name, 5, dd, batch=(2,)), rhs((2, 5), dd)
    b[1] = 0.0

    result = SOLVERS[name](matvec, b, a)

    assert result.converged.all()
    assert result.iterations[1] == 0
    assert (result.solution[1] == 0).all()
    assert torch.isfinite(result.solution).all()


@pytest.mark.parametrize("name", SOLVERS)
def test_converged_initial_guess(name: str) -> None:
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    a, b = matrix(name, 5, dd), rhs((5,), dd)
    x0 = torch.linalg.solve(a, b)

    result = SOLVERS[name](matvec, b, a, x0=x0)

    assert result.converged.item()
    assert result.iterations.item() == 0
    assert (result.solution == x0).all()


@pytest.mark.parametrize("name", SOLVERS)
def test_initial_guess(name: str) -> None:
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    a, b = matrix(name, 9, dd), rhs((9,), dd)
    ref = torch.linalg.solve(a, b)

    result = SOLVERS[name](matvec, b, a, x0=ref + 1e-3 * rhs((9,), dd, 5))
    cold = SOLVERS[name](matvec, b, a)

    assert pytest.approx(ref.cpu(), abs=1e-10) == result.solution.cpu()
    assert result.iterations.item() <= cold.iterations.item()


@pytest.mark.parametrize("name", SOLVERS)
def test_maxiter(name: str) -> None:
    """Hitting the iteration limit is not reported as convergence."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    a, b = matrix(name, 20, dd), rhs((20,), dd)

    result = SOLVERS[name](matvec, b, a, maxiter=2)
    assert result.status.item() == STATUS_NOT_CONVERGED
    assert result.iterations.item() == 2
    check_result(result, a, b)

    result = SOLVERS[name](matvec, b, a, maxiter=0)
    assert result.status.item() == STATUS_NOT_CONVERGED
    assert (result.solution == 0).all()


@pytest.mark.parametrize("name", SOLVERS)
def test_early_exit(name: str) -> None:
    """The loop exit policy does not change the result."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    a, b = matrix(name, 6, dd, batch=(3,)), rhs((3, 6), dd)

    early = SOLVERS[name](matvec, b, a, early_exit=True)
    full = SOLVERS[name](matvec, b, a, early_exit=False, maxiter=30)

    for x, y in zip(early, full):
        assert (x == y).all()


@pytest.mark.parametrize("name", SOLVERS)
def test_tolerance(name: str) -> None:
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    a, b = matrix(name, 30, dd), rhs((30,), dd)

    loose = SOLVERS[name](matvec, b, a, rtol=1e-3, atol=0.0)
    tight = SOLVERS[name](matvec, b, a, rtol=1e-12, atol=0.0)

    norm = torch.linalg.vector_norm(b)
    assert pytest.approx(1e-3 * norm.item()) == loose.target.item()
    assert loose.residual <= 1e-3 * norm
    assert tight.residual <= 1e-12 * norm
    assert loose.iterations < tight.iterations


def test_tolerance_floor() -> None:
    """The relative tolerance is never below 10 eps of the dtype."""
    dd: DD = {"device": DEVICE, "dtype": torch.float}
    a, b = spd(6, dd), rhs((6,), dd)

    result = cg(matvec, b, a, rtol=1e-14, atol=0.0)

    eps = torch.finfo(torch.float).eps
    norm = torch.linalg.vector_norm(b)
    assert pytest.approx(10 * eps * norm.item()) == result.target.item()
    assert result.converged.item()


def single_precision(
    condition: float, batch: tuple[int, ...] = (), seed: int = 3
) -> tuple[Tensor, Tensor]:
    """SPD system of size 40 in single precision with a given condition."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    cpu: DD = {"device": torch.device("cpu"), "dtype": torch.double}
    s = torch.randn(*batch, 40, 40, generator=gen, **cpu)
    q, _ = torch.linalg.qr(s)
    spectrum = torch.logspace(0, math.log10(condition), 40, **cpu)
    a = ((q * spectrum.unsqueeze(-2)) @ q.mT).float()
    b = torch.randn(*batch, 40, generator=gen, **cpu).float()
    return (0.5 * (a + a.mT)).to(DEVICE), b.to(DEVICE)


@pytest.mark.parametrize("name", ["cg", "bicgstab"])
def test_true_residual_check(name: str) -> None:
    """
    In single precision, the recursive residual drifts from the true one.
    A system whose recursive residual meets the target only stops if its
    true residual does, or if that stagnated (the tolerance of 10 eps is
    then below the attainable accuracy); it never stops early as
    "iteration limit reached".
    """
    a, b = single_precision(1e2, batch=(20,))

    result = SOLVERS[name](matvec, b, a, rtol=0.0, atol=0.0, maxiter=400)
    check_result(result, a, b)
    status = result.status.tolist()
    assert set(status) <= {STATUS_CONVERGED, STATUS_STAGNATED}
    assert STATUS_CONVERGED in status
    # stagnated just above the target, at the attainable accuracy
    assert (result.residual < 4 * result.target).all()

    # the same without stopping early (as under `vmap`)
    full = SOLVERS[name](
        matvec, b, a, rtol=0.0, atol=0.0, maxiter=400, early_exit=False
    )
    assert full.status.tolist() == status
    assert (full.solution == result.solution).all()


@pytest.mark.parametrize("name", ["cg", "bicgstab"])
def test_stagnation(name: str) -> None:
    """An unattainable tolerance stops long before the iteration limit."""
    a, b = single_precision(1e4)

    result = SOLVERS[name](matvec, b, a, rtol=0.0, atol=0.0, maxiter=4000)

    assert result.status.item() == STATUS_STAGNATED
    assert result.iterations.item() < 1000
    check_result(result, a, b)
    with pytest.raises(ConvergenceError, match="attainable accuracy"):
        require_convergence(result)


@pytest.mark.parametrize("name", ["cg", "bicgstab"])
def test_operator_applications(name: str) -> None:
    """
    The check of the true residual replaces its recomputation after the
    loop: no extra product for CG, one for BiCGSTAB (whose check trip
    applies the operator twice).
    """
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    a, b = matrix(name, 20, dd), rhs((20,), dd)
    calls = []

    def counted(x: Tensor, a: Tensor) -> Tensor:
        calls.append(1)
        return matvec(x, a)

    result = SOLVERS[name](counted, b, a)

    assert result.converged.item()
    # initial residual, iterations, and the check (the final product of
    # BiCGSTAB's check trip is part of the trip)
    per_iteration = 1 if name == "cg" else 2
    expected = 1 + per_iteration * (result.iterations.item() + 1)
    assert len(calls) == expected


@pytest.mark.parametrize("name", SOLVERS)
def test_preconditioner(name: str) -> None:
    """Jacobi preconditioning of a badly scaled system."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    scale = torch.logspace(0, 4, 40, **dd)
    a = matrix(name, 40, dd) * scale.unsqueeze(-1) * scale.unsqueeze(-2)
    b = rhs((40,), dd)
    ref = torch.linalg.solve(a, b)

    plain = SOLVERS[name](matvec, b, a, maxiter=400)
    prec = SOLVERS[name](matvec, b, a, preconditioner=jacobi, maxiter=400)

    assert prec.converged.item()
    assert prec.iterations < plain.iterations
    assert pytest.approx(ref.cpu(), rel=1e-6) == prec.solution.cpu()


def test_cg_indefinite() -> None:
    """CG stops at a direction of negative curvature."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    a = torch.diag(torch.tensor([1.0, 2.0, -3.0, 4.0], **dd))
    b = torch.ones(4, **dd)

    result = cg(matvec, b, a)

    assert result.status.item() == STATUS_BREAKDOWN
    assert not result.converged.item()
    assert torch.isfinite(result.solution).all()
    check_result(result, a, b)


def test_cg_bordered_eeq_like() -> None:
    """
    The bordered (KKT) matrix of a constrained SPD problem is indefinite.
    CG must not report it converged if it is not.
    """
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    a = spd(6, dd)
    c = torch.ones(6, 1, **dd)
    kkt = torch.cat(
        (torch.cat((a, c), -1), torch.cat((c.mT, torch.zeros(1, 1, **dd)), -1)),
        -2,
    )
    b = rhs((7,), dd)

    result = cg(matvec, b, kkt)

    if result.converged.item():  # pragma: no cover
        ref = torch.linalg.solve(kkt, b)
        assert pytest.approx(ref.cpu(), abs=1e-8) == result.solution.cpu()
    else:
        assert result.status.item() in (STATUS_BREAKDOWN, STATUS_NOT_CONVERGED)
    check_result(result, kkt, b)


def test_bicgstab_breakdown() -> None:
    """
    BiCGSTAB breaks down on a rotation: the shadow residual is orthogonal to
    the image of the first direction.
    """
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    a = torch.tensor([[0.0, 1.0], [-1.0, 0.0]], **dd)
    b = torch.tensor([1.0, 0.0], **dd)

    result = bicgstab(matvec, b, a)

    assert result.status.item() == STATUS_BREAKDOWN
    assert result.iterations.item() == 0
    assert torch.isfinite(result.solution).all()


@pytest.mark.parametrize("name", SOLVERS)
def test_singular(name: str) -> None:
    """An inconsistent singular system never reports convergence."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    a = torch.diag(torch.tensor([1.0, 2.0, 0.0], **dd))
    b = torch.ones(3, **dd)

    result = SOLVERS[name](matvec, b, a)

    assert not result.converged.item()
    assert result.status.item() in (
        STATUS_BREAKDOWN,
        STATUS_NOT_CONVERGED,
        STATUS_STAGNATED,
    )


@pytest.mark.parametrize("name", SOLVERS)
def test_nonfinite(name: str) -> None:
    """NaN in one system is reported, the other system is unaffected."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    a, b = matrix(name, 4, dd, batch=(2,)), rhs((2, 4), dd)
    a[0, 0, 0] = torch.nan

    result = SOLVERS[name](matvec, b, a)

    assert result.status[0].item() == STATUS_NONFINITE
    assert result.status[1].item() == STATUS_CONVERGED
    ref = torch.linalg.solve(a[1], b[1])
    assert pytest.approx(ref.cpu(), abs=1e-10) == result.solution[1].cpu()


@pytest.mark.parametrize("name", SOLVERS)
def test_vmap(name: str) -> None:
    """`vmap` with different convergence speeds and a failed system."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    a, b = matrix(name, 5, dd, batch=(3,)), rhs((3, 5), dd)
    a[1] = torch.eye(5, **dd)
    b[2, 0] = torch.nan

    def f(a: Tensor, b: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        result = SOLVERS[name](matvec, b, a)
        return result.solution, result.status, result.iterations

    x, status, iterations = vmap(f)(a, b)
    batched = SOLVERS[name](matvec, b, a)

    assert pytest.approx(batched.solution[:2].cpu(), abs=1e-12) == x[:2].cpu()
    assert (status == batched.status).all()
    assert (iterations == batched.iterations).all()
    assert status.tolist() == [STATUS_CONVERGED] * 2 + [STATUS_NONFINITE]


def test_result_is_pytree() -> None:
    """The result can be returned from a transform as a whole."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    a, b = spd(4, dd, batch=(2,)), rhs((2, 4), dd)

    result = vmap(lambda a, b: cg(matvec, b, a))(a, b)

    assert isinstance(result, LinearSolveResult)
    assert result.converged.shape == (2,)


def test_unrolled_gradient() -> None:
    """
    The plain solvers are ordinary PyTorch code: their gradient is the
    gradient of the executed iterations, which converges to the exact one.
    """
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    a, b = spd(6, dd), rhs((6,), dd).requires_grad_(True)

    (grad,) = torch.autograd.grad(cg(matvec, b, a).solution.sum(), b)
    ref = torch.linalg.solve(a, torch.ones(6, **dd))
    assert pytest.approx(ref.cpu(), abs=1e-8) == grad.cpu()


########################################################################
# Argument validation


def test_invalid_vdims() -> None:
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    a, b = spd(3, dd), rhs((3,), dd)

    with pytest.raises(ValueError, match="vector dimensions"):
        cg(matvec, b, a, vdims=0)
    with pytest.raises(ValueError, match="vector dimensions"):
        cg(matvec, b, a, vdims=2)


def test_invalid_x0() -> None:
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    a, b = spd(3, dd), rhs((3,), dd)

    with pytest.raises(ValueError, match="initial guess"):
        cg(matvec, b, a, x0=torch.zeros(4, **dd))


def test_invalid_operator() -> None:
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    a, b = spd(3, dd), rhs((3,), dd)

    with pytest.raises(ValueError, match="preserve the shape"):
        cg(lambda x, a: (a @ x).sum(), b, a)


def test_invalid_maxiter() -> None:
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    a, b = spd(3, dd), rhs((3,), dd)

    with pytest.raises(ValueError, match="negative"):
        bicgstab(matvec, b, a, maxiter=-1)


def test_exports() -> None:
    from tad_mctc import storch

    assert storch.cg is linsolve.cg
    assert storch.implicit_solve is linsolve.implicit_solve


def test_require_convergence() -> None:
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    a = torch.stack((torch.eye(3, **dd), torch.zeros(3, 3, **dd)))
    b = torch.ones(2, 3, **dd)

    good = cg(matvec, b[0], a[0])
    assert torch.equal(require_convergence(good), good.solution)

    failed = cg(matvec, b, a)
    with pytest.raises(ConvergenceError, match="1 of 2 system"):
        require_convergence(failed, name="Test solve")

    # under `vmap`: NaN for the failed system only
    def solve(a: Tensor, b: Tensor) -> Tensor:
        return require_convergence(cg(matvec, b, a))

    out = torch.vmap(solve)(a, b)
    assert pytest.approx(b[0].cpu()) == out[0].cpu()
    assert torch.isnan(out[1]).all()


def test_require_convergence_worst_system() -> None:
    """The message reports the residual and target of the same system."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    result = LinearSolveResult(
        solution=torch.zeros(3, 2, **dd),
        residual=torch.tensor([1.0, 5.0, 0.1], **dd),
        target=torch.tensor([2.0, 1.0, 0.01], **dd),
        iterations=torch.zeros(3, dtype=torch.int64, device=DEVICE),
        status=torch.tensor(
            [STATUS_CONVERGED, STATUS_NOT_CONVERGED, STATUS_STAGNATED],
            device=DEVICE,
        ),
    )

    with pytest.raises(ConvergenceError) as error:
        require_convergence(result)
    message = str(error.value)
    assert "2 of 3 system(s)" in message
    assert "iteration limit reached" in message
    assert "stagnation" in message
    assert "residual 1.000e-01, target 1.000e-02" in message


def test_gmres_invalid_restart() -> None:
    a, b = torch.eye(2, dtype=torch.double), torch.ones(2, dtype=torch.double)
    with pytest.raises(ValueError, match="restart"):
        gmres(matvec, b, a, restart=0)


@pytest.mark.parametrize("name", SOLVERS)
@pytest.mark.parametrize("dtype", [torch.cdouble, torch.int64])
def test_rejects_non_real(name: str, dtype: torch.dtype) -> None:
    a = torch.eye(2, dtype=torch.double)
    with pytest.raises(DtypeError, match="right-hand side"):
        SOLVERS[name](matvec, torch.ones(2, dtype=dtype), a)
    with pytest.raises(DtypeError, match="initial guess"):
        SOLVERS[name](
            matvec,
            torch.ones(2, dtype=torch.double),
            a,
            x0=torch.ones(2, dtype=dtype),
        )


@pytest.mark.parametrize("name", SOLVERS)
def test_rejects_complex_operator(name: str) -> None:
    b = torch.ones(2, dtype=torch.double)
    a = torch.eye(2, dtype=torch.cdouble)

    # as a parameter
    with pytest.raises(DtypeError, match="Parameter 0"):
        SOLVERS[name](lambda x, a: a @ x, b, a)

    # captured in a closure: the operator output is checked
    with pytest.raises(DtypeError, match="operator output"):
        SOLVERS[name](lambda x: a @ x.to(a.dtype), b)


@pytest.mark.parametrize("name", SOLVERS)
def test_rejects_mixed_precision_operator(name: str) -> None:
    b = torch.ones(2, dtype=torch.float)
    a = torch.eye(2, dtype=torch.double)
    with pytest.raises(DtypeError, match="operator output"):
        SOLVERS[name](lambda x: a @ x.double(), b)


@pytest.mark.parametrize("name", SOLVERS)
def test_integer_parameter(name: str) -> None:
    """Integer tensors are structural arguments, not rejected."""
    b = torch.ones(3, dtype=torch.double)
    k = torch.tensor(1)
    result = SOLVERS[name](lambda x, k: 2.0**k * x, b, k)
    assert pytest.approx([0.5, 0.5, 0.5]) == result.solution.tolist()


def test_rejects_operator_on_other_device() -> None:
    b = torch.ones(2, dtype=torch.double)
    with pytest.raises(DeviceError, match="operator output"):
        cg(lambda x: x.to("meta"), b)
