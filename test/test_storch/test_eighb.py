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
Tests taken from TBMaLT.
https://github.com/tbmalt/tbmalt/blob/development/tests/unittests/test_maths.py
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pytest
import torch

# required for generalized eigenvalue problem
from scipy import linalg

from tad_mctc import storch
from tad_mctc.autograd import dgradcheck
from tad_mctc.batch import pack
from tad_mctc.convert import numpy_to_tensor, symmetrizef, tensor_to_numpy
from tad_mctc.typing import DD, Tensor

from ..conftest import DEVICE, FAST_MODE
from ..utils import _rng, _symrng


def clean_zero_padding(m: Tensor, sizes: Tensor) -> Tensor:
    """Removes perturbations induced in the zero padding values by gradcheck.

    When performing gradient stability tests via PyTorch's gradcheck function
    small perturbations are induced in the input data. However, problems are
    encountered when these perturbations occur in the padding values. These
    values should always be zero, and so the test is not truly representative.
    Furthermore, this can even prevent certain tests from running. Thus this
    function serves to remove such perturbations in a gradient safe manner.

    Note that this is intended to operate on 3D matrices where. Specifically a
    batch of square matrices.

    Arguments:
        m (torch.Tensor):
            The tensor whose padding is to be cleaned.
        sizes (torch.Tensor):
            The true sizes of the tensors.

    Returns:
        cleaned (torch.Tensor):
            Cleaned tensor.

    Notes:
        This is only intended for 2D matrices packed into a 3D tensor.
    """

    # Identify the device
    device = m.device

    # First identify the maximum tensor size
    max_size = int(torch.max(sizes))

    # Build a mask that is True anywhere that the tensor should be zero, i.e.
    # True for regions of the tensor that should be zero padded.
    mask_1d = (
        (torch.arange(max_size, device=device) - sizes.unsqueeze(1)) >= 0
    ).repeat(max_size, 1, 1)

    # This, rather round about, approach to generating and applying the masks
    # must be used as some PyTorch operations like masked_scatter do not seem
    # to function correctly
    mask_full = torch.zeros(*m.shape, device=device).bool()
    mask_full[mask_1d.permute(1, 2, 0)] = True
    mask_full[mask_1d.transpose(0, 1)] = True

    # Create and apply the subtraction mask
    temp = torch.zeros_like(m, device=device)
    temp[mask_full] = m[mask_full]
    cleaned = m - temp

    return cleaned


def _spectral_density(w: Tensor, v: Tensor) -> Tensor:
    r"""
    Contract eigenvalues and eigenvectors into a density-matrix-like quantity.

    The eigenvectors themselves are *not* a well-defined function of the input
    matrix: they are only determined up to the sign of each column, and within
    a degenerate subspace up to an arbitrary rotation. Which representative
    LAPACK returns is not a continuous function of the input either, i.e. an
    infinitesimal perturbation can flip the sign of a whole eigenvector or
    re-mix a degenerate subspace. `gradcheck` approximates the derivative by
    central differences with a step size of 1e-6, so such a jump shows up as a
    numerical derivative on the order of 1e6 while the analytical gradient is
    perfectly fine. As the choice depends on the LAPACK/BLAS implementation and
    its internal blocking, this only ever surfaced sporadically on CI machines
    and was practically impossible to reproduce locally.

    The zero-padded batches are particularly susceptible because `eighb` pads
    the diagonal with an estimate of the largest eigenvalue, which creates an
    exactly degenerate subspace (one dimension per padded row) whose basis is
    completely arbitrary.

    Therefore, the gradients are not tested for the eigenvectors themselves but
    for

    .. math:: P = V f(\Lambda) V^{T}

    with the Fermi-like occupation :math:`f(\lambda) = 1 / (1 + e^{\lambda})`.
    This is the quantity that eigenvectors are actually used for (a density
    matrix), it is invariant under both sign changes and rotations within a
    degenerate subspace, and it is a smooth function of the input matrix. Since
    `f` is injective, no information about the eigenspaces is lost.

    Arguments:
        w (Tensor):
            The eigenvalues.
        v (Tensor):
            The eigenvectors, stored column-wise.

    Returns:
        p (Tensor):
            Density-matrix-like contraction of eigenvalues and eigenvectors.
    """
    occupation = torch.sigmoid(-w)
    return v @ torch.diag_embed(occupation) @ v.transpose(-1, -2)


def _metric_rng(size: int, dd: DD) -> Tensor:
    """
    Create a random, diagonal, positive definite metric matrix.

    The diagonal entries are drawn from ``[0.5, 1.5)`` instead of ``[0, 1)``.
    Entries close to zero render the generalized eigenvalue problem
    ``Az = λBz`` ill-conditioned: the associated eigenvalues (and their
    eigenvectors) grow without bound and react very sensitively to
    perturbations of the inputs. This invalidates the absolute tolerances of
    the accuracy tests (deviations of more than 1e-11 were observed for
    diagonal entries on the order of 1e-5) and it makes the eigenvector sign
    flips described in `_fix_eigenvector_sign` far more likely in the gradient
    tests.

    Since the test inputs are drawn from the global RNG, whose state depends on
    the test order and on the distribution across `xdist` workers, such an
    ill-conditioned instance only appeared sporadically, and thus practically
    only in CI. Bounding the diagonal away from zero limits the condition
    number of the metric matrix to three and removes this failure mode without
    weakening the tests.

    Arguments:
        size (int):
            Number of rows/columns of the matrix.
        dd (DD):
            Device and dtype of the matrix.

    Returns:
        b (Tensor):
            Random diagonal positive definite matrix.
    """
    return symmetrizef(torch.eye(size, **dd) * (_rng((size,), dd) + 0.5))


def test_eighb_fail() -> None:
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    a = symmetrizef(numpy_to_tensor(np.random.rand(10, 10), **dd))
    with pytest.raises(ValueError):
        storch.linalg.eighb(a, broadening_method="unknown")  # type: ignore

    with pytest.raises(ValueError):
        storch.linalg.eighb(a, b=a, scheme="unknown")  # type: ignore

    l_inv = storch.linalg.inv_cholesky_factor(_metric_rng(10, dd))
    with pytest.raises(ValueError):
        storch.linalg.eighb(a, b=a, l_inv=l_inv)

    with pytest.raises(ValueError):
        storch.linalg.eighb(a, l_inv=l_inv, scheme="lowd")


def test_eighb_standard_single() -> None:
    """eighb accuracy on a single standard eigenvalue problem."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    for _ in range(10):
        a = _symrng((10, 10), dd)

        w_ref = linalg.eigh(tensor_to_numpy(a))[0]
        w_ref = numpy_to_tensor(w_ref, **dd)

        factor = torch.tensor(1e-12, **dd)
        w_calc, v_calc = storch.linalg.eighb(a, factor=factor, aux=False)

        mae_w = torch.max(torch.abs(w_calc - w_ref))
        mae_v = torch.max(torch.abs((v_calc @ v_calc.T).fill_diagonal_(0)))

        dev_str = torch.device("cpu") if DEVICE is None else DEVICE
        same_device = w_calc.device == dev_str == v_calc.device

        assert mae_w < 1e-12, "Eigenvalue tolerance test"
        assert mae_v < 1e-12, "Eigenvector orthogonality test"
        assert same_device, "Device persistence check"


def test_eighb_standard_batch() -> None:
    """eighb accuracy on a batch of standard eigenvalue problems."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    for _ in range(10):
        sizes = np.random.randint(2, 10, (11,))
        a = [_symrng((s, s), dd) for s in sizes]
        a_batch = pack(a)

        w_ref = pack(
            [
                numpy_to_tensor(linalg.eigh(tensor_to_numpy(i))[0], **dd)
                for i in a
            ]
        )

        w_calc = storch.linalg.eighb(a_batch)[0]

        mae_w = torch.max(torch.abs(w_calc - w_ref))
        assert mae_w < 1e-12, "Eigenvalue tolerance test"

        dev_str = torch.device("cpu") if DEVICE is None else DEVICE
        same_device = w_calc.device == dev_str
        assert same_device, "Device persistence check"


def test_eighb_general_single() -> None:
    """eighb accuracy on a single general eigenvalue problem."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    for _ in range(10):
        a = _symrng((10, 10), dd)
        b = _metric_rng(10, dd)

        w_ref = linalg.eigh(tensor_to_numpy(a), tensor_to_numpy(b))[0]
        w_ref = numpy_to_tensor(w_ref, **dd)

        schemes: list[Literal["chol", "lowd"]] = ["chol", "lowd"]
        for scheme in schemes:
            w_calc, v_calc = storch.linalg.eighb(a, b, scheme=scheme)

            mae_w = torch.max(torch.abs(w_calc - w_ref))
            mae_v = torch.max(torch.abs((v_calc @ v_calc.T).fill_diagonal_(0)))

            dev_str = torch.device("cpu") if DEVICE is None else DEVICE
            same_device = w_calc.device == dev_str == v_calc.device

            assert mae_w < 1e-11, f"Eigenvalue tolerance test {scheme}"
            assert mae_v < 1e-11, f"Eigenvector orthogonality test {scheme}"
            assert same_device, "Device persistence check"


def test_eighb_general_batch() -> None:
    """eighb accuracy on a batch of general eigenvalue problems."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    for _ in range(10):
        sizes = np.random.randint(2, 10, (11,))
        a = [_symrng((s, s), dd) for s in sizes]
        b = [_metric_rng(s, dd) for s in sizes]
        a_batch, b_batch = pack(a), pack(b)

        w_ref = pack(
            [
                numpy_to_tensor(
                    linalg.eigh(tensor_to_numpy(i), tensor_to_numpy(j))[0], **dd
                )
                for i, j in zip(a, b)
            ]
        )

        is_zero = torch.eq(b_batch, 0)
        mask = torch.all(is_zero, dim=-1) & torch.all(is_zero, dim=-2)
        b_batch = b_batch + torch.diag_embed(mask.type(b_batch.dtype))

        aux_settings = [True, False]
        schemes: list[Literal["chol", "lowd"]] = ["chol", "lowd"]
        for scheme in schemes:
            for aux in aux_settings:
                w_calc, _ = storch.linalg.eighb(
                    a_batch, b_batch, scheme=scheme, aux=aux, is_posdef=True
                )

                mae_w = torch.max(torch.abs(w_calc - w_ref))

                dev_str = torch.device("cpu") if DEVICE is None else DEVICE
                same_device = w_calc.device == dev_str

                assert mae_w < 1e-10, f"Eigenvalue tolerance test {scheme}"
                assert same_device, "Device persistence check"


def test_eighb_l_inv_single() -> None:
    """eighb with a precomputed inverse Cholesky factor on a single problem."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    for _ in range(10):
        a = _symrng((10, 10), dd)
        b = _metric_rng(10, dd)

        w_ref = linalg.eigh(tensor_to_numpy(a), tensor_to_numpy(b))[0]
        w_ref = numpy_to_tensor(w_ref, **dd)

        w_b, v_b = storch.linalg.eighb(a, b)

        l_inv = storch.linalg.inv_cholesky_factor(b)
        w_calc, v_calc = storch.linalg.eighb(a, l_inv=l_inv)

        assert torch.max(torch.abs(w_calc - w_ref)) < 1e-11
        assert torch.max(torch.abs(w_calc - w_b)) < 1e-14
        assert torch.max(torch.abs(v_calc - v_b)) < 1e-14


def test_eighb_l_inv_batch() -> None:
    """eighb with a precomputed inverse Cholesky factor on a padded batch."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    for _ in range(10):
        sizes = np.random.randint(2, 10, (11,))
        a = [_symrng((s, s), dd) for s in sizes]
        b = [_metric_rng(s, dd) for s in sizes]
        a_batch, b_batch = pack(a), pack(b)

        w_ref = pack(
            [
                numpy_to_tensor(
                    linalg.eigh(tensor_to_numpy(i), tensor_to_numpy(j))[0], **dd
                )
                for i, j in zip(a, b)
            ]
        )

        # zero-padded metric, identity-padded inside `inv_cholesky_factor`
        l_inv = storch.linalg.inv_cholesky_factor(b_batch)

        for aux in [True, False]:
            w_b, v_b = storch.linalg.eighb(a_batch, b_batch, aux=aux)
            w_calc, v_calc = storch.linalg.eighb(a_batch, l_inv=l_inv, aux=aux)

            assert torch.max(torch.abs(w_calc - w_ref)) < 1e-10
            assert torch.max(torch.abs(w_calc - w_b)) < 1e-14
            assert torch.max(torch.abs(v_calc - v_b)) < 1e-14


def test_eighb_general_batch_multithreaded() -> None:
    """
    Batched Cholesky scheme must not use LU factorisation.

    On CPU with more than one thread, batched LU factorisation (as used by
    `torch.linalg.solve` and `torch.inverse`) returns corrupt pivots or
    deadlocks for matrices of size ~150 and above (pytorch/pytorch#142815).
    """
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    size = 200

    a = [_symrng((size, size), dd) for _ in range(4)]
    b = []
    for _ in range(4):
        x = _rng((size, size), dd)
        b.append(x @ x.mT / size + torch.eye(size, **dd))

    w_ref = torch.stack(
        [
            numpy_to_tensor(
                linalg.eigh(tensor_to_numpy(i), tensor_to_numpy(j))[0], **dd
            )
            for i, j in zip(a, b)
        ]
    )

    nthreads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        w_calc, _ = storch.linalg.eighb(
            torch.stack(a), torch.stack(b), scheme="chol", is_posdef=True
        )
    finally:
        torch.set_num_threads(nthreads)

    assert torch.max(torch.abs(w_calc - w_ref)) < 1e-10


################################################################################


def _eigen_proxy(
    m: Tensor,
    target_method: Literal["cond", "lorn"] | None,
    size_data: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    m = symmetrizef(m)
    if size_data is not None:
        m = clean_zero_padding(m, size_data)
    if target_method is None:
        w, v = torch.linalg.eigh(m)
    else:
        w, v = storch.linalg.eighb(m, broadening_method=target_method)

    return w, _spectral_density(w, v)


@pytest.mark.grad
@pytest.mark.parametrize("bmethod", [None, "cond", "lorn"])
def test_eighb_broadening_grad(bmethod: Literal["cond", "lorn"] | None) -> None:
    """
    eighb gradient stability on standard, broadened, eigenvalue problems.

    There is no separate test for the standard eigenvalue problem without
    broadening as this would result in a direct call to torch.symeig which is
    unnecessary. However, it is important to note that conditional broadening
    technically is never tested, i.e. the lines:

    .. code-block:: python
        ...
        if ctx.bm == 'cond':  # <- Conditional broadening
            deltas = 1 / torch.where(torch.abs(deltas) > bf,
                                     deltas, bf) * torch.sign(deltas)
        ...

    of `_SymEigB` are never actual run. This is because it only activates when
    there are true eigen-value degeneracies; & degenerate eigenvalue problems
    do not "play well" with the gradcheck operation.
    """
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    # Generate a single standard eigenvalue test instance
    a1 = _symrng((8, 8), dd)
    a1.requires_grad = True

    assert dgradcheck(
        lambda a2, method_l=bmethod: _eigen_proxy(
            a2,
            target_method=method_l,  # pyright: ignore[reportArgumentType]
        ),
        (a1,),
        fast_mode=FAST_MODE,
    ), f"Non-degenerate single test failed on {bmethod}"


@pytest.mark.grad
@pytest.mark.parametrize("bmethod", ["cond", "lorn"])
def test_eighb_broadening_grad_batch(bmethod: Literal["cond", "lorn"]) -> None:
    """
    eighb gradient stability on standard, broadened, eigenvalue problems.

    There is no separate test for the standard eigenvalue problem without
    broadening as this would result in a direct call to torch.symeig which is
    unnecessary. However, it is important to note that conditional broadening
    technically is never tested, i.e. the lines:

    .. code-block:: python
        ...
        if ctx.bm == 'cond':  # <- Conditional broadening
            deltas = 1 / torch.where(torch.abs(deltas) > bf,
                                     deltas, bf) * torch.sign(deltas)
        ...

    of `_SymEigB` are never actual run. This is because it only activates when
    there are true eigen-value degeneracies; & degenerate eigenvalue problems
    do not "play well" with the gradcheck operation.
    """
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    # Generate a batch of standard eigenvalue test instances
    sizes = np.random.randint(3, 8, (5,))
    a2 = pack([_symrng((s, s), dd) for s in sizes])
    a2.requires_grad = True

    assert dgradcheck(
        lambda a2_l, method_l=bmethod: _eigen_proxy(
            a2_l,
            target_method=method_l,  # pyright: ignore[reportArgumentType]
            size_data=numpy_to_tensor(sizes, **dd),
        ),
        (a2,),
        fast_mode=FAST_MODE,
    ), f"Non-degenerate batch test failed on {bmethod}"


################################################################################


def _eigen_proxy_general(
    m: Tensor,
    n: Tensor,
    target_scheme: Literal["chol", "lowd"],
    size_data: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    m, n = symmetrizef(m), symmetrizef(n)
    if size_data is not None:
        m = clean_zero_padding(m, size_data)
        n = clean_zero_padding(n, size_data)

    factor = torch.tensor(1e-12, device=m.device, dtype=m.dtype)
    w, v = storch.linalg.eighb(m, n, scheme=target_scheme, factor=factor)

    return w, _spectral_density(w, v)


def _eigen_proxy_l_inv(
    m: Tensor, n: Tensor, size_data: Tensor | None = None
) -> tuple[Tensor, Tensor]:
    m, n = symmetrizef(m), symmetrizef(n)
    if size_data is not None:
        m = clean_zero_padding(m, size_data)
        n = clean_zero_padding(n, size_data)

    factor = torch.tensor(1e-12, device=m.device, dtype=m.dtype)
    l_inv = storch.linalg.inv_cholesky_factor(n)
    w, v = storch.linalg.eighb(m, l_inv=l_inv, factor=factor)

    return w, _spectral_density(w, v)


@pytest.mark.grad
def test_eighb_l_inv_grad() -> None:
    """eighb gradient stability through a precomputed Cholesky factor."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    a1 = _symrng((8, 8), dd)
    b1 = _metric_rng(8, dd)

    a1.requires_grad, b1.requires_grad = True, True

    assert dgradcheck(_eigen_proxy_l_inv, (a1, b1), fast_mode=False)


@pytest.mark.grad
def test_eighb_l_inv_grad_batch() -> None:
    """eighb gradient stability through a precomputed Cholesky factor."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    sizes = np.random.randint(3, 8, (5,))
    a2 = pack([_symrng((s, s), dd) for s in sizes])
    b2 = pack([_metric_rng(s, dd) for s in sizes])

    a2.requires_grad, b2.requires_grad = True, True

    assert dgradcheck(
        _eigen_proxy_l_inv,
        (a2, b2, numpy_to_tensor(sizes, **dd)),
        fast_mode=False,
    )


@pytest.mark.grad
@pytest.mark.parametrize("scheme", ["chol", "lowd"])
def test_eighb_general_grad(scheme: Literal["chol", "lowd"]) -> None:
    """eighb gradient stability on general eigenvalue problems."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    # Generate a single generalised eigenvalue test instance
    a1 = _symrng((8, 8), dd)
    b1 = _metric_rng(8, dd)

    a1.requires_grad, b1.requires_grad = True, True

    # dgradcheck only takes tensors, but: Loop variable capture of lambda
    assert dgradcheck(
        lambda a1_l, b1_l, scheme_l=scheme: _eigen_proxy_general(
            a1_l,
            b1_l,
            target_scheme=scheme_l,  # pyright: ignore[reportArgumentType]
        ),
        (a1, b1),
        fast_mode=False,
    ), f"Non-degenerate single test failed on {scheme}"


@pytest.mark.grad
@pytest.mark.parametrize("scheme", ["chol", "lowd"])
def test_eighb_general_grad_batch(scheme: Literal["chol", "lowd"]) -> None:
    """eighb gradient stability on general eigenvalue problems."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    # Generate a batch of generalised eigenvalue test instances
    sizes = np.random.randint(3, 8, (5,))
    a2 = pack([_symrng((s, s), dd) for s in sizes])
    b2 = pack([_metric_rng(s, dd) for s in sizes])

    a2.requires_grad, b2.requires_grad = True, True

    assert dgradcheck(
        lambda a2_l, b2_l, size_data_l, scheme_l=scheme: _eigen_proxy_general(
            a2_l,
            b2_l,
            size_data=size_data_l,
            target_scheme=scheme_l,  # pyright: ignore[reportArgumentType]
        ),
        (a2, b2, numpy_to_tensor(sizes, **dd)),
        fast_mode=False,
    ), f"Non-degenerate batch test failed on {scheme}"


def test_eig_sort_out_auxiliary() -> None:
    """Auxiliary eigenvalues (one) are zeroed and moved to the end."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    w = torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]], **dd)
    v = torch.eye(3, **dd).expand(2, 3, 3).clone()

    w_out, v_out = storch.linalg._eig_sort_out(w, v, ghost=False)

    # only the first system has an auxiliary eigenvalue
    w_ref = torch.tensor([[2.0, 3.0, 0.0], [2.0, 3.0, 4.0]], **dd)
    v_ref = v.clone()
    v_ref[0] = v[0][:, [1, 2, 0]]

    torch.testing.assert_close(w_out, w_ref)
    torch.testing.assert_close(v_out, v_ref)


def test_eighb_no_sort_out() -> None:
    """Without `sort_out`, the eigenvalues are returned as computed."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    a = _symrng((10, 10), dd)
    w_ref = torch.linalg.eigvalsh(a)

    w, _ = storch.linalg.eighb(a, sort_out=False, aux=False)
    torch.testing.assert_close(w, w_ref)


def test_eighb_forward_mode_tensor_factor() -> None:
    """Forward mode with the broadening factor given as a tensor."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    a = _symrng((5, 5), dd)
    factor = torch.tensor(1e-12, **dd)

    def f(x: Tensor) -> Tensor:
        return storch.linalg.eighb(x, factor=factor, aux=False)[0]

    fwd = torch.func.jacfwd(f)(a)
    rev = torch.func.jacrev(f)(a)
    torch.testing.assert_close(fwd, rev, atol=1e-8, rtol=1e-8)
