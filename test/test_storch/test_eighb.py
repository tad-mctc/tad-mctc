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

import numpy as np
import pytest
import torch

# required for generalized eigenvalue problem
from scipy import linalg

from tad_mctc import storch
from tad_mctc.autograd import dgradcheck
from tad_mctc.batch import pack
from tad_mctc.convert import numpy_to_tensor, symmetrizef, tensor_to_numpy
from tad_mctc.typing import DD, Literal, Tensor

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


def test_eighb_fail() -> None:
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    a = symmetrizef(numpy_to_tensor(np.random.rand(10, 10), **dd))
    with pytest.raises(ValueError):
        storch.linalg.eighb(a, broadening_method="unknown")  # type: ignore

    with pytest.raises(ValueError):
        storch.linalg.eighb(a, b=a, scheme="unknown")  # type: ignore


def test_eighb_standard_single() -> None:
    """eighb accuracy on a single standard eigenvalue problem."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    for _ in range(10):
        a = _symrng((10, 10), dd)

        w_ref = linalg.eigh(tensor_to_numpy(a))[0]
        w_ref = numpy_to_tensor(w_ref, **dd)

        factor = torch.tensor(1e-12, **dd)
        w_calc, v_calc = storch.linalg.eighb(a, factor=factor, aux=False)

        mae_w = torch.max(torch.abs(w_calc.cpu() - w_ref))
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


@pytest.mark.parametrize("direct_inverse", [True, False])
def test_eighb_general_single(direct_inverse: bool) -> None:
    """eighb accuracy on a single general eigenvalue problem."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    for _ in range(10):
        a = _symrng((10, 10), dd)
        b = symmetrizef(torch.eye(10, **dd) * _rng((10,), dd))

        w_ref = linalg.eigh(tensor_to_numpy(a), tensor_to_numpy(b))[0]
        w_ref = numpy_to_tensor(w_ref, **dd)

        schemes: list[Literal["chol", "lowd"]] = ["chol", "lowd"]
        for scheme in schemes:
            w_calc, v_calc = storch.linalg.eighb(
                a, b, scheme=scheme, direct_inv=direct_inverse
            )

            mae_w = torch.max(torch.abs(w_calc.cpu() - w_ref))
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
        b = [symmetrizef(torch.eye(s, **dd) * _rng((s,), dd)) for s in sizes]
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
        return torch.linalg.eigh(m)
    else:
        return storch.linalg.eighb(m, broadening_method=target_method)


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
            target_method=method_l,  # type: ignore[arg-type]
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
            target_method=method_l,  # type: ignore[arg-type]
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
    return storch.linalg.eighb(m, n, scheme=target_scheme, factor=factor)


@pytest.mark.grad
@pytest.mark.parametrize("scheme", ["chol", "lowd"])
def test_eighb_general_grad(scheme: Literal["chol", "lowd"]) -> None:
    """eighb gradient stability on general eigenvalue problems."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    # Generate a single generalised eigenvalue test instance
    a1 = _symrng((8, 8), dd)
    b1 = symmetrizef(torch.eye(8, **dd) * _rng((8,), dd), force=True)

    a1.requires_grad, b1.requires_grad = True, True

    # dgradcheck only takes tensors, but: Loop variable capture of lambda
    assert dgradcheck(
        lambda a1_l, b1_l, scheme_l=scheme: _eigen_proxy_general(
            a1_l,
            b1_l,
            target_scheme=scheme_l,  # type: ignore[arg-type]
        ),
        (a1, b1),
        fast_mode=False,
        atol=1e-1,
        rtol=1e-1,
    ), f"Non-degenerate single test failed on {scheme}"


@pytest.mark.grad
@pytest.mark.flaky(reruns=5, reruns_delay=2)
@pytest.mark.parametrize("scheme", ["chol", "lowd"])
def test_eighb_general_grad_batch(scheme: Literal["chol", "lowd"]) -> None:
    """eighb gradient stability on general eigenvalue problems."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    # Generate a batch of generalised eigenvalue test instances
    sizes = np.random.randint(3, 8, (5,))
    a2 = pack([_symrng((s, s), dd) for s in sizes])
    b2 = pack([symmetrizef(torch.eye(s, **dd) * _rng((s,), dd)) for s in sizes])

    a2.requires_grad, b2.requires_grad = True, True

    # -> loosen tolerances
    # sometimes randomly fails with "chol" on random GA runners
    assert dgradcheck(
        lambda a2_l, b2_l, size_data_l, scheme_l=scheme: _eigen_proxy_general(
            a2_l,
            b2_l,
            size_data=size_data_l,
            target_scheme=scheme_l,  # type: ignore[arg-type]
        ),
        (a2, b2, numpy_to_tensor(sizes, **dd)),
        fast_mode=False,
        atol=1e-1,
        rtol=1e-1,
    ), f"Non-degenerate batch test failed on {scheme}"
