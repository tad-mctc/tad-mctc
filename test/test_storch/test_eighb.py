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
from tad_mctc._typing import DD, Literal, Tensor
from tad_mctc.autograd import dgradcheck
from tad_mctc.batch import pack
from tad_mctc.convert import numpy_to_tensor, symmetrize, tensor_to_numpy

from ..conftest import DEVICE, FAST_MODE


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


def test_eighb_standard_single():
    """eighb accuracy on a single standard eigenvalue problem."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    for _ in range(10):
        a = symmetrize(
            numpy_to_tensor(np.random.rand(10, 10), **dd),
            force=True,
        )

        w_ref = linalg.eigh(tensor_to_numpy(a))[0]

        w_calc, v_calc = storch.linalg.eighb(a)

        mae_w = torch.max(torch.abs(w_calc.cpu() - w_ref))
        mae_v = torch.max(torch.abs((v_calc @ v_calc.T).fill_diagonal_(0)))

        dev_str = torch.device("cpu") if DEVICE is None else DEVICE
        same_device = w_calc.device == dev_str == v_calc.device

        assert mae_w < 1e-12, "Eigenvalue tolerance test"
        assert mae_v < 1e-12, "Eigenvector orthogonality test"
        assert same_device, "Device persistence check"


def test_eighb_standard_batch():
    """eighb accuracy on a batch of standard eigenvalue problems."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    for _ in range(10):
        sizes = np.random.randint(2, 10, (11,))
        a = [
            symmetrize(numpy_to_tensor(np.random.rand(s, s), **dd), force=True)
            for s in sizes
        ]
        a_batch = pack(a)

        w_ref = pack(
            [torch.tensor(linalg.eigh(tensor_to_numpy(i))[0], **dd) for i in a]
        )

        w_calc = storch.linalg.eighb(a_batch)[0]

        mae_w = torch.max(torch.abs(w_calc - w_ref))
        assert mae_w < 1e-12, "Eigenvalue tolerance test"

        dev_str = torch.device("cpu") if DEVICE is None else DEVICE
        same_device = w_calc.device == dev_str
        assert same_device, "Device persistence check"


def test_eighb_general_single():
    """eighb accuracy on a single general eigenvalue problem."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    for _ in range(10):
        a_np = numpy_to_tensor(np.random.rand(10, 10), **dd)
        a = symmetrize(a_np, force=True)

        b_np = numpy_to_tensor(np.random.rand(10), **dd)
        b = symmetrize(torch.eye(10, **dd) * b_np, force=True)

        w_ref = linalg.eigh(tensor_to_numpy(a), tensor_to_numpy(b))[0]

        schemes: list[Literal["chol", "lowd"]] = ["chol", "lowd"]
        for scheme in schemes:
            w_calc, v_calc = storch.linalg.eighb(a, b, scheme=scheme)

            mae_w = torch.max(torch.abs(w_calc.cpu() - w_ref))
            mae_v = torch.max(torch.abs((v_calc @ v_calc.T).fill_diagonal_(0)))

            dev_str = torch.device("cpu") if DEVICE is None else DEVICE
            same_device = w_calc.device == dev_str == v_calc.device

            assert mae_w < 1e-12, f"Eigenvalue tolerance test {scheme}"
            assert mae_v < 1e-12, f"Eigenvector orthogonality test {scheme}"
            assert same_device, "Device persistence check"


def test_eighb_general_batch():
    """eighb accuracy on a batch of general eigenvalue problems."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    for _ in range(10):
        sizes = np.random.randint(2, 10, (11,))
        a = [
            symmetrize(numpy_to_tensor(np.random.rand(s, s), **dd), force=True)
            for s in sizes
        ]
        b = [
            symmetrize(
                torch.eye(s, **dd) * numpy_to_tensor(np.random.rand(s), **dd),
                force=True,
            )
            for s in sizes
        ]
        a_batch, b_batch = pack(a), pack(b)

        w_ref = pack(
            [
                torch.tensor(
                    linalg.eigh(tensor_to_numpy(i), tensor_to_numpy(j))[0], **dd
                )
                for i, j in zip(a, b)
            ]
        )

        aux_settings = [True, False]
        schemes: list[Literal["chol", "lowd"]] = ["chol", "lowd"]
        for scheme in schemes:
            for aux in aux_settings:
                (w_calc, _) = storch.linalg.eighb(
                    a_batch, b_batch, scheme=scheme, aux=aux
                )

                mae_w = torch.max(torch.abs(w_calc - w_ref))

                dev_str = torch.device("cpu") if DEVICE is None else DEVICE
                same_device = w_calc.device == dev_str

                assert mae_w < 1e-10, f"Eigenvalue tolerance test {scheme}"
                assert same_device, "Device persistence check"


@pytest.mark.grad
def test_eighb_broadening_grad():
    """eighb gradient stability on standard, broadened, eigenvalue problems.

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

    def eigen_proxy(m, target_method, size_data=None):
        m = symmetrize(m, force=True)
        if size_data is not None:
            m = clean_zero_padding(m, size_data)
        if target_method is None:
            return torch.linalg.eigh(m)
        else:
            return storch.linalg.eighb(m, broadening_method=target_method)

    # Generate a single standard eigenvalue test instance
    a1_np = numpy_to_tensor(np.random.rand(8, 8), **dd)
    a1 = symmetrize(a1_np, force=True)

    broadening_methods = [None, "none", "cond", "lorn"]
    for method in broadening_methods:
        # dgradcheck detaches
        a1.requires_grad = True

        grad_is_safe = dgradcheck(
            eigen_proxy,
            (a1, method),
            raise_exception=False,
            fast_mode=FAST_MODE,
        )
        assert grad_is_safe, f"Non-degenerate single test failed on {method}"

    # Generate a batch of standard eigenvalue test instances
    sizes = np.random.randint(3, 8, (5,))
    a2 = pack(
        [
            symmetrize(numpy_to_tensor(np.random.rand(s, s), **dd), force=True)
            for s in sizes
        ]
    )

    for method in broadening_methods[2:]:
        # dgradcheck detaches!
        a2.requires_grad = True

        grad_is_safe = dgradcheck(
            eigen_proxy,
            (a2, method, numpy_to_tensor(sizes, **dd)),
            raise_exception=False,
            fast_mode=FAST_MODE,
        )
        assert grad_is_safe, f"Non-degenerate batch test failed on {method}"


@pytest.mark.grad
def test_eighb_general_grad():
    """eighb gradient stability on general eigenvalue problems."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    def eigen_proxy(m, n, target_scheme, size_data=None):
        m, n = symmetrize(m, force=True), symmetrize(n, force=True)
        if size_data is not None:
            m = clean_zero_padding(m, size_data)
            n = clean_zero_padding(n, size_data)

        return storch.linalg.eighb(m, n, scheme=target_scheme)

    # Generate a single generalised eigenvalue test instance
    a1_np = numpy_to_tensor(np.random.rand(8, 8), **dd)
    a1 = symmetrize(a1_np, force=True)

    b1_np = numpy_to_tensor(np.random.rand(8), **dd)
    b1 = symmetrize(torch.eye(8, **dd) * b1_np, force=True)

    schemes: list[Literal["chol", "lowd"]] = ["chol", "lowd"]
    for scheme in schemes:
        # dgradcheck detaches!
        a1.requires_grad, b1.requires_grad = True, True

        grad_is_safe = dgradcheck(
            eigen_proxy,
            (a1, b1, scheme),  # type: ignore
            fast_mode=FAST_MODE,
        )
        assert grad_is_safe, f"Non-degenerate single test failed on {scheme}"

    # Generate a batch of generalised eigenvalue test instances
    sizes = np.random.randint(3, 8, (5,))
    a2 = pack(
        [
            symmetrize(numpy_to_tensor(np.random.rand(s, s), **dd), force=True)
            for s in sizes
        ]
    )
    b2 = pack(
        [
            symmetrize(
                torch.eye(int(s), **dd) * numpy_to_tensor(np.random.rand(int(s)), **dd),
                force=True,
            )
            for s in sizes
        ]
    )

    for scheme in schemes:
        # dgradcheck detaches!
        a2.requires_grad, b2.requires_grad = True, True

        grad_is_safe = dgradcheck(
            eigen_proxy,
            (a2, b2, scheme, numpy_to_tensor(sizes, **dd)),  # type: ignore
            fast_mode=FAST_MODE,
        )
        assert grad_is_safe, f"Non-degenerate batch test failed on {scheme}"
