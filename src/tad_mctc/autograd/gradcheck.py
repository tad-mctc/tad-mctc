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
Autograd Utility: Gradcheck
===========================

Collection of utility functions for testing.
"""
from __future__ import annotations

from torch.autograd.gradcheck import gradcheck, gradgradcheck

from ..typing import Any, Callable, Protocol, Tensor, TensorOrTensors

__all__ = ["dgradcheck", "dgradgradcheck"]


FAST_MODE = True
"""Default for fast_mode argument (True)."""


class _GradcheckFunction(Protocol):
    """
    Type annotation for gradcheck function.
    """

    def __call__(  # type: ignore
        self,
        func: Callable[..., TensorOrTensors],
        inputs: TensorOrTensors,
        *,
        eps: float = 1e-6,
        atol: float = 1e-5,
        rtol: float = 1e-3,
        raise_exception: bool = True,
        check_sparse_nnz: bool = False,
        nondet_tol: float = 0.0,
        check_undefined_grad: bool = True,
        check_grad_dtypes: bool = False,
        check_batched_grad: bool = False,
        check_batched_forward_grad: bool = False,
        check_forward_ad: bool = False,
        check_backward_ad: bool = True,
        fast_mode: bool = False,
    ) -> bool:
        ...


class _GradgradcheckFunction(Protocol):
    """
    Type annotation for gradgradcheck function.
    """

    def __call__(  # type: ignore
        self,
        func: Callable[..., TensorOrTensors],
        inputs: TensorOrTensors,
        grad_outputs: TensorOrTensors | None = None,
        *,
        eps: float = 1e-6,
        atol: float = 1e-5,
        rtol: float = 1e-3,
        gen_non_contig_grad_outputs: bool = False,
        raise_exception: bool = True,
        nondet_tol: float = 0.0,
        check_undefined_grad: bool = True,
        check_grad_dtypes: bool = False,
        check_batched_grad: bool = False,
        check_fwd_over_rev: bool = False,
        check_rev_over_rev: bool = True,
        fast_mode: bool = False,
    ) -> bool:
        ...


def _wrap_gradcheck(
    gradcheck_func: _GradcheckFunction | _GradgradcheckFunction,
    func: Callable[..., TensorOrTensors],
    diffvars: TensorOrTensors,
    **kwargs: Any,
) -> bool:
    fast_mode = kwargs.pop("fast_mode", FAST_MODE)
    try:
        assert gradcheck_func(func, diffvars, fast_mode=fast_mode, **kwargs)
    finally:
        if isinstance(diffvars, Tensor):
            diffvars.detach_()
        else:
            for diffvar in diffvars:
                if isinstance(diffvar, Tensor):
                    diffvar.detach_()

    return True


def dgradcheck(
    func: Callable[..., TensorOrTensors], diffvars: TensorOrTensors, **kwargs: Any
) -> bool:
    """
    Wrapper for `torch.autograd.gradcheck` that detaches the differentiated
    variables after the check.

    Parameters
    ----------
    func : Callable[..., TensorOrTensors]
        Forward function.
    diffvars : TensorOrTensors
        Variables w.r.t. which we differentiate.

    Returns
    -------
    bool
        Status of check.
    """
    return _wrap_gradcheck(gradcheck, func, diffvars, **kwargs)


def dgradgradcheck(
    func: Callable[..., TensorOrTensors], diffvars: TensorOrTensors, **kwargs: Any
) -> bool:
    """
    Wrapper for `torch.autograd.gradgradcheck` that detaches the differentiated
    variables after the check.

    Parameters
    ----------
    func : Callable[..., TensorOrTensors]
        Forward function.
    diffvars : TensorOrTensors
        Variables w.r.t. which we differentiate.

    Returns
    -------
    bool
        Status of check.
    """
    return _wrap_gradcheck(gradgradcheck, func, diffvars, **kwargs)
