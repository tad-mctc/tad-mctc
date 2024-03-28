# This file is part of tad-multicharge.
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
Autograd Utility: Jacobian
==========================

Utilities for calculating Jacobians via autograd.
"""
from __future__ import annotations

from ..typing import Any, Callable, Tensor

__all__ = ["jacrev_compat"]


def jacrev_compat(
    f: Callable[..., Tensor], argnums: int = 0, **kwargs: Any
) -> Any:  # pragma: no cover
    """
    Wrapper for Jacobian calcluation.

    Parameters
    ----------
    f : Callable[[Any], Tensor]
        The function whose result is differentiated.
    argnums : int, optional
        The variable w.r.t. which will be differentiated. Defaults to 0.
    """

    try:
        # pylint: disable=import-outside-toplevel
        from torch.autograd.functional import jacobian  # type: ignore[import-error]
    except ImportError as e:
        raise ImportError(
            f"Failed to import required modules. {e}. {e.name} provides "
            "an API for Jacobian calculations for older PyTorch versions."
        ) from e

    def wrap(*inps: Any) -> Any:
        """
        Wrapper to imitate the calling signature of functorch's `jacrev`
        with `torch.autograd.functional.jacobian`.

        Parameters
        ----------
        inps : tuple[Any, ...]
            The input parameters of the function `f`.

        Returns
        -------
        Any
            Jacobian function.

        Raises
        ------
        RuntimeError
            The parameter selected for differentiation (via `argnums`) is
            not a tensor.
        """
        diffarg = inps[argnums]
        if not isinstance(diffarg, Tensor):
            raise RuntimeError(
                f"The {argnums}'th input parameter must be a tensor but is "
                f"of type '{type(diffarg)}'."
            )

        before = inps[:argnums]
        after = inps[(argnums + 1) :]

        # `jacobian` only takes tensors, requiring another wrapper that
        # passes the non-tensor arguments to the function `f`
        def _f(arg: Tensor) -> Tensor:
            return f(*(*before, arg, *after))

        return jacobian(_f, inputs=diffarg, **kwargs)  # type: ignore # pylint: disable=used-before-assignment

    return wrap
