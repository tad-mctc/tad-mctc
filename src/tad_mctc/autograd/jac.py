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
Autograd Utility: Jacobian
==========================

Utilities for calculating Jacobians via autograd.
"""
from __future__ import annotations

from .._typing import Any, Callable, Tensor
from .internals import jacrev  # type: ignore[import-error]

__all__ = ["jac"]


def jac(f: Callable[..., Tensor], argnums: int = 0) -> Any:  # pragma: no cover
    """
    Wrapper for Jacobian calcluation.

    Parameters
    ----------
    f : Callable[[Any], Tensor]
        The function whose result is differentiated.
    argnums : int, optional
        The variable w.r.t. which will be differentiated. Defaults to 0.
    """

    if jacrev is None:
        try:
            from torch.autograd.functional import jacobian  # type: ignore[import-error]
        except ImportError as e:
            raise ImportError(
                f"Failed to import required modules. {e}. {e.name} provides "
                "an API for Jacobian calculations for older PyTorch versions."
            )

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

            # `jacobian` only takes tensors, requiring another wrapper than
            # passes the non-tensor arguments to the function `f`
            def _f(arg: Tensor) -> Tensor:
                return f(*(*before, arg, *after))

            return jacobian(_f, inputs=diffarg)  # type: ignore # pylint: disable=used-before-assignment

        return wrap

    return jacrev(f, argnums=argnums)  # type: ignore
