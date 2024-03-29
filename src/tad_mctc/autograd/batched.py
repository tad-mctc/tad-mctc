"""
Autograd Utility: Batched
=========================

Batched versions of autograd functions.
"""

from __future__ import annotations

from ..typing import Callable, Tensor
from .internals import jacrev, vmap


def bjacrev(
    func: Callable[..., Tensor], argnums: int = 0, **kwargs
) -> Callable[..., Tensor]:
    """
    Batched Jacobian of a function.

    Parameters
    ----------
    func : Callable[..., Tensor]
        The function whose result is differentiated.
    argnums : int, optional
        The variable w.r.t. which will be differentiated. Defaults to 0.

    Returns
    -------
    Callable[..., Tensor]
        Batched Jacobian function.
    """
    f: Callable = jacrev(func, argnums=argnums, **kwargs)  # type: ignore
    return vmap(f, in_dims=0, out_dims=0)
