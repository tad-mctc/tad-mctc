"""
Typing and numerical gradient for coordination number functions.
"""

from __future__ import annotations

import torch

from tad_mctc.ncoord.typing import CNFunction
from tad_mctc.typing import Tensor, CountingFunction

__all__ = ["numgrad"]


def numgrad(
    function: CNFunction,
    cf: CountingFunction,
    numbers: Tensor,
    positions: Tensor,
) -> Tensor:
    nat = numbers.shape[-1]
    pos = positions.clone().type(torch.double)

    # setup numerical gradient
    gradient = torch.zeros(
        (*numbers.shape[:-1], nat, nat, 3), dtype=pos.dtype, device=pos.device
    )
    step = 1.0e-6

    for i in range(nat):
        for j in range(3):
            pos[..., i, j] += step
            cnr = function(numbers, pos, counting_function=cf)

            pos[..., i, j] -= 2 * step
            cnl = function(numbers, pos, counting_function=cf)

            pos[..., i, j] += step
            gradient[..., :, i, j] = 0.5 * (cnr - cnl) / step

    return gradient
