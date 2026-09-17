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
Utility functions for testing.
"""

from __future__ import annotations

import numpy as np
import torch

from tad_mctc.convert import numpy_to_tensor, symmetrizef
from tad_mctc.typing import DD, Tensor

__all__ = ["_rng", "_symrng", "DYNAMO_SUPPORTED", "DYNAMO_UNSUPPORTED_REASON"]


def _rng(size: tuple[int, ...] | int, dd: DD) -> Tensor:
    s = (size,) if isinstance(size, int) else size
    n = np.random.rand(*s)
    return numpy_to_tensor(n, **dd)


def _symrng(size: tuple[int, ...] | int, dd: DD) -> Tensor:
    return symmetrizef(_rng(size, dd))


def _dynamo_is_supported() -> bool:
    """
    Whether ``torch.compile``/Dynamo tracing is usable on this Python/
    PyTorch combination -- either not, because this PyTorch predates
    ``torch.compile`` entirely, or because
    ``torch._dynamo.is_dynamo_supported()`` reports that Dynamo does not
    support this Python version yet (support for a new Python release
    consistently lags the PyTorch release that first runs on it).
    """
    if not hasattr(torch, "compile"):
        return False

    try:
        import torch._dynamo as dynamo  # pylint: disable=protected-access
    except ImportError:
        return False

    return bool(
        dynamo.is_dynamo_supported()
    )  # pyright: ignore[reportPrivateImportUsage]


DYNAMO_SUPPORTED = _dynamo_is_supported()
"""For ``@pytest.mark.skipif(not DYNAMO_SUPPORTED, reason=DYNAMO_UNSUPPORTED_REASON)``
on any test that calls ``torch.compile``."""

DYNAMO_UNSUPPORTED_REASON = (
    "torch.compile/Dynamo is not supported on this Python/PyTorch combination"
)
