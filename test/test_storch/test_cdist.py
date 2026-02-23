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
Test cdist safeop version.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tad_mctc import storch
from tad_mctc.convert import numpy_to_tensor
from tad_mctc.typing import DD

from ..conftest import DEVICE


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_all(dtype: torch.dtype) -> None:
    """
    The single precision test sometimes fails on my GPU with the following
    thresholds:

    ```
    tol = 1e-6 if dtype == torch.float else 1e-14
    ```

    Only one matrix element seems to be affected. It also appears that the
    failure only happens if `torch.rand` was run before. To be precise,

    ```
    pytest -vv test/test_ncoord/test_grad.py test/test_storch/ --cuda --slow
    ```

    fails, while

    ```
    pytest -vv test/test_storch/ --cuda --slow
    ```

    works. It also works if I remove the random tensors in the gradient test
    (test/test_ncoord/test_grad.py).

    It can be fixed with

    ```
    torch.use_deterministic_algorithms(True)
    ```

    and following the PyTorch instructions to set a specific
    environment variable.

    ```
    CUBLAS_WORKSPACE_CONFIG=:4096:8 pytest -vv test/test_ncoord/test_grad.py test/test_utils/ --cuda --slow
    ```

    For simplicity, one can just reduce the tolerances for single precision.
    ```
    # only one element actually fails
    if "cuda" in str(DEVICE) and dtype == torch.float:
        tol = 1e-3
    ```

    BETTER SOLUTION:
    Use numpy for generating random tensors and convert the array to a tensor.
    """
    dd: DD = {"device": DEVICE, "dtype": dtype}

    if "cuda" in str(DEVICE) and dtype == torch.float:
        tol = 1e-6
    elif "cuda" in str(DEVICE) and dtype == torch.double:
        tol = 1e-7
    elif DEVICE is None and dtype == torch.float:
        tol = 1e-6
    elif DEVICE is None and dtype == torch.double:
        tol = 1e-14
    else:
        raise RuntimeError("Unknown device or dtype.")

    x = numpy_to_tensor(np.random.randn(2, 3, 4), **dd)

    d1 = storch.cdist(x)
    d2 = storch.distance.cdist_direct_expansion(x, x, p=2)
    d3 = storch.distance.euclidean_dist_quadratic_expansion(x, x)

    assert pytest.approx(d1.cpu(), abs=tol) == d2.cpu()
    assert pytest.approx(d2.cpu(), abs=tol) == d3.cpu()
    assert pytest.approx(d3.cpu(), abs=tol) == d1.cpu()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("p", [2, 3, 4, 5])
def test_ps(dtype: torch.dtype, p: int) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}
    tol = 1e-6 if dtype == torch.float else 1e-14

    x = numpy_to_tensor(np.random.randn(2, 4, 5), **dd)
    y = numpy_to_tensor(np.random.randn(2, 4, 5), **dd)

    d1 = storch.cdist(x, y, p=p)
    d2 = torch.cdist(x, y, p=p)

    assert pytest.approx(d1.cpu(), abs=tol) == d2.cpu()
