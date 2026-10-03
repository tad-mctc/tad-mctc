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
Tests of the coordination-number models.

- `test_reference.py`, `test_grad/`: every variant against the mctc-lib
  Fortran references, through every evaluation path (see `_paths.py`).
- `test_periodic_cells.py`: cell geometry (unwrapped atoms, slabs, wires,
  left-handed cells) through every evaluation path.
- `test_transforms.py`, `test_compile.py`: `vmap`, `jacrev` and
  `torch.compile` of every variant, through every evaluation path that can
  be traced.
- `test_model.py`, `test_general.py`: `CNModel` and the counting functions,
  independent of the evaluation path.
- One module per quadrant, i.e. evaluation path crossed with geometry:
  `test_dense_molecular.py`, `test_dense_periodic.py`,
  `test_sparse_molecular.py`, `test_sparse_periodic.py`, and
  `test_sparse.py` for the sparse checks shared by both geometries.
"""
