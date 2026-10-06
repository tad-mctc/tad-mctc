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
Module containing the version string.
"""

from __future__ import annotations

import torch
from torch.torch_version import TorchVersion

__all__ = ["__version__", "__tversion__"]


__version__ = "0.9.2"
"""Version of tad-mctc in semantic versioning."""

__tversion__ = TorchVersion(torch.__version__)
"""
Version of PyTorch. Compares with tuples (``__tversion__ >= (2, 8, 0)``) and
strings, and orders pre-releases before their release (``2.8.0a0`` is older
than ``2.8.0``).
"""
