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
Safe Operations
===============

This module contains autograd safe versions of common mathematical operations.
These often called "safeops" remove non-differentiable points or restrict
functions to differentiable domains. Most functions aims to retain the syntax
of the underlying PyTorch versions, but some functions are modified or
simplified for more convenience.
"""

from .distance import *
from .elemental import *
from .linalg import *
from .utils import *
