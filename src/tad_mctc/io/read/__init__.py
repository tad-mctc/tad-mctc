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
I/O: Read
=========

This module contains functions for reading files.

The simplest interface is provided by the `read` function, which tries to infer
the file type from the file name or extension.

Example
-------
>>> from tad_mctc.io import read
>>> path = "mol.xyz"
>>> numbers, positions = read.read(path)
"""
from .dotfiles import *
from .frompath import *
from .orca import *
from .qcschema import *
from .reader import *
from .tblite import *
from .turbomole import *
from .xyz import *
