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
Command line interface
=======================

`tad_mctc` as a command line tool: read a structure file and print its
coordination number, optionally with the wall time of every step,
including the steps of the neighbour-list build.

.. code-block:: sh

   tad_mctc structure.xyz
   tad_mctc --cn d4 --timing --cuda --omp 4 coord
   tad_mctc --cn d4 --nlist-only --timing structure.xyz

The package splits into the argument parser (:mod:`._args`), the step
timer (:mod:`._timing`), the printed sections (:mod:`._output`) and the
run itself (:mod:`._main`).
"""

from ._main import main

__all__ = ["main"]
