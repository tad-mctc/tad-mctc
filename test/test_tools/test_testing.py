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
Test that the `pytest` helpers in ``tad_mctc.tools.testing`` keep `pytest`
an optional dependency.
"""

from __future__ import annotations

import subprocess
import sys


def test_package_does_not_import_pytest() -> None:
    # `pytest` is already imported here, so check in a fresh interpreter
    code = (
        "import sys\n"
        "import tad_mctc, tad_mctc.tools, tad_mctc.autograd\n"
        "assert 'pytest' not in sys.modules\n"
        "assert 'tad_mctc.tools.testing' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
