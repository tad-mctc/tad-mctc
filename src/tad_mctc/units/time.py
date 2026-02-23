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
Units: Time
===========

This module contains conversions for units of time.
"""

from __future__ import annotations

from .codata import get_constant

__all__ = ["AU2SECOND", "SECOND2AU"]


AU2SECOND = get_constant("atomic unit of time")
"""Conversion from atomic units to seconds. The atomic unit of time (s)."""

SECOND2AU = 1.0 / AU2SECOND
"""Conversion from seconds to atomic units of time."""
