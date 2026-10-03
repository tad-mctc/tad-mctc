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
Sample `Node` subclasses shared by the tests of `tad_mctc.tree`.
"""

from __future__ import annotations

from typing import ClassVar

import torch

from tad_mctc.tree import Node, child, context

__all__ = ["Base", "Sub", "Plain", "Holder", "Override", "_double"]


class Base(Node):
    numbers: torch.Tensor = child()
    positions: torch.Tensor = child()
    charge: torch.Tensor | None = child(default=None)
    uhf: torch.Tensor | None = child(default=None, keep_dtype=True)
    label: str = context(default="x")

    def _validate(self) -> None:
        if self.positions.shape[-1] != 3:
            raise RuntimeError("positions must have shape (..., 3)")


def _double(x: float) -> float:
    return 2 * x


class Sub(Base):
    rcov: object = child(default=None)
    cutoff: float = context(default=25.0)
    implemented: ClassVar[tuple[str, ...]] = ("a",)


class Plain(Sub):
    pass


class Holder(Node):
    system: Node = child()
    extra: dict[str, torch.Tensor] = child(default_factory=dict)


class Override(Sub):
    def _convert_child(self, name, value, device, dtype):  # type: ignore[no-untyped-def]
        if name == "charge":
            return "overridden"
        return super()._convert_child(name, value, device, dtype)
