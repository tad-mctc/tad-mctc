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
Command line interface: timing
==============================

Wall times of the steps of a run, reported live and as a final table.
"""

from __future__ import annotations

import contextlib
import time
from typing import Generator

import torch

__all__ = ["Timings"]

# Column width of the step labels. The longest label the tool prints is
# `cn_eeqbc (sparse, recompute)`; a longer one only shifts its own line.
_LABEL_WIDTH = 28


def _format_duration(seconds: float) -> str:
    """Format ``seconds`` as ``"xx min xx sec xxx ms"``."""
    total_ms = round(seconds * 1000)
    ms, total_s = total_ms % 1000, total_ms // 1000
    s, m = total_s % 60, total_s // 60
    return f"{m:d} min {s:02d} sec {ms:03d} ms"


# Width of `_format_duration`'s output for up to (and including) 99
# minutes, used to right-align it in both the live lines and the table.
_DURATION_WIDTH = len(_format_duration(99 * 60 + 59.999))


def _synchronize_cuda() -> None:
    """Wait for queued CUDA kernels, which launch asynchronously, to finish."""
    if torch.cuda.is_initialized():
        torch.cuda.synchronize()


class Timings:
    """
    Reports each step live -- a "label ..." line as it starts, overwritten
    in place with the elapsed time once it finishes, so a run stuck on a
    slow step is visible immediately rather than only after the whole
    command completes -- and also collects the wall times so they can be
    reported together as one aligned table at the end.

    When disabled, :meth:`stage` does nothing and :meth:`report` prints
    nothing.
    """

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self._entries: list[tuple[str, float]] = []

    @contextlib.contextmanager
    def stage(self, label: str) -> Generator[None, None, None]:
        """Time the ``with`` block under ``label``, reporting it live."""
        if not self.enabled:
            yield
            return

        running = f"  {label:<{_LABEL_WIDTH}} ..."
        print(running, end="", flush=True)
        _synchronize_cuda()
        start = time.perf_counter()

        yield

        _synchronize_cuda()
        elapsed = time.perf_counter() - start
        self._entries.append((label, elapsed))

        duration = _format_duration(elapsed)
        done = (
            f"  {label:<{_LABEL_WIDTH}} ... done "
            f"({duration:>{_DURATION_WIDTH}})"
        )
        print(f"\r{done}")

    def report(self) -> None:
        """Print the recorded steps as an aligned table, with a total row."""
        if not self.enabled or not self._entries:
            return

        total = sum(elapsed for _, elapsed in self._entries)
        width = max(len(label) for label, _ in self._entries)

        print()
        print("Timing")
        print("------")
        for label, elapsed in self._entries:
            share = 100 * elapsed / total if total > 0 else 0.0
            print(
                f"  {label:<{width}}  "
                f"{_format_duration(elapsed):>{_DURATION_WIDTH}}  "
                f"{share:5.1f} %"
            )
        print(
            f"  {'total':<{width}}  "
            f"{_format_duration(total):>{_DURATION_WIDTH}}  100.0 %"
        )
