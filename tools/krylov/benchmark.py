# SPDX-Identifier: CC0-1.0
"""Opt-in synthetic and EEQ solver measurements; see README.md."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
import traceback
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from typing import Any
from unittest.mock import patch

import torch
from torch.func import jacfwd, jacrev, jvp, vmap

from tad_mctc.storch.linsolve import implicit as implementation
from tad_mctc.storch.linsolve import implicit_solve

Tensor = torch.Tensor
Case = tuple[
    Callable[[Tensor], Tensor],
    Callable[[Tensor], Tensor],
    Tensor,
    Tensor,
    Tensor,
]


def snapshot(value: Tensor | None) -> Any:
    """Read diagnostics only; never unwrap a solver input or alter AD."""
    if value is None:
        return None
    ft = torch._C._functorch
    while ft.is_functorch_wrapped_tensor(value):
        value = ft.get_unwrapped(value)
    with torch._C._DisableFuncTorch():
        return value.detach().cpu().tolist()


@contextmanager
def instrumentation(records: dict[str, Any]) -> Iterator[None]:
    """Observe Krylov/refinement calls on both the PR base and head."""
    original_refine = implementation._refine
    fast_operators: list[Callable[..., Tensor]] = []

    def record(method: Callable[..., Any]) -> Callable[..., Any]:
        def solve(*args: Any, **kwargs: Any) -> Any:
            phase = "exact"
            if fast_operators:
                phase = "fast" if args[0] is fast_operators[-1] else "fallback"
                records["solver_calls"][phase] += 1
            result = method(*args, **kwargs)
            records["solves"].append(
                dict(
                    phase=phase,
                    status=snapshot(result.status),
                    iterations=snapshot(result.iterations),
                    residual=snapshot(result.residual),
                    target=snapshot(result.target),
                )
            )
            return result

        return solve

    def refine(system: Any, *args: Any, **kwargs: Any) -> Any:
        fast_operators.append(system.fast_matvec)
        try:
            result = original_refine(system, *args, **kwargs)
        finally:
            fast_operators.pop()
        records["refinements"].append(
            dict(
                status=snapshot(result.status),
                refinement_status=snapshot(result.refinement_status),
                fallback_iterations=snapshot(result.fallback_iterations),
            )
        )
        return result

    from tad_mctc.storch.linsolve import constrained

    methods = {
        name: record(method) for name, method in implementation._METHODS.items()
    }
    with patch.dict(implementation._METHODS, methods), patch.object(
        constrained, "projected_cg", record(constrained.projected_cg)
    ), patch.object(implementation, "_refine", refine):
        yield


def synthetic(args: argparse.Namespace, records: dict[str, Any]) -> Case:
    dd = dict(device=args.device, dtype=torch.double)
    generator = torch.Generator().manual_seed(0)
    parameter = torch.randn(
        args.size, args.size, generator=generator, dtype=torch.double
    ).to(args.device)
    rhs = torch.linspace(1.0, 2.0, args.size, **dd)
    offsets = torch.tensor([0.0, 0.1], **dd)
    identity = torch.eye(args.size, **dd)
    parameter = parameter / math.sqrt(args.size)

    def matrix(values: Tensor, shift: Tensor) -> Tensor:
        return values @ values.mT + (2.0 + shift) * identity

    def matvec(vector: Tensor, values: Tensor, shift: Tensor) -> Tensor:
        records["products"]["exact"] += 1
        return matrix(values, shift) @ vector

    def fast(vector: Tensor, values: Tensor, shift: Tensor) -> Tensor:
        records["products"]["fast"] += 1
        operator = matrix(values, shift)
        if args.case == "fallback":
            return -(operator @ vector)
        return (operator.float() @ vector.float()).double()

    def linearize(values: Tensor, shift: Tensor) -> Callable[[Tensor], Tensor]:
        records["products"]["cache_setup"] += 1
        operator = matrix(values, shift)

        def product(vector: Tensor) -> Tensor:
            records["products"]["cached"] += 1
            return operator @ vector

        return product

    def evaluate(shift: Tensor, dense: bool) -> Tensor:
        values = (
            parameter + 0.01 * shift * identity
            if args.mapped_parameter
            else parameter
        )
        current_rhs = rhs + shift
        if dense:
            factor = torch.linalg.cholesky(matrix(values, shift))
            solution = torch.cholesky_solve(
                current_rhs[:, None], factor
            ).squeeze(-1)
        else:
            result = implicit_solve(
                matvec,
                current_rhs,
                values,
                shift,
                linearize=linearize if args.case == "linearization" else None,
                fast_matvec=(
                    fast if args.case in ("refined", "fallback") else None
                ),
                fallback=True,
                maxiter=args.maxiter,
                rtol=1e-10,
                atol=1e-12,
            )
            solution = result.solution
        return solution.sum()

    # A mapped RHS keeps the baseline refinement's initial residual mapped.
    # Linearization benchmarks map only RHS, or also the operator parameter.
    def objective(displacement: Tensor, dense: bool) -> Tensor:
        if args.case != "linearization":
            return evaluate(displacement, dense)
        values = (
            parameter + 0.01 * displacement * identity
            if args.mapped_parameter
            else parameter
        )
        current_rhs = rhs + displacement
        if dense:
            factor = torch.linalg.cholesky(
                matrix(values, displacement.new_zeros(()))
            )
            return torch.cholesky_solve(current_rhs[:, None], factor).sum()
        result = implicit_solve(
            matvec,
            current_rhs,
            values,
            displacement.new_zeros(()),
            linearize=linearize,
            maxiter=args.maxiter,
        )
        return result.solution.sum()

    def actual(displacement: Tensor) -> Tensor:
        return objective(displacement, False)

    def reference(displacement: Tensor) -> Tensor:
        return objective(displacement, True)

    return actual, reference, rhs.new_zeros(()), rhs.new_ones(()), offsets


def eeq_case(
    args: argparse.Namespace, records: dict[str, Any], patches: ExitStack
) -> Case:
    # Optional integration dependency; the synthetic cases need only mctc.
    from tad_multicharge import eeq
    from tad_multicharge.model import _matrix_free as mf

    from tad_mctc.io.structure import Structure
    from tad_mctc.neighbor import build_neighborlist

    molecules = args.size // 3
    side = math.ceil(molecules ** (1 / 3))
    axes = torch.arange(side, device=args.device, dtype=torch.double)
    centers = (
        5.86
        * torch.stack(
            torch.meshgrid(axes, axes, axes, indexing="ij"), -1
        ).reshape(-1, 3)[:molecules]
    )
    water = torch.tensor(
        [[0.0, 0.0, 0.0], [1.43, 1.11, 0.0], [-1.43, 1.11, 0.0]],
        device=args.device,
        dtype=torch.double,
    )
    positions = (centers[:, None] + water).reshape(-1, 3)
    numbers = torch.tensor([8, 1, 1], device=args.device).repeat(molecules)
    structure = Structure(
        numbers=numbers, positions=positions, charge=positions.new_tensor(-1.0)
    )
    model = eeq.EEQModel.param2019(device=args.device, dtype=torch.double)
    neighbors = build_neighborlist(structure, model.cn.cutoff, skin=1.0)
    direction = torch.sin(
        torch.arange(positions.numel(), device=args.device, dtype=torch.double)
    ).reshape_as(positions)
    direction = direction / torch.linalg.vector_norm(direction)
    original = mf._matvec
    preconditioner = mf._preconditioner

    def matvec(*values: Any, **kwargs: Any) -> Tensor:
        phase = "fast" if kwargs.get("dtype") == torch.float32 else "exact"
        records["products"][phase] += 1
        return original(*values, **kwargs)

    def select(*values: Any, **kwargs: Any) -> Any:
        selected = preconditioner(*values, **kwargs)
        records["preconditioners"].append(
            "jacobi" if selected is mf._jacobi else "balancing"
        )
        return selected

    patches.enter_context(patch.object(mf, "_matvec", matvec))
    patches.enter_context(patch.object(mf, "_preconditioner", select))

    def objective(current_positions: Tensor, solve_mode: str) -> Tensor:
        current = structure.replace(positions=current_positions)
        charges, energies = model(
            current, neighbors, solve_mode=solve_mode, return_energy=True
        )
        return energies.sum() + 0.01 * charges.square().sum()

    def actual(current_positions: Tensor) -> Tensor:
        return objective(current_positions, "matrix-free")

    def reference(current_positions: Tensor) -> Tensor:
        return objective(current_positions, "schur")

    return (
        actual,
        reference,
        positions,
        direction,
        torch.stack((positions, positions + 0.01 * direction)),
    )


def execute(
    function: Callable[[Tensor], Tensor],
    value: Tensor,
    direction: Tensor,
    batch: Tensor,
    mode: str,
) -> Tensor:
    if mode == "eager":
        return function(value)
    if mode == "native":
        parameter = value.clone().requires_grad_()
        function(parameter).backward()
        assert parameter.grad is not None
        return parameter.grad
    if mode == "jvp":
        return jvp(function, (value,), (direction,))[1]
    if mode == "jacrev":
        gradient = jacrev(function)
        return gradient(value)
    if mode == "jacfwd":

        def displacement(offset: Tensor) -> Tensor:
            return function(value + offset * direction)

        derivative = jacfwd(displacement)
        return derivative(value.new_zeros(()))
    if mode == "vmap":
        evaluate = vmap(function)
        return evaluate(batch)
    gradient = jacrev(function)
    return jvp(gradient, (value,), (direction,))[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=("synthetic", "refined", "fallback", "linearization", "eeq"),
        default="synthetic",
    )
    parser.add_argument(
        "--mode",
        choices=("eager", "native", "jvp", "jacrev", "jacfwd", "vmap", "mixed"),
        default="jacrev",
    )
    parser.add_argument("--size", type=int, default=12)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--maxiter", type=int, default=80)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--mapped-parameter", action="store_true")
    args = parser.parse_args()
    if min(args.size, args.maxiter, args.repeats) < 1 or (
        args.case == "eeq" and args.size % 3 != 0
    ):
        parser.error(
            "positive sizes/repeats required; EEQ size must be a multiple of three"
        )
    torch.set_num_threads(1)
    records = dict(
        products=Counter(),
        solver_calls=Counter(),
        solves=[],
        refinements=[],
        preconditioners=[],
    )
    stage = "setup"

    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    try:
        device = torch.device(args.device)
        with ExitStack() as patches:
            data = (
                eeq_case(args, records, patches)
                if args.case == "eeq"
                else synthetic(args, records)
            )
            actual, reference, value, direction, batch = data
            stage = "dense reference"
            expected = execute(
                reference, value, direction, batch, args.mode
            ).detach()
            synchronize()
            stage = "cold evaluation"
            start = time.perf_counter()
            with instrumentation(records):
                execute(actual, value, direction, batch, args.mode)
            synchronize()
            cold = time.perf_counter() - start
            times, peaks = [], []

            for sample in range(args.repeats):
                for item in records.values():
                    item.clear()
                synchronize()
                if device.type == "cuda":
                    baseline = torch.cuda.memory_allocated(device)
                    torch.cuda.reset_peak_memory_stats(device)
                stage = f"warm sample {sample + 1}"
                start = time.perf_counter()
                with instrumentation(records):
                    output = execute(actual, value, direction, batch, args.mode)
                synchronize()
                times.append(time.perf_counter() - start)
                if device.type == "cuda":
                    peaks.append(
                        torch.cuda.max_memory_allocated(device) - baseline
                    )

            stage = "numerical parity"
            torch.testing.assert_close(output, expected, rtol=1e-8, atol=1e-8)
            result = dict(
                vars(args),
                torch=torch.__version__,
                dtype="float64",
                solver_source=implementation.__file__,
                status="ok",
                cold_seconds=cold,
                median_seconds=statistics.median(times),
                seconds=times,
                peak_incremental_bytes=max(peaks) if peaks else None,
                max_absolute_error=(output.detach() - expected)
                .abs()
                .max()
                .item(),
                **records,
            )
            print(json.dumps(result, allow_nan=False), flush=True)
    except Exception as error:
        diagnostic = dict(
            vars(args),
            status="error",
            stage=stage,
            error_type=type(error).__name__,
            message=str(error),
            diagnostics=records,
        )
        print(json.dumps(diagnostic), file=sys.stderr, flush=True)
        traceback.print_exc()
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
