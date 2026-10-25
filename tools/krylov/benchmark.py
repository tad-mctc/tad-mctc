# SPDX-Identifier: CC0-1.0
"""Opt-in numerical-work benchmark; see README.md for paired measurements."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

import torch
from torch.func import jacfwd, jacrev, jvp, vmap

from tad_mctc.storch.linsolve import constrained_solve, implicit_solve, krylov
from tad_mctc.storch.linsolve import implicit as implementation

MODES = (
    "eager",
    "native",
    "jvp",
    "jacrev",
    "jacfwd",
    "vmap",
    "vmap-jacrev",
    "jacrev-vmap",
    "mixed",
)
CASES = (
    "spd",
    "conditioned",
    "projected",
    "preconditioned",
    "refined",
    "fallback",
    "no-correction",
    "limit",
    "breakdown",
    "singular",
    "stagnation",
    "different-rates",
    "matrix-free",
)


def snapshot(tensor: torch.Tensor | None) -> Any:
    """Read diagnostics only; never unwrap a numerical solver input."""
    if tensor is None:
        return None
    ft = torch._C._functorch
    while ft.is_functorch_wrapped_tensor(tensor):
        tensor = ft.get_unwrapped(tensor)
    with torch._C._DisableFuncTorch():
        return tensor.detach().cpu().tolist()


@contextmanager
def count_iterations(
    records: list[dict[str, Any]], refinements: list[dict[str, Any]]
) -> Iterator[None]:
    """Record every actual Krylov invocation, including correction/fallback."""
    originals = dict(implementation._METHODS)
    projected = krylov.projected_cg
    original_refine = implementation._refine

    def record(method: Callable[..., Any]) -> Callable[..., Any]:
        def measured(*args: Any, **kwargs: Any) -> Any:
            result = method(*args, **kwargs)
            diagnostics = {
                name: snapshot(getattr(result, name))
                for name in ("status", "iterations", "residual", "target")
            }
            diagnostics["rtol"] = kwargs.get("rtol")
            diagnostics["operator"] = getattr(
                args[0], "__name__", type(args[0]).__name__
            )
            records.append(diagnostics)
            return result

        return measured

    def record_refinement(*args: Any, **kwargs: Any) -> Any:
        start = len(records)
        result = original_refine(*args, **kwargs)
        inner_rtol = args[0].refinement[0]
        solves = records[start:]
        fast_solves = sum(solve["rtol"] == inner_rtol for solve in solves)
        refinements.append(
            dict(
                fast_solves=fast_solves,
                corrections=max(0, fast_solves - 1),
                exact_fallback_calls=len(solves) - fast_solves,
                refinement_status=snapshot(result.refinement_status),
                fallback_iterations=snapshot(result.fallback_iterations),
            )
        )
        return result

    with patch(
        "tad_mctc.storch.linsolve.implicit._refine", record_refinement
    ), patch.dict(
        implementation._METHODS,
        {name: record(method) for name, method in originals.items()},
    ), patch(
        "tad_mctc.storch.linsolve.constrained.projected_cg", record(projected)
    ):
        yield


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark(args: argparse.Namespace, case: str, mode: str) -> dict[str, Any]:
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    n = args.size
    dd = {"device": device, "dtype": dtype}
    generator = torch.Generator().manual_seed(0)
    random = torch.randn(n, n, generator=generator, dtype=torch.double).to(**dd)
    orthogonal = torch.linalg.qr(random).Q
    condition = 1000.0 if case == "conditioned" else 4.0
    spectrum = torch.logspace(
        0.0, torch.log10(torch.tensor(condition)).item(), n, **dd
    )
    matrix = (orthogonal * spectrum) @ orthogonal.mT
    if case == "no-correction":
        matrix = torch.eye(n, **dd)
    if case == "breakdown":
        matrix = -matrix
    if case == "singular":
        matrix = torch.zeros_like(matrix)
    if case == "different-rates":
        matrix = torch.stack((torch.eye(n, **dd), matrix))
    if case == "matrix-free":
        matrix = (
            3 * torch.eye(n, **dd)
            + 0.1 * torch.eye(n, **dd).roll(1, -1)
            + 0.1 * torch.eye(n, **dd).roll(-1, -1)
        )
    rhs = torch.linspace(1.0, 2.0, n, **dd)
    if case == "different-rates":
        rhs = rhs.expand(2, n)
    weight = torch.linspace(-0.5, 1.0, n, **dd)
    constraint = torch.ones(1, n, **dd)
    value = torch.tensor([0.3], **dd)
    zero = torch.zeros((), **dd)
    offsets = torch.tensor([0.0, 0.1], **dd)
    calls: Counter[str] = Counter()
    results = []
    inner = []
    refinement_records = []
    preconditioned = case == "preconditioned"
    refined = case in ("refined", "fallback", "stagnation", "no-correction")
    rtol, atol = 1e-10, 1e-12

    def matvec(
        vector: torch.Tensor, operator: torch.Tensor, shift: torch.Tensor
    ) -> torch.Tensor:
        calls["exact"] += 1
        if case == "matrix-free":
            return (3 + shift) * vector + 0.1 * (
                vector.roll(1, -1) + vector.roll(-1, -1)
            )
        return (
            torch.einsum("...ij,...j->...i", operator, vector) + shift * vector
        )

    def fast(
        vector: torch.Tensor, operator: torch.Tensor, shift: torch.Tensor
    ) -> torch.Tensor:
        calls["fast"] += 1
        if case in ("fallback", "stagnation"):
            factor = -1.0 if case == "fallback" else 0.2
            return factor * (
                torch.einsum("...ij,...j->...i", operator, vector)
                + shift * vector
            )
        return (
            torch.einsum("...ij,...j->...i", operator.float(), vector.float())
            + shift.float() * vector.float()
        ).to(dtype)

    def precondition(
        vector: torch.Tensor, operator: torch.Tensor, shift: torch.Tensor
    ) -> torch.Tensor:
        calls["preconditioner"] += 1
        return vector / (operator.diagonal(dim1=-2, dim2=-1) + shift)

    def objective(shift: torch.Tensor) -> torch.Tensor:
        options = dict(
            rtol=rtol,
            atol=atol,
            maxiter=1 if case == "limit" else args.maxiter,
            fast_matvec=fast if refined else None,
            fallback=case != "stagnation",
            preconditioner=precondition if preconditioned else None,
        )
        if case == "projected":
            result = constrained_solve(
                matvec, rhs, constraint, value, matrix, shift, **options
            )
        else:
            result = implicit_solve(
                matvec,
                rhs,
                matrix,
                shift,
                method=args.method,
                x0=rhs if case == "no-correction" else None,
                **options,
            )
        results.append(
            {
                name: snapshot(getattr(result, name))
                for name in (
                    "status",
                    "residual",
                    "target",
                    "iterations",
                    "refinement_status",
                    "fallback_iterations",
                )
            }
        )
        return (weight * result.solution).sum()

    def reference(shift: torch.Tensor) -> torch.Tensor:
        shifted = matrix + shift * torch.eye(n, **dd)
        if case == "projected":
            bordered = torch.cat(
                (
                    torch.cat((shifted, constraint.mT), -1),
                    torch.cat((constraint, torch.zeros(1, 1, **dd)), -1),
                ),
                -2,
            )
            lu, pivots = torch.linalg.lu_factor(bordered)
            solution = torch.linalg.lu_solve(
                lu, pivots, torch.cat((rhs, value)).unsqueeze(-1)
            ).squeeze(-1)[:-1]
        else:
            factor = torch.linalg.cholesky(shifted)
            solution = torch.cholesky_solve(rhs.unsqueeze(-1), factor).squeeze(
                -1
            )
        return (weight * solution).sum()

    def execute(
        function: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        if mode == "eager":
            return function(zero)
        if mode == "native":
            parameter = zero.clone().requires_grad_()
            output = function(parameter)
            output.backward()
            assert parameter.grad is not None
            return parameter.grad
        if mode == "jvp":
            _, tangent = jvp(function, (zero,), (torch.ones_like(zero),))
            return tangent
        if mode == "jacrev":
            gradient = jacrev(function)
            return gradient(zero)
        if mode == "jacfwd":
            derivative = jacfwd(function)
            return derivative(zero)
        if mode == "vmap":
            evaluate_batch = vmap(function)
            return evaluate_batch(offsets)
        if mode == "vmap-jacrev":
            gradient = jacrev(function)
            evaluate_gradients = vmap(gradient)
            return evaluate_gradients(offsets)
        if mode == "jacrev-vmap":
            evaluate_batch = vmap(function)

            def shared(parameter: torch.Tensor) -> torch.Tensor:
                return evaluate_batch(parameter + offsets)

            gradient = jacrev(shared)
            return gradient(zero)

        gradient = jacrev(function)
        hessian = jacfwd(gradient)
        return hessian(zero)

    synchronize(device)
    start = time.perf_counter()
    execute(objective)
    synchronize(device)
    cold = time.perf_counter() - start
    expected = (
        None
        if case in ("breakdown", "singular", "stagnation", "limit")
        else execute(reference).detach()
    )
    times, peaks = [], []
    for _ in range(args.repeats):
        calls.clear()
        results.clear()
        inner.clear()
        refinement_records.clear()
        synchronize(device)
        if device.type == "cuda":
            baseline = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        with count_iterations(inner, refinement_records):
            output = execute(objective)
        synchronize(device)
        times.append(time.perf_counter() - start)
        if device.type == "cuda":
            peaks.append(torch.cuda.max_memory_allocated(device) - baseline)
    return dict(
        case=case,
        mode=mode,
        torch=torch.__version__,
        device=str(device),
        dtype=str(dtype),
        method="projected_cg" if case == "projected" else args.method,
        size=n,
        batch_size=(2 if case == "different-rates" else 1)
        * (2 if "vmap" in mode else 1),
        preconditioner="jacobi" if preconditioned else None,
        rtol=rtol,
        atol=atol,
        maxiter=1 if case == "limit" else args.maxiter,
        refined=refined,
        cold_seconds=cold,
        median_seconds=statistics.median(times),
        min_seconds=min(times),
        max_seconds=max(times),
        peak_incremental_bytes=max(peaks) if peaks else None,
        error=(
            None
            if expected is None
            else (output.detach() - expected).abs().max().item()
        ),
        operator_calls=dict(calls),
        public_results=results,
        inner_solves=inner,
        refinement_sequences=refinement_records,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--dtype", choices=("float32", "float64"), default="float64"
    )
    parser.add_argument(
        "--method", choices=("cg", "bicgstab", "gmres"), default="cg"
    )
    parser.add_argument("--cases", choices=CASES, nargs="+", default=CASES)
    parser.add_argument("--modes", choices=MODES, nargs="+", default=MODES)
    parser.add_argument("--size", type=int, default=12)
    parser.add_argument("--maxiter", type=int, default=80)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if min(args.size, args.maxiter, args.repeats) < 1:
        parser.error("size, maxiter and repeats must be positive")
    torch.set_num_threads(1)
    for case in args.cases:
        for mode in args.modes:
            try:
                result = benchmark(args, case, mode)
            except Exception as error:
                result = dict(
                    case=case,
                    mode=mode,
                    error_type=type(error).__name__,
                    error=str(error),
                )
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
