# Reproducible solver measurements

`benchmark.py` is opt-in and emits one JSON result per invocation. It covers
deterministic SPD solves, float32 refinement, genuine exact fallback,
cached linearizations, and matrix-free EEQ water clusters. No raw results
belong in the repository. EEQ additionally requires merged tad-multicharge.

Export the PR base, then run **the same script/environment** against both
source trees. For example, from the tad-mctc checkout:

```sh
mkdir -p /tmp/mctc-base
git archive 010833d70be6e27336877d9d528d2de7c316ca98 | tar -x -C /tmp/mctc-base
PYTHONPATH=/tmp/mctc-base/src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python tools/krylov/benchmark.py --case refined --mode jacrev
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python tools/krylov/benchmark.py --case refined --mode jacrev
```

Use `--case fallback` to force exact fallback. Use `--case linearization
--mode vmap --size 768`, with and without `--mapped-parameter`, to measure
cached products versus repeated ordinary matvec setup. Compare an unbatched
control with `--mode eager`. Operator matrices are SPD by construction;
references use independent Cholesky solves with native AD.

For EEQ, repeat each command against the two source trees:

```sh
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python tools/krylov/benchmark.py --case eeq --size 99 --device cuda --mode jacrev
```

Repeat for sizes 300 and 999 and modes `native`, `jacfwd`, and `vmap`.
`jacfwd` uses a single displacement direction; `jacrev` uses a scalar
observable. Water geometry, charge -1, sparse coordination neighbors,
solver defaults (`--maxiter` applies only to synthetic cases) and `sum(energy) + 0.01 * sum(charges**2)` match PR #69.

The default is three warm repeats after one separately reported cold run.
CUDA timing synchronizes both boundaries. Dense references run outside the
timed regions. Memory is peak incremental allocated CUDA memory above the
sample baseline; CPU peak memory is not measured. Records include exact,
fast and cached products, cache setup, fast/fallback solver invocations,
true residuals, targets, iteration diagnostics and individual sample times.
Numerical disagreement, nonfinite results and other failures emit the case,
stage, exception type/message and traceback to stderr and exit nonzero.
There is no subprocess-output parser or ignored malformed-output branch.

Diagnostic snapshots unwrap functorch tensors only for readout. They never
change numerical inputs or control flow. Production uses public batching
rules. Run benchmarks without concurrent GPU tests or other GPU workloads.

Shared operator parameters retain their cache and dynamic termination.
With mapped operator parameters and no fast operator, numerical execution
keeps the base's cached fixed-budget path. This avoids replacing an expensive
cache with repeated ordinary-matvec setup. With a fast operator, mapped
factories are still bypassed for exact residual checks, since public vmap
cannot return per-system Python closures. Measure expensive setup in that
configuration. A general mapped-cache interface needs separate design work.
Implicit AD remains independent of numerical iterations and uses the original
operator. Compilation retains fixed-budget execution.
