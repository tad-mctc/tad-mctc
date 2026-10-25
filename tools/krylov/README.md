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
solver defaults and `sum(energy) + 0.01 * sum(charges**2)` match PR #69.

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

Shared operator parameters retain their numerical cache under mapped solves.
If any operator parameter is mapped, the factory is bypassed: public vmap
cannot return per-system Python closures. Expensive setup inside ordinary
matvecs can therefore outweigh the iteration savings. Both ordinary and
cached operators must remain mathematically equivalent; implicit AD always
uses the original operator. A general mapped-cache interface requires
separate design work. Compilation retains fixed-budget execution.
