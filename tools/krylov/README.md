# Krylov transform benchmarks

These opt-in scripts count real operator work, including work after the
reported iteration count stops increasing. They are not CI tests. Raw JSONL
output belongs outside the repository.

## Paired measurements

Use the same Python environment, device, dtype, inputs and command for both
revisions. The dependency supports PyTorch 2.7 and later; the optimization
uses the documented [`Function.vmap` batching rule](https://docs.pytorch.org/docs/stable/notes/extending.func.html).
For example, export the reference source without changing your checkout:

```sh
mkdir -p /tmp/mctc-reference
git archive 010833d70be6e27336877d9d528d2de7c316ca98 | tar -x -C /tmp/mctc-reference

PYTHONPATH=/tmp/mctc-reference/src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python tools/krylov/benchmark.py --repeats 5 > /tmp/krylov-before.jsonl
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python tools/krylov/benchmark.py --repeats 5 > /tmp/krylov-after.jsonl
```

`benchmark.py` covers SPD conditioning, projected CG, preconditioning,
float32 refinement, genuine exact fallback, an already accepted initial
guess, iteration exhaustion, breakdown, singularity, stagnation and a batch
with different convergence rates. The circular three-point operator in
`matrix-free` applies a stencil without allocating a matrix in its products;
a dense reference is prepared separately. Select `--method bicgstab` or
`--method gmres` to exercise the other implicit Krylov methods. Select
`--device cuda --dtype float32` for a CUDA single-precision run.

Each execution returns a scalar observable or a directional/scalar
parameter derivative. `vmap` evaluates two parameter values; `jacrev(vmap)`
shares one differentiable input across them. Full large Jacobians are never
constructed. References use independent dense factorizations; deliberately
failed cases report diagnostics rather than asserting solution accuracy.
Exceptions are emitted as explicit error rows, so baseline transform
failures remain visible instead of silently disappearing.

Records include requested tolerances, solver status, true residual, target,
reported iterations, exact/fast/preconditioner calls, every inner solve,
refinement corrections, and exact fallback calls and iterations. A correction
count excludes the initial fast solve. An exact fallback call with zero
reported iterations still represents actual operator work.

## Matrix-free EEQ integration

Install the merged `tad-multicharge` package in the same isolated environment.
The script does not change either package's dependency requirements.

```sh
PYTHONPATH=/tmp/mctc-reference/src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python tools/krylov/eeq.py --device cuda --sizes 99 300 999 \
  --modes eager native jvp jacrev jacfwd vmap mixed --repeats 5 \
  --timeout 600 --output /tmp/eeq-before.jsonl
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python tools/krylov/eeq.py --device cuda --sizes 99 300 999 \
  --modes eager native jvp jacrev jacfwd vmap mixed --repeats 5 \
  --timeout 600 --output /tmp/eeq-after.jsonl
```

This reproduces the PR #69 water clusters: fixed liquid-density geometry in
bohr, charge -1, sparse coordination neighbors, and the scalar objective
`sum(atomic_energies) + 0.01 * sum(charges**2)`. Forward AD differentiates one
normalized displacement, while mixed AD differentiates a force projection
with respect to hydrogen hardness. Mixed derivatives default to at most 300
atoms. Model defaults, tolerances, memory budgets, mixed precision and
fallback stay unchanged. `native-jacobi` is an optional separate control
that disables only the coarse preconditioner memory budget.

Each EEQ case runs in a fresh subprocess with a deadline. Initialize CUDA
before timing; report the cold/compiling sample separately. Dense references
and their derivatives run outside measured regions. CUDA measurements
synchronize at both boundaries. Compare warm medians and the individual
samples (or their min/max), rather than cold initialization. Memory is peak
incremental *allocated* CUDA memory above the pre-sample baseline, not the
allocator's reserved pool. CPU peak memory is not measured.

Instrumentation delegates to the original operators. Private functorch
unwrapping in these scripts is confined to **reading diagnostics** for JSON;
it never changes solver inputs, control flow, or derivatives. Production
batch promotion uses public custom batching rules.

## Scope of the optimization

Numerical mapped inputs become ordinary batch dimensions before the
implicit solve. The existing loops then stop when all lanes are inactive,
skip unnecessary correction solves and omit an unneeded exact fallback.
All original residual checks, per-lane masks and failure diagnostics remain.
The numerical history is detached; the existing implicit rules attach AD.

Direct low-level Krylov calls under `vmap`, explicitly requested
`early_exit=False`, and compilation retain fixed-budget execution. A slow or
failed active lane can still require work for the whole explicit batch; this
change does not compact lanes or cache preconditioners. Operator parameters
that vary across mapped systems should be passed explicitly, as required for
their implicit derivatives. A cached per-system linearization is not mapped
as a Python closure; batching retains the existing operator-mapping policy.

`torch.while_loop` is not a supported replacement across these version
boundaries: standalone probes of `vmap`, `jacrev` and `jacfwd` fail on both
2.7.1 and 2.14.1. PyTorch also documents it as a
[prototype without training support](https://docs.pytorch.org/docs/main/higher_order_ops/while_loop.html).
Whole-model compilation, compiled higher derivatives, and geometry-dependent
preconditioner reuse remain separate work packages.
