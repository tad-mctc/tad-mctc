# Changelog

## Unreleased

### Changed

- **`NeighborList.idx_i` and `idx_j` are stored as `int32`** (were `int64`),
  which cuts a list from 17 to 9 bytes per slot (23 to 15 for a periodic
  one). The native kernel writes `int32` directly, so a build never holds an
  `int64` copy. `estimate_neighborlist_memory` follows.
  - **External code must pass a slice of `nbl.idx_i` / `nbl.idx_j` through
    `tad_mctc.neighbor.gather_index` (or call `.long()` on it) before
    indexing with it** (`index_select`, `index_add`, `gather`, advanced
    indexing). PyTorch before 2.8 rejects an `int32` index in the backward
    pass (`gather(): Expected dtype int64 for index`) under
    `torch.autograd`, `jacrev`, `jacfwd` and `vmap`. `gather_index` returns
    the slice itself from PyTorch 2.8 on and an `int64` copy before;
    `.long()` always copies. Widen one slice at a time, not the whole list.
    `NeighborList.real_entries()` returns `int32` indices.
  - From PyTorch 2.8, `sum_over_neighborlist` and `pair_distance_squared`
    use the `int32` slices directly, so a backward pass keeps no index
    copies. Before 2.8 they widen each chunk, and autograd keeps those
    copies until the backward pass (16 bytes per pair, more than an `int64`
    list needed).
  - The indices handed to a `sum_over_neighborlist` callback are `int32`
    from PyTorch 2.8 on. On CPU, an `int32` gather from rows of a table
    (such as `(n, 3)` positions) has a ~4x slower backward pass than an
    `int64` one; gather from one column per quantity instead.
  - `NeighborList.create` still accepts `int64` indices (an existing list):
    they are range-checked and stored as `int32`.
  - A build raises `ValueError` if the largest stored index (the flattened,
    batched atom count, or the size of a periodic search's ghost pool, each
    including the padding index) exceeds `2**31 - 1`. There is no `int64`
    fallback. Pair counts and capacities are unaffected and stay 64-bit.
- `pair_distance_squared` forms the image translation column by column
  (`shift[0] * cell[0, k] + shift[1] * cell[1, k] + shift[2] * cell[2, k]`)
  instead of through an `(n, 3)` matrix product: with periodic boundary
  conditions the kernel takes 54 instead of 61 ns per pair for one cell and 70
  instead of 97 ns for a cell per system (131k float64 pairs, one CPU thread,
  forward and backward); with a lattice that requires a gradient, the cell per
  system takes 133 instead of 220 ns. Values and gradients of the positions
  are the same bits; the gradient of a single shared lattice can differ in the
  last digits (relative 1e-15).
- `pair_distance_squared` gathers through `gather_rows`, which is `table[index]`
  instead of `index_select` while `torch.compile` traces on PyTorch before
  2.14. Compiled `jacrev` of an `index_select` is wrong on PyTorch 2.5 to 2.13
  (every Jacobian row is the sum of all rows), which gave wrong gradients for
  `torch.compile(jacrev(cn))` of the sparse coordination number. Eager code
  and 2.14 and later are unchanged.
- The neighbour-list form of the cold-fusion check works in blocks of pairs.
- `pair_distance_squared` gathers each coordinate from its own column
  (`position_columns`, `pair_distance_squared_from_columns`) instead of rows
  of the `(n, 3)` positions. On CPU (one thread, PyTorch 2.14), the sparse
  coordination number without gradients took 35.3 instead of 40.6 s for
  967M pairs (one run). With gradients, for 47M pairs, the forward pass is
  ~7% slower (5.1 vs 4.8 s) and the backward pass ~30% faster (2.4 vs
  3.5 s). Values are unchanged; gradients of large systems can differ in the
  last bit (summation order).

### Fixed

- `pair_distance_squared` sums the three components by hand instead of
  reducing over an axis of length 3, which PyTorch evaluates about 10x
  slower. The sparse coordination number is about 2x faster, with identical
  results.

### Added

- `tad_mctc --compile` evaluates the coordination number through
  `torch.compile(fullgraph=True)`, timing the compilation as its own step.
