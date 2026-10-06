// This file is part of tad-mctc.
//
// SPDX-Identifier: Apache-2.0
// Copyright (C) 2024 Grimme Group
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// OpenMP-parallel equivalent of the per-candidate filter-and-compact body
// in `tad_mctc.neighbor.list._atom_pairs_within_thresholds` (the
// `kernel.compute` -> `keep` -> `nonzero` -> two fancy-index gathers
// sequence). See `_native.py`'s module docstring for why this exists, why
// it is optional, and exactly what output it must reproduce.
//
// The search runs in two passes over the candidate tile pairs:
//
// 1. Compute the distances of every candidate and record, for every row of
//    tile A and every threshold, which columns of tile B are within it, as
//    a bit mask (the row's "hit mask"). The set bits count the candidate's
//    pairs, and a prefix sum over the counts gives each candidate its first
//    slot in the output.
// 2. Read the hit masks and write every candidate's pairs straight into its
//    slots of the final tensors.
//
// Because every candidate knows its slots, the output order does not depend
// on which thread handled a candidate, so both passes can balance their
// work dynamically. Writing straight into the final tensors needs no
// per-thread buffers, and the second pass computes no distances. The hit
// masks take one bit per row, column and threshold, 128 bytes per candidate
// and threshold for tiles of 32 atoms, which is less than a tenth of the
// memory of the pairs they describe in dense and in periodic systems.
//
// The build turns fast math and floating-point contraction off
// (`-fno-fast-math -ffp-contract=off`), so that the compiler evaluates every
// squared distance exactly as `squared_norm` writes it: the box skip then
// drops only pairs the exact test would drop too, and the distances match
// the Python path's. Intel's `icpx` enables fast math by default, and under
// it `-ffp-contract=off` alone does not stop multiply-adds.
//
// The search writes every byte of its large buffers (the hit masks and the
// output) once. On Linux, it asks for huge pages for them, which spares
// most of the kernel's page faults on that first write (see
// `request_huge_pages`).
//
// An optional per-atom `anchor` mask drops every pair without an anchor
// atom. A periodic search passes the primary-cell atoms of its ghost pool,
// so pairs of two periodic images never reach the output.

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <tuple>
#include <vector>

#include <omp.h>

#ifdef __linux__
#include <sys/mman.h>
#endif

namespace {

// Candidates handed to a thread at a time. Neighbouring candidates differ a
// lot in cost (a dense core against a sparse surface), so the passes
// schedule dynamically; a small chunk keeps the threads balanced, and 16
// candidates are enough work to make the scheduling overhead negligible.
constexpr int64_t kCandidatesPerChunk = 16;

// `dx * dx + dy * dy + dz * dz`, the one formula for every squared
// distance in the scan. A pair's distance and an atom's distance to a
// tile's bounding box both go through it, so they round the same way:
// the box distance of an atom can then never exceed its computed distance
// to any atom in that box, and skipping by the box drops only pairs that
// the exact test would drop too.
template <typename scalar_t>
inline scalar_t squared_norm(scalar_t dx, scalar_t dy, scalar_t dz) {
  return dx * dx + dy * dy + dz * dz;
}

// Squared distance from a point to an axis-aligned box, zero inside it.
template <typename scalar_t>
inline scalar_t squared_distance_to_box(
  const scalar_t *lo,
  const scalar_t *hi,
  scalar_t x,
  scalar_t y,
  scalar_t z
) {
  const scalar_t gap_x =
    std::max<scalar_t>(lo[0] - x, 0) + std::max<scalar_t>(x - hi[0], 0);
  const scalar_t gap_y =
    std::max<scalar_t>(lo[1] - y, 0) + std::max<scalar_t>(y - hi[1], 0);
  const scalar_t gap_z =
    std::max<scalar_t>(lo[2] - z, 0) + std::max<scalar_t>(z - hi[2], 0);
  return squared_norm(gap_x, gap_y, gap_z);
}

// The coordinates of every tile, packed so that the x (y, z) coordinates
// of one tile's slots are contiguous, and the bounding box of its real
// atoms. Built from the same positions the distances are computed from,
// which matters for a batched search: its `Tiles` were built on shifted
// coordinates.
template <typename scalar_t> struct TileGeometry {
    int64_t tile_width;
    std::vector<scalar_t> xyz; // (ntile, 3, tile_width)
    std::vector<scalar_t> lo;  // (ntile, 3)
    std::vector<scalar_t> hi;  // (ntile, 3)

    const scalar_t *x(int64_t t) const {
      return xyz.data() + t * 3 * tile_width;
    }
    const scalar_t *y(int64_t t) const { return x(t) + tile_width; }
    const scalar_t *z(int64_t t) const { return x(t) + 2 * tile_width; }
    const scalar_t *box_lo(int64_t t) const { return lo.data() + t * 3; }
    const scalar_t *box_hi(int64_t t) const { return hi.data() + t * 3; }
};

template <typename scalar_t>
TileGeometry<scalar_t> pack_tiles(
  const int64_t *index_ptr,
  const bool *valid_ptr,
  const scalar_t *pos_ptr,
  int64_t ntile,
  int64_t tile_width,
  int nthreads
) {
  TileGeometry<scalar_t> geometry{
    tile_width,
    std::vector<scalar_t>(static_cast<size_t>(ntile * 3 * tile_width)),
    std::vector<scalar_t>(static_cast<size_t>(ntile * 3)),
    std::vector<scalar_t>(static_cast<size_t>(ntile * 3)),
  };

#pragma omp parallel for num_threads(nthreads) schedule(static)
  for (int64_t t = 0; t < ntile; ++t) {
    scalar_t *xyz = geometry.xyz.data() + t * 3 * tile_width;
    scalar_t *lo = geometry.lo.data() + t * 3;
    scalar_t *hi = geometry.hi.data() + t * 3;
    for (int axis = 0; axis < 3; ++axis) {
      lo[axis] = std::numeric_limits<scalar_t>::max();
      hi[axis] = std::numeric_limits<scalar_t>::lowest();
    }

    // Padded slots repeat the tile's first atom (see `Tiles.index`), so
    // they hold real coordinates; `valid` keeps them out of the box.
    for (int64_t slot = 0; slot < tile_width; ++slot) {
      const int64_t atom = index_ptr[t * tile_width + slot];
      const bool real = valid_ptr[t * tile_width + slot];
      for (int axis = 0; axis < 3; ++axis) {
        const scalar_t value = pos_ptr[atom * 3 + axis];
        xyz[axis * tile_width + slot] = value;
        if (real) {
          lo[axis] = std::min(lo[axis], value);
          hi[axis] = std::max(hi[axis], value);
        }
      }
    }
  }
  return geometry;
}

// Buffers at least this large are advised to use huge pages. glibc serves
// every allocation above 32 MB from its own `mmap`, so the advice ends with
// the buffer and never reaches memory that other code allocates later.
constexpr size_t kMinHugePageBuffer = size_t{64} << 20;

// Asks Linux to back the 2 MB-aligned interior of a freshly allocated
// buffer with huge pages. With 4 KB pages, the page faults of the first
// write to the output cost about a third of a single-threaded search; one
// 2 MB page takes the place of 512 of them. The advice covers only this
// buffer's own pages, and changes nothing where transparent huge pages
// are off or on other systems.
inline void request_huge_pages(void *data, size_t n_bytes) {
#ifdef MADV_HUGEPAGE
  if (n_bytes < kMinHugePageBuffer) { return; }
  constexpr uintptr_t kHugePage = uintptr_t{2} << 20;
  const auto first = reinterpret_cast<uintptr_t>(data);
  const uintptr_t begin = (first + kHugePage - 1) & ~(kHugePage - 1);
  const uintptr_t end = (first + n_bytes) & ~(kHugePage - 1);
  // Advice only: if the kernel refuses it, the buffer keeps 4 KB pages.
  madvise(reinterpret_cast<void *>(begin), end - begin, MADV_HUGEPAGE);
#else
  static_cast<void>(data);
  static_cast<void>(n_bytes);
#endif
}

// One word of a hit mask. A row of tile A takes `ceil(tile_width / 32)`
// words: bit `col % kBitsPerWord` of word `col / kBitsPerWord` is set if the
// row pairs with column `col` of tile B.
using MaskWord = uint32_t;
constexpr int64_t kBitsPerWord = 32;

// `bit` if `keep`, otherwise zero, without a branch.
inline MaskWord bit_if(bool keep, MaskWord bit) {
  return bit & (MaskWord{0} - static_cast<MaskWord>(keep));
}

// The number of set bits, and the position of the lowest set bit of a
// nonzero word. GCC, Clang and Intel's `icpx` all provide these builtins.
inline int64_t count_bits(MaskWord word) { return __builtin_popcount(word); }
inline int64_t lowest_bit(MaskWord word) { return __builtin_ctz(word); }

// The hit masks of every candidate, threshold and row of tile A, row after
// row, `n_words` words per row.
struct HitMasks {
    int64_t n_thresholds;
    int64_t tile_width;
    int64_t n_words;
    std::unique_ptr<MaskWord[]> words;

    HitMasks(int64_t n_candidates, int64_t n_thresholds, int64_t tile_width)
      : n_thresholds(n_thresholds), tile_width(tile_width),
        n_words((tile_width + kBitsPerWord - 1) / kBitsPerWord),
        // Not initialised: pass 1 writes every word.
        words(new MaskWord[static_cast<size_t>(
          n_candidates * n_thresholds * tile_width * n_words
        )]) {
      request_huge_pages(
        words.get(),
        static_cast<size_t>(
          n_candidates * n_thresholds * tile_width * n_words
        ) *
          sizeof(MaskWord)
      );
    }

    // The masks of candidate `c` for threshold `k`, from its first row on.
    MaskWord *rows(int64_t c, int64_t k) const {
      return words.get() + (c * n_thresholds + k) * tile_width * n_words;
    }
};

// One thread's working space for a candidate: the columns of tile B that
// can pair with tile A, compacted in ascending column order, and the
// squared distances of the current row to them.
template <typename scalar_t> struct RowScratch {
    std::vector<int64_t> near_col;
    std::vector<int64_t> near_atom;
    std::vector<MaskWord> near_bit;  // the column's bit in its mask word
    std::vector<int64_t> word_start; // first near column of each mask word
    std::vector<scalar_t> near_x;
    std::vector<scalar_t> near_y;
    std::vector<scalar_t> near_z;
    std::vector<scalar_t> distance_sq;

    RowScratch(int64_t tile_width, int64_t n_words)
      : near_col(static_cast<size_t>(tile_width)),
        near_atom(static_cast<size_t>(tile_width)),
        near_bit(static_cast<size_t>(tile_width)),
        word_start(static_cast<size_t>(n_words + 1)),
        near_x(static_cast<size_t>(tile_width)),
        near_y(static_cast<size_t>(tile_width)),
        near_z(static_cast<size_t>(tile_width)),
        distance_sq(static_cast<size_t>(tile_width)) {}
};

// The row of one candidate currently visited, see `for_each_row`: its slot
// in tile A, whether its atom is an anchor, and the range `[first, n_near)`
// of near columns it can pair with.
struct Row {
    int64_t slot;
    bool is_anchor;
    int64_t first;
    int64_t n_near;
};

// Everything both passes read: the tiles, the candidate tile pairs, the
// anchor mask and the thresholds.
template <typename scalar_t> struct CandidateSearch {
    const int64_t *index_ptr;
    const bool *valid_ptr;
    const int64_t *tile_a_ptr;
    const int64_t *tile_b_ptr;
    const TileGeometry<scalar_t> &geometry;
    const bool *anchor_ptr; // `nullptr` keeps every pair
    int64_t tile_width;
    int64_t n_words; // hit mask words per row
    const std::vector<scalar_t> &thresholds_sq;
    scalar_t max_threshold_sq;

    // Calls `on_row(row)` for every row of candidate `c` that can pair
    // with tile B, once `scratch.distance_sq[row.first, row.n_near)` holds
    // the row's squared distances to the near columns.
    template <typename OnRow>
    void for_each_row(
      int64_t c,
      RowScratch<scalar_t> &scratch,
      OnRow &&on_row
    ) const {
      const int64_t a = tile_a_ptr[c];
      const int64_t b = tile_b_ptr[c];
      const int64_t *index_a = index_ptr + a * tile_width;
      const int64_t *index_b = index_ptr + b * tile_width;
      const bool *valid_a = valid_ptr + a * tile_width;
      const bool *valid_b = valid_ptr + b * tile_width;
      const scalar_t *x_a = geometry.x(a), *y_a = geometry.y(a),
                     *z_a = geometry.z(a);
      const scalar_t *x_b = geometry.x(b), *y_b = geometry.y(b),
                     *z_b = geometry.z(b);

      // Skip padded slots, and atoms farther from tile A's box than the
      // largest threshold: those are farther than that from every atom
      // of tile A.
      int64_t n_near = 0;
      for (int64_t col = 0; col < tile_width; ++col) {
        if (!valid_b[col]) { continue; }
        const scalar_t to_box_a = squared_distance_to_box(
          geometry.box_lo(a), geometry.box_hi(a), x_b[col], y_b[col], z_b[col]
        );
        if (to_box_a > max_threshold_sq) { continue; }

        const auto q = static_cast<size_t>(n_near);
        scratch.near_col[q] = col;
        scratch.near_atom[q] = index_b[col];
        scratch.near_bit[q] = MaskWord{1} << (col % kBitsPerWord);
        scratch.near_x[q] = x_b[col];
        scratch.near_y[q] = y_b[col];
        scratch.near_z[q] = z_b[col];
        ++n_near;
      }

      // The near columns ascend, so those of one mask word are contiguous:
      // word `w` holds the near columns `[word_start[w], word_start[w + 1])`.
      int64_t q = 0;
      for (int64_t w = 0; w <= n_words; ++w) {
        while (q < n_near &&
               scratch.near_col[static_cast<size_t>(q)] < w * kBitsPerWord) {
          ++q;
        }
        scratch.word_start[static_cast<size_t>(w)] = q;
      }

      for (int64_t row = 0; row < tile_width; ++row) {
        if (!valid_a[row]) { continue; }
        const scalar_t to_box_b = squared_distance_to_box(
          geometry.box_lo(b), geometry.box_hi(b), x_a[row], y_a[row], z_a[row]
        );
        if (to_box_b > max_threshold_sq) { continue; }

        // Same-tile candidate: strict upper triangle only (`row < col`),
        // matching `list.py`'s `strict_upper_triangle` -- this excludes
        // self-pairs and, since both sides are the same atom set, avoids
        // listing both (i, j) and (j, i).
        int64_t first = 0;
        if (a == b) {
          while (first < n_near &&
                 scratch.near_col[static_cast<size_t>(first)] <= row) {
            ++first;
          }
        }

        // Accumulated in `scalar_t`, matching the Python reference
        // exactly: a pair almost exactly at the threshold can land on
        // either side of it depending on the precision of its distance.
        for (int64_t q = first; q < n_near; ++q) {
          const auto u = static_cast<size_t>(q);
          scratch.distance_sq[u] = squared_norm(
            x_a[row] - scratch.near_x[u],
            y_a[row] - scratch.near_y[u],
            z_a[row] - scratch.near_z[u]
          );
        }

        const bool is_anchor =
          anchor_ptr == nullptr || anchor_ptr[index_a[row]];
        on_row(Row{row, is_anchor, first, n_near});
      }
    }

    // Whether the current row's distance to near column `q` is within
    // `threshold_sq`.
    static bool within(
      const RowScratch<scalar_t> &scratch,
      int64_t q,
      scalar_t threshold_sq
    ) {
      return scratch.distance_sq[static_cast<size_t>(q)] <= threshold_sq;
    }

    bool
      partner_is_anchor(const RowScratch<scalar_t> &scratch, int64_t q) const {
      return anchor_ptr[scratch.near_atom[static_cast<size_t>(q)]];
    }

    // Writes the hit mask of `row` for `threshold_sq` to `mask` and returns
    // its number of pairs. A pair needs at least one anchor atom, so the
    // partner's anchor flag is read only for a row atom that is not an
    // anchor. Both loops are branch-free, so that they vectorise.
    int64_t hit_mask(
      const Row &row,
      const RowScratch<scalar_t> &scratch,
      scalar_t threshold_sq,
      MaskWord *mask
    ) const {
      int64_t n_pairs = 0;
      for (int64_t w = 0; w < n_words; ++w) {
        const int64_t begin =
          std::max(row.first, scratch.word_start[static_cast<size_t>(w)]);
        const int64_t end = scratch.word_start[static_cast<size_t>(w + 1)];
        MaskWord word = 0;
        if (row.is_anchor) {
          for (int64_t q = begin; q < end; ++q) {
            const bool keep = within(scratch, q, threshold_sq);
            word |= bit_if(keep, scratch.near_bit[static_cast<size_t>(q)]);
          }
        } else {
          for (int64_t q = begin; q < end; ++q) {
            const bool keep =
              within(scratch, q, threshold_sq) & partner_is_anchor(scratch, q);
            word |= bit_if(keep, scratch.near_bit[static_cast<size_t>(q)]);
          }
        }
        mask[w] = word;
        n_pairs += count_bits(word);
      }
      return n_pairs;
    }
};

// Pass 1: the hit masks of every candidate, and per threshold the offsets
// of its pairs, from a prefix sum over their numbers: `offsets[k][c]` is
// the first slot of candidate `c` in the list of threshold `k`, and
// `offsets[k][n_candidates]` is that list's number of pairs.
template <typename scalar_t>
std::vector<std::vector<int64_t>> record_hit_masks(
  const CandidateSearch<scalar_t> &search,
  int64_t n_candidates,
  int nthreads,
  HitMasks &masks
) {
  const size_t n_thresholds = search.thresholds_sq.size();
  std::vector<std::vector<int64_t>> offsets(
    n_thresholds, std::vector<int64_t>(static_cast<size_t>(n_candidates + 1))
  );
  const int64_t words_per_candidate =
    masks.n_thresholds * masks.tile_width * masks.n_words;

#pragma omp parallel num_threads(nthreads)
  {
    RowScratch<scalar_t> scratch(search.tile_width, search.n_words);
    std::vector<int64_t> counts(n_thresholds);

#pragma omp for schedule(dynamic, kCandidatesPerChunk)
    for (int64_t c = 0; c < n_candidates; ++c) {
      std::fill(counts.begin(), counts.end(), 0);
      // Rows that `for_each_row` skips have no pairs.
      std::fill_n(masks.rows(c, 0), words_per_candidate, MaskWord{0});
      search.for_each_row(c, scratch, [&](const Row &row) {
        for (size_t k = 0; k < n_thresholds; ++k) {
          MaskWord *mask =
            masks.rows(c, static_cast<int64_t>(k)) + row.slot * masks.n_words;
          counts[k] +=
            search.hit_mask(row, scratch, search.thresholds_sq[k], mask);
        }
      });
      for (size_t k = 0; k < n_thresholds; ++k) {
        offsets[k][static_cast<size_t>(c + 1)] = counts[k];
      }
    }
  }

  for (auto &offsets_k : offsets) {
    std::partial_sum(offsets_k.begin(), offsets_k.end(), offsets_k.begin());
  }
  return offsets;
}

// Pass 2: every candidate writes the pairs of its hit masks into its slots
// of `idx_i` and `idx_j` (one tensor per threshold), in row and then column
// order. Slots at or past a tensor's length are dropped, which truncates a
// list to a fixed capacity.
template <typename scalar_t>
void write_pairs(
  const CandidateSearch<scalar_t> &search,
  const HitMasks &masks,
  int64_t n_candidates,
  int nthreads,
  const std::vector<std::vector<int64_t>> &offsets,
  std::vector<torch::Tensor> &idx_i,
  std::vector<torch::Tensor> &idx_j
) {
  const size_t n_thresholds = offsets.size();
  const int64_t tile_width = masks.tile_width;
  const int64_t n_words = masks.n_words;
  std::vector<int32_t *> out_i(n_thresholds);
  std::vector<int32_t *> out_j(n_thresholds);
  std::vector<int64_t> length(n_thresholds);
  for (size_t k = 0; k < n_thresholds; ++k) {
    out_i[k] = idx_i[k].data_ptr<int32_t>();
    out_j[k] = idx_j[k].data_ptr<int32_t>();
    length[k] = idx_i[k].size(0);
  }

#pragma omp parallel for num_threads(nthreads)                                 \
  schedule(dynamic, kCandidatesPerChunk)
  for (int64_t c = 0; c < n_candidates; ++c) {
    const int64_t *index_a =
      search.index_ptr + search.tile_a_ptr[c] * tile_width;
    const int64_t *index_b =
      search.index_ptr + search.tile_b_ptr[c] * tile_width;

    for (size_t k = 0; k < n_thresholds; ++k) {
      int64_t slot = offsets[k][static_cast<size_t>(c)];
      const int64_t stop =
        std::min(offsets[k][static_cast<size_t>(c + 1)], length[k]);
      const MaskWord *mask = masks.rows(c, static_cast<int64_t>(k));

      for (int64_t row = 0; row < tile_width && slot < stop; ++row) {
        for (int64_t w = 0; w < n_words; ++w) {
          MaskWord word = mask[row * n_words + w];
          while (word != 0 && slot < stop) {
            const int64_t col = w * kBitsPerWord + lowest_bit(word);
            word &= word - 1; // clears the lowest set bit
            // Cannot truncate: `atom_pairs_within_thresholds_cpu` checks that
            // every index, and the padding value, fits `int32_t`.
            out_i[k][slot] = static_cast<int32_t>(index_a[row]);
            out_j[k][slot] = static_cast<int32_t>(index_b[col]);
            ++slot;
          }
        }
      }
    }
  }
}

// Intel's and LLVM's OpenMP runtimes, which `icpx` and `clang++` link,
// keep their threads spinning for a while after a parallel region (the
// "blocktime", 200 ms by default). Torch runs its own thread pool, usually
// GCC's, right after the search, and would share the cores with them. This
// turns the spinning off for the search and restores the setting after it.
// GCC's runtime has no blocktime, so nothing changes there.
class NoSpinningAfterRegions {
  public:
    NoSpinningAfterRegions() {
#ifdef KMP_VERSION_MAJOR
      previous_ = kmp_get_blocktime();
      kmp_set_blocktime(0);
#endif
    }

    ~NoSpinningAfterRegions() {
#ifdef KMP_VERSION_MAJOR
      kmp_set_blocktime(previous_);
#endif
    }

    NoSpinningAfterRegions(const NoSpinningAfterRegions &) = delete;
    NoSpinningAfterRegions &operator=(const NoSpinningAfterRegions &) = delete;

  private:
    [[maybe_unused]] int previous_ = 0;
};

// Number of slots of a padded output holding `n_found` pairs: `capacity`
// if fixed, otherwise `n_found` rounded up to a multiple of `bucket`. The
// same rule as `_capacity_for` in `list.py`.
int64_t capacity_for(
  int64_t n_found,
  std::optional<int64_t> capacity,
  int64_t bucket
) {
  if (capacity.has_value()) { return *capacity; }
  return (n_found + bucket - 1) / bucket * bucket;
}

// Both passes, templated on the positions' scalar type. A function of its
// own because the `#pragma omp` lines of the passes may not appear inside
// the macro call `AT_DISPATCH_FLOATING_TYPES(...)`.
template <typename scalar_t>
std::vector<std::tuple<torch::Tensor, torch::Tensor, int64_t>> search_pairs(
  const torch::Tensor &index,
  const torch::Tensor &valid,
  const torch::Tensor &tile_a,
  const torch::Tensor &tile_b,
  const torch::Tensor &positions,
  const std::vector<double> &thresholds_sq,
  const bool *anchor_ptr,
  std::optional<int64_t> pad_value,
  std::optional<int64_t> capacity,
  int64_t capacity_bucket
) {
  const NoSpinningAfterRegions no_spinning;
  const int nthreads = std::max(at::get_num_threads(), 1);
  const int64_t n_candidates = tile_a.size(0);
  const int64_t tile_width = index.size(1);
  const int64_t *index_ptr = index.data_ptr<int64_t>();
  const bool *valid_ptr = valid.data_ptr<bool>();

  // Every comparison, the exact test and the box skips alike, is made in
  // `scalar_t` against the squared thresholds rounded once to it, like the
  // Python path's: in `double`, a pair right at a threshold that rounds up
  // in `float` would be dropped here and kept there.
  const std::vector<scalar_t> thresholds_sq_scalar(
    thresholds_sq.begin(), thresholds_sq.end()
  );

  const TileGeometry<scalar_t> geometry = pack_tiles<scalar_t>(
    index_ptr,
    valid_ptr,
    positions.data_ptr<scalar_t>(),
    index.size(0),
    tile_width,
    nthreads
  );
  const size_t n_thresholds = thresholds_sq.size();
  HitMasks masks(n_candidates, static_cast<int64_t>(n_thresholds), tile_width);
  const CandidateSearch<scalar_t> search{
    index_ptr,
    valid_ptr,
    tile_a.data_ptr<int64_t>(),
    tile_b.data_ptr<int64_t>(),
    geometry,
    anchor_ptr,
    tile_width,
    masks.n_words,
    thresholds_sq_scalar,
    *std::max_element(thresholds_sq_scalar.begin(), thresholds_sq_scalar.end()),
  };

  const auto offsets = record_hit_masks(search, n_candidates, nthreads, masks);

  std::vector<int64_t> n_found(n_thresholds);
  std::vector<torch::Tensor> idx_i(n_thresholds);
  std::vector<torch::Tensor> idx_j(n_thresholds);
  for (size_t k = 0; k < n_thresholds; ++k) {
    n_found[k] = offsets[k].back();
    const int64_t length =
      pad_value.has_value()
        ? capacity_for(n_found[k], capacity, capacity_bucket)
        : n_found[k];
    idx_i[k] = torch::empty({length}, torch::kInt);
    idx_j[k] = torch::empty({length}, torch::kInt);
    request_huge_pages(idx_i[k].data_ptr(), idx_i[k].nbytes());
    request_huge_pages(idx_j[k].data_ptr(), idx_j[k].nbytes());
  }

  write_pairs(search, masks, n_candidates, nthreads, offsets, idx_i, idx_j);

  std::vector<std::tuple<torch::Tensor, torch::Tensor, int64_t>> results;
  for (size_t k = 0; k < n_thresholds; ++k) {
    const int64_t n_padding = idx_i[k].size(0) - n_found[k];
    if (n_padding > 0) {
      idx_i[k].narrow(0, n_found[k], n_padding).fill_(*pad_value);
      idx_j[k].narrow(0, n_found[k], n_padding).fill_(*pad_value);
    }
    results.emplace_back(idx_i[k], idx_j[k], n_found[k]);
  }
  return results;
}

} // namespace

// Returns, per threshold, `(idx_i, idx_j, n_found)`. Without `pad_value`,
// `idx_i`/`idx_j` hold exactly the `n_found` pairs. With it, they have the
// capacity of `capacity_for`: the pairs first (truncated to that capacity)
// and `pad_value` in the slots after them.
std::vector<std::tuple<torch::Tensor, torch::Tensor, int64_t>>
  atom_pairs_within_thresholds_cpu(
    torch::Tensor index,
    torch::Tensor valid,
    torch::Tensor tile_a,
    torch::Tensor tile_b,
    torch::Tensor positions,
    std::vector<double> thresholds_sq,
    std::optional<torch::Tensor> anchor,
    std::optional<int64_t> pad_value,
    std::optional<int64_t> capacity,
    int64_t capacity_bucket
  ) {
  TORCH_CHECK(index.device().is_cpu(), "index must be a CPU tensor");
  TORCH_CHECK(valid.device().is_cpu(), "valid must be a CPU tensor");
  TORCH_CHECK(tile_a.device().is_cpu(), "tile_a must be a CPU tensor");
  TORCH_CHECK(tile_b.device().is_cpu(), "tile_b must be a CPU tensor");
  TORCH_CHECK(positions.device().is_cpu(), "positions must be a CPU tensor");
  TORCH_CHECK(index.scalar_type() == torch::kLong, "index must be int64");
  TORCH_CHECK(valid.scalar_type() == torch::kBool, "valid must be bool");
  TORCH_CHECK(tile_a.scalar_type() == torch::kLong, "tile_a must be int64");
  TORCH_CHECK(tile_b.scalar_type() == torch::kLong, "tile_b must be int64");
  TORCH_CHECK(
    positions.dim() == 2 && positions.size(1) == 3,
    "positions must have shape (nat, 3)"
  );
  TORCH_CHECK(index.dim() == 2, "index must have shape (ntile, tile_width)");
  TORCH_CHECK(
    index.sizes() == valid.sizes(), "index and valid must have the same shape"
  );
  TORCH_CHECK(capacity_bucket >= 1, "capacity_bucket must be positive");
  TORCH_CHECK(!thresholds_sq.empty(), "at least one threshold is required");

  // The pair indices are written as `int32_t`. Every real index is below
  // the number of positions, and the padding value is written as it is. The
  // pair counts, slots and offsets stay `int64_t`: a list can hold more
  // pairs than an `int32_t` counts, never more atoms.
  constexpr int64_t kMaxIndex = std::numeric_limits<int32_t>::max();
  TORCH_CHECK(
    positions.size(0) <= kMaxIndex,
    "positions has ",
    positions.size(0),
    " atoms, more than the int32 pair "
    "indices hold (",
    kMaxIndex,
    ")"
  );
  TORCH_CHECK(
    !pad_value.has_value() || (*pad_value >= 0 && *pad_value <= kMaxIndex),
    "pad_value ",
    pad_value.value_or(0),
    " does not fit the int32 pair "
    "indices (maximum ",
    kMaxIndex,
    ")"
  );

  // `anchor`, when given, is one boolean per atom: only pairs with at
  // least one anchor atom are returned (`nullptr` keeps every pair).
  torch::Tensor anchor_c;
  const bool *anchor_ptr = nullptr;
  if (anchor.has_value()) {
    TORCH_CHECK(anchor->device().is_cpu(), "anchor must be a CPU tensor");
    TORCH_CHECK(anchor->scalar_type() == torch::kBool, "anchor must be bool");
    TORCH_CHECK(
      anchor->dim() == 1 && anchor->size(0) == positions.size(0),
      "anchor must have shape (nat,)"
    );
    anchor_c = anchor->contiguous();
    anchor_ptr = anchor_c.data_ptr<bool>();
  }

  std::vector<std::tuple<torch::Tensor, torch::Tensor, int64_t>> results;
  AT_DISPATCH_FLOATING_TYPES(
    positions.scalar_type(), "atom_pairs_within_thresholds_cpu", [&] {
      results = search_pairs<scalar_t>(
        index.contiguous(),
        valid.contiguous(),
        tile_a.contiguous(),
        tile_b.contiguous(),
        positions.contiguous(),
        thresholds_sq,
        anchor_ptr,
        pad_value,
        capacity,
        capacity_bucket
      );
    }
  );
  return results;
}

// Digest of this file, set by the ahead-of-time build in `setup.py` (see
// `_native_flags.source_digest`), so that `_native.py` can tell whether the
// module was built from the installed source. Empty in a just-in-time
// build, which PyTorch recompiles whenever the source changes.
#ifndef TAD_MCTC_SOURCE_DIGEST
#define TAD_MCTC_SOURCE_DIGEST
#endif
#define TAD_MCTC_STRINGIFY(x) #x
#define TAD_MCTC_EXPAND_AND_STRINGIFY(x) TAD_MCTC_STRINGIFY(x)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.attr("source_digest") =
    TAD_MCTC_EXPAND_AND_STRINGIFY(TAD_MCTC_SOURCE_DIGEST);
  m.def(
    "atom_pairs_within_thresholds_cpu",
    &atom_pairs_within_thresholds_cpu,
    "Exact atom pairs within thresholds (CPU, OpenMP)",
    py::arg("index"),
    py::arg("valid"),
    py::arg("tile_a"),
    py::arg("tile_b"),
    py::arg("positions"),
    py::arg("thresholds_sq"),
    py::arg("anchor") = py::none(),
    py::arg("pad_value") = py::none(),
    py::arg("capacity") = py::none(),
    py::arg("capacity_bucket") = 1
  );
}
