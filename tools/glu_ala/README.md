# tools/glu_ala

`convert.py` packs the full `glu_ala_a_0001_to_2048` size ladder (28 to
53,250 atoms, 26 structures), plus the two smallest genuinely-larger
structures from `glu_ala_b_512_to_65536` (106,498 and 212,994 atoms), from
https://www.ergoscf.org/xyz/gluala.php into the compressed
`src/tad_mctc/data/structures/glu_ala/data.npz` this package ships. It
holds parsing/packing logic only, same as `tools/mstore/convert.py` -- it
never contains a structure itself.

The rest of `glu_ala_b` (up to 1.7 million atoms) is not packaged: float
positions are close to incompressible, so the remaining, larger structures
alone would run to tens of MB -- see `convert.py`'s own
`EXTRA_LADDER_B_LABELS` for the exact cutoff and its measured sizes, and
`examples/scaling/glu_ala.py`, which downloads and reads the full combined
ladder directly instead.

```sh
curl -LO https://www.ergoscf.org/files/molecules/glu_ala_a_0001_to_2048.tar.gz
tar xzf glu_ala_a_0001_to_2048.tar.gz
curl -LO https://www.ergoscf.org/files/molecules/glu_ala_b_512_to_65536.tar.gz
tar xzf glu_ala_b_512_to_65536.tar.gz
python tools/glu_ala/convert.py glu_ala_a_0001_to_2048 glu_ala_b_512_to_65536
```

The second argument is optional; omitting it packs `glu_ala_a` alone. This
overwrites `src/tad_mctc/data/structures/glu_ala/data.npz`. Rerunning
against the same downloads produces a byte-identical file: record order
follows each ladder's own filename order, ladder `a` first, and there is
no other source of nondeterminism.

Positions are stored as `float32`, not this library's usual `float64`:
`glu_ala` is a size-scaling benchmark, not a reference-energy dataset like
`mstore`'s, and the source coordinates carry no more than `float32` worth
of significant digits anyway. Combined with the LZMA compression
`_write_npz_lzma` uses instead of `np.savez_compressed`'s hardcoded
deflate (deflate barely touches float32 mantissas: 8.48 MB vs LZMA's
7.41 MB for the same 28 structures), this keeps the packaged file to
about 7.4 MB; a `float64` copy would be closer to 13 MB for no real
precision gain (`get_structure("glu_ala", ...)` returns positions in the
library's default dtype, same as for `mstore` records).
