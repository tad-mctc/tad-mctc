# tools/glu_ala

`convert.py` packs the `glu_ala_a_0001_to_2048` size ladder (28 to 53,250
atoms, 26 structures) from https://www.ergoscf.org/xyz/gluala.php into the
compressed `src/tad_mctc/data/structures/glu_ala/data.npz` this package
ships. It holds parsing/packing logic only, same as
`tools/mstore/convert.py` -- it never contains a structure itself.

The much larger `glu_ala_b_512_to_65536` ladder (up to 1.7 million atoms)
is never packaged.

```sh
curl -LO https://www.ergoscf.org/files/molecules/glu_ala_a_0001_to_2048.tar.gz
tar xzf glu_ala_a_0001_to_2048.tar.gz
python tools/glu_ala/convert.py glu_ala_a_0001_to_2048
```

This overwrites `src/tad_mctc/data/structures/glu_ala/data.npz`. Rerunning
against the same download produces a byte-identical file: record order
follows the ladder's own filename order, and there is no other source of
nondeterminism.

Positions are stored as `float32`, not this library's usual `float64`:
`glu_ala` is a size-scaling benchmark, not a reference-energy dataset like
`mstore`'s, and the source coordinates carry no more than `float32` worth
of significant digits anyway. This alone keeps the packaged file under
6 MB; a `float64` copy would be closer to 11 MB for no real precision gain
(`get_structure("glu_ala", ...)` returns positions in the library's
default dtype, same as for `mstore` records).
