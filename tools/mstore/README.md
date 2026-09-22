# tools/mstore

`convert.py` turns an [mstore](https://github.com/grimme-lab/mstore) checkout's
Fortran dataset files into the `dict[str, dict[str, Tensor]]` modules under
`src/tad_mctc/data/structures/mstore/`. It holds parsing and code-generation
logic only -- it never contains a structure itself.

```sh
git clone https://github.com/grimme-lab/mstore.git /tmp/mstore
git -C /tmp/mstore checkout <commit>
python tools/mstore/convert.py /tmp/mstore
```

This overwrites all 12 dataset modules
(`amino20x4.py`, `amylose.py`, `but14diol.py`, `f_block.py`, `heavy28.py`,
`ice10.py`, `il16.py`, `mb16_43.py`, `polyalanine.py`, `rc21.py`, `upu23.py`,
`x23.py`), then formats them with `black --line-length 80` and
`isort --profile black --line-length 80` (both already project dependencies)
so the generated files match the rest of the repository's style. Rerunning
against the same checkout produces no diff: record order follows mstore's
own `new_record(...)` registration order in each dataset file, and there is
no other source of nondeterminism.

Each generated module's docstring names its source file and the exact
mstore commit passed on the command line. `datasets` in
`src/tad_mctc/data/structures/mstore/__init__.py` is hand-maintained, so
after adding or removing a whole dataset, update its imports and dict by hand;
the converter only regenerates the per-dataset data files.

The parser fails loudly -- raises, never skips -- on any record it cannot
parse fully: a size mismatch between `nat` and the parsed coordinate or
symbol array, an unrecognized element symbol, or a `call new(...)` shape it
does not handle (currently: `sym`+`xyz`, `num`+`xyz`, plus optional
`lattice`, `charge`, `uhf`, in any combination -- this covers every record
in mstore's `a9070de` `main`).

Mirrored: mstore's `lattice(3, 3)` stores its columns as the lattice
vectors; tad-mctc stores lattice vectors as rows. Because Fortran's
`reshape` into a `(3, 3)` array fills column-by-column, the flat literal's
consecutive triples already *are* the row vectors tad-mctc wants -- no
transpose arithmetic is needed, only reading the flat list in groups of
three. Verified against mstore's `anthracene` record, already mirrored and
independently checked before this tool existed.
