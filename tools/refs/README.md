# tools/refs

Regenerates `test/references/`, one JSON file per molecule holding the
coordination number and its Cartesian gradient for each of the seven
counting functions `tad_mctc.ncoord` implements (`cn_d3`, `cn_d4`,
`cn_eeq`, `cn_eeq_en`, `cn_gfn2`, `cn_eeqbc`, `cn_eeqbc_en`). The first
five are computed by mctc-lib -- the Fortran library `tad_mctc.ncoord` is
a PyTorch port of -- via its `mctc_ncoord` module, which exposes exactly
five counting functions through one generic constructor (`new_ncoord`)
and one generic accessor (`get_cn`, which also returns the analytic
gradient). The two EEQBC variants reuse mctc-lib's `erf`/`erf_en`
counting functions with EEQBC's own steepness, radii-sum normalization
exponent and covalent radii (see `gen_refs_fortran.f90`'s
`eeqbc_cov_radii` parameter, copied from `multicharge`'s
`eeqbc2025` parametrization) -- mctc-lib itself has no EEQBC-specific
counting function, since EEQBC lives in `multicharge`, one level up the
Grimme-group stack.

Building `gen_refs_fortran` fetches mctc-lib v0.5.2 from GitHub and builds
it from source, via [fpm](https://fpm.fortran-lang.org/) (`fpm.toml`) or
[Meson](https://mesonbuild.com/) (`meson.build`, `subprojects/mctc-lib.wrap`)
-- whichever you prefer. Neither hardcodes a compiler: both use whatever
Fortran compiler they default to, or `$FC`/`FPM_FC` if set.

With fpm:

```sh
cd tools/refs
fpm install --profile release --build-dir _build_fpm --prefix _install_fpm
```

With Meson:

```sh
cd tools/refs
meson setup _build_meson --buildtype=release --prefix "$PWD/_install_meson"
meson install -C _build_meson
```

Either way, `gen_refs.py` finds the installed binary itself, at
`_install_fpm/bin/gen_refs_fortran` or `_install_meson/bin/gen_refs_fortran`
respectively (both -- along with the `_build_fpm`/`_build_meson` build
directories -- gitignored: machine-specific, not portable; only the
Fortran source and `subprojects/mctc-lib.wrap` are tracked). Then:

```sh
python tools/refs/gen_refs.py
```

`gen_refs.py` writes one `test/references/<collection>/<record>.json` per
`(collection, record)` entry in its `SAMPLE_LIST`, in place; commit the
result. `test/test_ncoord/samples.py` loads every JSON file it finds there,
so adding a structure to `SAMPLE_LIST` and rerunning this script is all a
new reference needs.
`"other"` is not an mstore collection -- it names a bespoke entry from
`tad_mctc.data.structures.other` (e.g. `("other", "C6H5I-CH3SH")`). Every
entry is resolved through `tad_mctc.data.structures.get_structure` and
written to the same `<collection>/<record>.json` shape.

The steepness/cutoff parameters `gen_refs_fortran.f90` passes to each
`new_ncoord` call were checked against `tad_mctc.ncoord.defaults` so that
each reproduces tad-mctc's own formula exactly (see the comments next to
each `eval(...)` call in that file), not just mctc-lib's own library
defaults, which for `cn_d4`, `cn_eeq` and `cn_eeq_en` differ slightly.

The EEQBC coordination number (Froitzheim, Müller, Hansen, Grimme,
*J. Chem. Phys.* 2025, 162, 214109) is not capped and uses its own
steepness (`kcn=2.0`), radii-sum normalization exponent (`norm_exp=0.75`)
and covalent radii table -- none of which match EEQ's or mctc-lib's
`erf`/`erf_en` defaults.

## Periodic cells

Periodic cells sit in the same `SAMPLE_LIST` as the molecules and are
regenerated in the same run -- no separate list, script or directory.
Periodicity is not a different code path in mctc-lib: `ncoord_type%get_cn`
always calls `get_lattice_points(mol%periodic, mol%lattice, ...)`, and a
non-periodic `mol` (built via `new(mol, num, xyz)` with no lattice) is
simply the case where that call returns a single lattice point at the
origin. `gen_refs_fortran.f90` reads an optional lattice and periodicity
mask straight off stdin after the atom list (see the header comment in
that file for the exact format) and, when present, builds `mol` via
`new(mol, num, xyz, lattice=lattice, periodic=periodic)` instead.

One convention worth remembering: mctc-lib stores lattice vectors as
*columns* (`lattice(:, i)` is the i-th vector -- see `mctc/cutoff.f90`'s
`get_lattice_points`), while tad-mctc stores them as *rows*. The Fortran
tool reads each row of stdin input straight into a column, which is the
transpose, so nothing needs transposing on the Python side.

For periodic entries, `"other"` names a bespoke cell from
`tad_mctc.data.structures.other`: either a synthetic test cell or a real
bulk solid (diamond, rock-salt NaCl) built from standard crystallographic
lattice constants, since mstore has no covalent-network or ionic solid to
mirror. Any other collection names an mstore record.
