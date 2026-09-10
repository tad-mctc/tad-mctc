project('gen_refs_fortran', 'fortran')

mctc_dep = dependency(
  'mctc-lib',
  fallback: ['mctc-lib', 'mctc_dep'],
  default_options: [
    'default_library=static',
  ],
)

executable(
  'gen_refs_fortran',
  'gen_refs_fortran.f90',
  dependencies: mctc_dep,
  install: true,
)
