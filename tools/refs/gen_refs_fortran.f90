! This file is part of tad-mctc.
!
! SPDX-Identifier: Apache-2.0
! Copyright (C) 2024 Grimme Group
!
! Licensed under the Apache License, Version 2.0 (the "License");
! you may not use this file except in compliance with the License.
! You may obtain a copy of the License at
!
!     http://www.apache.org/licenses/LICENSE-2.0
!
! Unless required by applicable law or agreed to in writing, software
! distributed under the License is distributed on an "AS IS" BASIS,
! WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
! See the License for the specific language governing permissions and
! limitations under the License.

!> Small standalone tool that dumps the coordination number and its Cartesian
!> gradient, for the seven counting functions tad-mctc's ``ncoord``
!> subpackage implements (``cn_d3``, ``cn_d4``, ``cn_eeq``, ``cn_eeq_en``,
!> ``cn_gfn2``, ``cn_eeqbc``, ``cn_eeqbc_en``), computed by the mctc-lib
!> Fortran library itself -- mctc-lib's own ``mctc_ncoord`` module exposes
!> exactly five counting functions (``cn_count%exp``, ``%dftd4``, ``%erf``,
!> ``%erf_en``, ``%dexp``) through one generic constructor and one generic
!> accessor; the two EEQBC variants reuse ``%erf``/``%erf_en`` with EEQBC's
!> own steepness/normalization/radii (see the ``eeqbc_cov_radii`` parameter
!> below), since mctc-lib itself has no EEQBC-specific counting function --
!> EEQBC lives in ``multicharge``, one level up the Grimme-group stack.
!>
!> The steepness/cutoff parameters passed to ``new_ncoord`` below for each
!> variant were checked against tad-mctc's ``ncoord/defaults.py`` so that
!> each call reproduces tad-mctc's own formula exactly, not just mctc-lib's
!> library defaults -- ``cn_d3`` and ``cn_gfn2`` need no overrides, ``cn_d4``
!> needs an explicit ``cutoff``, and ``cn_eeq``/``cn_eeq_en`` need an
!> explicit ``kcn``. ``cn_eeq`` here matches
!> ``test/test_ncoord/test_cn_eeq.py``'s own reference construction (which
!> rebuilds ``coordination_number`` directly with no CN cap), not the public
!> ``tad_mctc.ncoord.eeq.cn_eeq`` function (which additionally caps the CN
!> via ``cut=8``).
!>
!> This program is molecular only (no periodic boundary conditions):
!> ``mctc_io``'s ``new(mol, num, xyz)`` constructor leaves ``mol%periodic``
!> false, so ``ncoord_type%get_cn`` builds a single lattice point at the
!> origin internally -- the correct (and only) input for a finite system.
!>
!> Input is read from standard input, as plain whitespace-separated text:
!>
!>     <number of atoms>
!>     <atomic number> <x> <y> <z>   (repeated once per atom, x/y/z in Bohr)
!>
!> Output is a single line of JSON on standard output, one ``cn_*``/
!> ``dcn_*dr`` pair per counting function:
!>
!>     {"cn_d3": [...], "dcn3dr": [[[...]]], "cn_d4": [...], ...,
!>      "cn_eeqbc": [...], "dcn_eeqbcdr": [[[...]]], "cn_eeqbc_en": [...],
!>      "dcn_eeqbc_endr": [[[...]]]}
!>
!> ``dcn_*dr`` has shape (nat, nat, 3), written out nested (one JSON array
!> per atom pair), which is the shape and order ``numpy.array(...)``/
!> ``torch.tensor(...)`` reconstruct directly with no reshaping needed on
!> the Python side.
program gen_refs_fortran
   use, intrinsic :: iso_fortran_env, only : output_unit, error_unit, input_unit
   use mctc_env, only : wp, error_type
   use mctc_io, only : structure_type, new
   use mctc_ncoord, only : new_ncoord, cn_count, ncoord_type
   implicit none

   type(structure_type) :: mol

   integer :: nat, iat
   integer, allocatable :: num(:)
   real(wp), allocatable :: xyz(:, :)

   real(wp), allocatable :: cn_d3(:), cn_d4(:), cn_eeq(:), cn_gfn2(:), cn_eeq_en(:)
   real(wp), allocatable :: dcn_d3dr(:, :, :), dcn_d4dr(:, :, :), dcn_eeqdr(:, :, :)
   real(wp), allocatable :: dcn_gfn2dr(:, :, :), dcn_eeq_endr(:, :, :)

   real(wp), allocatable :: cn_eeqbc(:), cn_eeqbc_en(:)
   real(wp), allocatable :: dcn_eeqbcdr(:, :, :), dcn_eeqbc_endr(:, :, :)

   !> Element-specific covalent radii for the EEQBC coordination number,
   !> copied verbatim from ``multicharge``'s
   !> ``multicharge_param_eeqbc2025``'s ``eeqbc_cov_radii`` -- already in
   !> atomic units (Bohr), no further conversion applied there either.
   !> tad_mctc.data.radii.EEQBC_COV_RADII is the same table, prefixed with
   !> a dummy Z=0 entry to match tad-mctc's own radii-table convention.
   real(wp), parameter :: eeqbc_cov_radii(103) = 0.5_wp*[&
      &  1.0873678902_wp,  0.0045628280_wp,  2.8385414023_wp,  2.2369359793_wp, & !1-4
      &  2.2631432568_wp,  2.5556464299_wp,  2.6528219471_wp,  2.5471166478_wp, & !5-8
      &  2.0970520036_wp,  1.1527679853_wp,  3.9222564151_wp,  3.6628112720_wp, & !9-12
      &  3.1200757440_wp,  3.2311571633_wp,  3.3714412240_wp,  3.4966508157_wp, & !13-16
      &  3.1641167151_wp,  1.4177781099_wp,  4.2825156987_wp,  4.0979720404_wp, & !17-20
      &  3.4027683492_wp,  3.1330930644_wp,  3.2083129359_wp,  3.1971240284_wp, & !21-24
      &  3.0122827243_wp,  3.0215412255_wp,  2.9286697665_wp,  2.9318983659_wp, & !25-28
      &  2.9333228012_wp,  3.4211414769_wp,  3.5543149265_wp,  3.3547895521_wp, & !29-32
      &  3.8566746758_wp,  4.0522579752_wp,  3.7055903624_wp,  2.1533955559_wp, & !33-36
      &  4.8750192244_wp,  4.2251415193_wp,  3.8193395754_wp,  3.7700784196_wp, & !37-40
      &  3.8026660286_wp,  3.4791250751_wp,  3.3748738252_wp,  3.3400607232_wp, & !41-44
      &  3.3194948126_wp,  3.5185381046_wp,  3.6974620558_wp,  4.2120386946_wp, & !45-48
      &  4.2834967376_wp,  4.0408029917_wp,  4.1029792717_wp,  4.5056357496_wp, & !49-52
      &  4.1912939737_wp,  3.1889722321_wp,  5.3761906399_wp,  4.9848540155_wp, & !53-56
      &  4.1643020686_wp,  4.2242687055_wp,  4.0906998457_wp,  4.0483164017_wp, & !57-60
      &  4.0130748483_wp,  3.6618368303_wp,  3.8161213688_wp,  3.6044393411_wp, & !61-64
      &  3.7159335631_wp,  3.8610077243_wp,  3.8543967507_wp,  3.7804332520_wp, & !65-68
      &  3.6171475823_wp,  3.6614908934_wp,  3.9127576452_wp,  3.7447075110_wp, & !69-72
      &  3.7737132271_wp,  3.3371881773_wp,  3.3105897209_wp,  3.3868061092_wp, & !73-76
      &  3.4036207674_wp,  3.5310959808_wp,  3.5697281366_wp,  4.3942379403_wp, & !77-80
      &  4.6313791191_wp,  4.3952892419_wp,  4.2961541488_wp,  4.6294486870_wp, & !81-84
      &  4.5547581691_wp,  3.6325616087_wp,  5.0182872162_wp,  4.4284455579_wp, & !85-88
      &  3.7478960318_wp,  2.9868700652_wp,  3.6306659286_wp,  3.8514208797_wp, & !89-92
      &  3.4838982283_wp,  3.5107992105_wp,  3.4255592145_wp,  3.5581888894_wp, & !93-96
      &  3.3079867688_wp,  3.4972376288_wp,  3.4590842861_wp,  3.2327976004_wp, & !97-100
      &  3.4619632872_wp,  3.7360296053_wp,  3.5692246969_wp] !101-103

   read(input_unit, *) nat

   allocate(num(nat), xyz(3, nat))
   do iat = 1, nat
      read(input_unit, *) num(iat), xyz(1, iat), xyz(2, iat), xyz(3, iat)
   end do

   call new(mol, num, xyz)

   allocate(cn_d3(nat), cn_d4(nat), cn_eeq(nat), cn_gfn2(nat), cn_eeq_en(nat))
   allocate(dcn_d3dr(3, nat, nat), dcn_d4dr(3, nat, nat), dcn_eeqdr(3, nat, nat))
   allocate(dcn_gfn2dr(3, nat, nat), dcn_eeq_endr(3, nat, nat))

   allocate(cn_eeqbc(nat), cn_eeqbc_en(nat))
   allocate(dcn_eeqbcdr(3, nat, nat), dcn_eeqbc_endr(3, nat, nat))

   ! tad_mctc.ncoord.d3.cn_d3: exp counting, kcn=16, cutoff=25 (mctc-lib defaults)
   call eval(mol, cn_count%exp, cn_d3, dcn_d3dr)

   ! tad_mctc.ncoord.d4.cn_d4: dftd4 counting, cutoff=30 (tad-mctc overrides
   ! mctc-lib's default of 25); kcn/rcov/en/k4/k5/k6 match mctc-lib defaults
   call eval(mol, cn_count%dftd4, cn_d4, dcn_d4dr, cutoff=30.0_wp)

   ! test_ncoord/test_cn_eeq.py's own reference rebuilds coordination_number
   ! directly (not tad_mctc.ncoord.eeq.cn_eeq) with kcn=7.5, cutoff=25 and no
   ! CN cap (cn_max=None by default) -- match that, not cn_eeq's cut=8 cap.
   call eval(mol, cn_count%erf, cn_eeq, dcn_eeqdr, kcn=7.5_wp)

   ! tad_mctc.ncoord.gfn2.cn_gfn2: dexp counting, ka=10/kb=20/r_shift=2/
   ! cutoff=25 (mctc-lib defaults)
   call eval(mol, cn_count%dexp, cn_gfn2, dcn_gfn2dr)

   ! tad_mctc.ncoord.eeq.cn_eeq_en: erf_en counting, kcn=2.60 (tad-mctc
   ! overrides mctc-lib's erf_en default of 2.65)
   call eval(mol, cn_count%erf_en, cn_eeq_en, dcn_eeq_endr, kcn=2.60_wp)

   ! tad_mctc.ncoord.eeqbc.cn_eeqbc: erf counting with EEQBC's own kcn=2.0,
   ! norm_exp=0.75 and covalent radii (none of which match mctc-lib's or
   ! EEQ's defaults); cutoff=25 matches mctc-lib's generic-erf default.
   call eval(mol, cn_count%erf, cn_eeqbc, dcn_eeqbcdr, &
      & kcn=2.0_wp, norm_exp=0.75_wp, rcov=eeqbc_cov_radii(mol%num))

   ! tad_mctc.ncoord.eeqbc.cn_eeqbc_en: erf_en counting, same EEQBC
   ! kcn/norm_exp/radii as cn_eeqbc above.
   call eval(mol, cn_count%erf_en, cn_eeqbc_en, dcn_eeqbc_endr, &
      & kcn=2.0_wp, norm_exp=0.75_wp, rcov=eeqbc_cov_radii(mol%num))

   call write_json(output_unit, cn_d3, dcn_d3dr, cn_d4, dcn_d4dr, cn_eeq, &
      & dcn_eeqdr, cn_gfn2, dcn_gfn2dr, cn_eeq_en, dcn_eeq_endr, &
      & cn_eeqbc, dcn_eeqbcdr, cn_eeqbc_en, dcn_eeqbc_endr)

contains

   !> Build one coordination-number evaluator and evaluate CN + gradient.
   subroutine eval(mol, cn_count_type, cn, dcndr, kcn, cutoff, cut, norm_exp, rcov)
      type(structure_type), intent(in) :: mol
      integer, intent(in) :: cn_count_type
      real(wp), intent(out) :: cn(:)
      real(wp), intent(out) :: dcndr(:, :, :)
      real(wp), intent(in), optional :: kcn
      real(wp), intent(in), optional :: cutoff
      real(wp), intent(in), optional :: cut
      real(wp), intent(in), optional :: norm_exp
      real(wp), intent(in), optional :: rcov(:)

      class(ncoord_type), allocatable :: ncoord
      type(error_type), allocatable :: error
      real(wp), allocatable :: dcndL(:, :, :)

      call new_ncoord(ncoord, mol, cn_count_type, error, kcn=kcn, cutoff=cutoff, &
         & cut=cut, norm_exp=norm_exp, rcov=rcov)
      if (allocated(error)) then
         write(error_unit, "(a)") error%message
         error stop 1
      end if

      ! get_coordination_number only takes the gradient branch (vs. the
      ! energy-only one) when BOTH dcndr and dcndL are present -- dcndL
      ! (strain derivative) is irrelevant for this molecular-only tool and
      ! discarded, but has to be passed to make get_cn compute dcndr at all.
      allocate(dcndL(3, 3, size(cn)))
      call ncoord%get_cn(mol, cn, dcndr, dcndL)
   end subroutine eval

   !> Write all fourteen arrays as one line of JSON, at full double precision.
   subroutine write_json(unit, cn_d3, dcn_d3dr, cn_d4, dcn_d4dr, cn_eeq, &
         & dcn_eeqdr, cn_gfn2, dcn_gfn2dr, cn_eeq_en, dcn_eeq_endr, &
         & cn_eeqbc, dcn_eeqbcdr, cn_eeqbc_en, dcn_eeqbc_endr)
      integer, intent(in) :: unit
      real(wp), intent(in) :: cn_d3(:), dcn_d3dr(:, :, :)
      real(wp), intent(in) :: cn_d4(:), dcn_d4dr(:, :, :)
      real(wp), intent(in) :: cn_eeq(:), dcn_eeqdr(:, :, :)
      real(wp), intent(in) :: cn_gfn2(:), dcn_gfn2dr(:, :, :)
      real(wp), intent(in) :: cn_eeq_en(:), dcn_eeq_endr(:, :, :)
      real(wp), intent(in) :: cn_eeqbc(:), dcn_eeqbcdr(:, :, :)
      real(wp), intent(in) :: cn_eeqbc_en(:), dcn_eeqbc_endr(:, :, :)

      write(unit, "(a)", advance="no") '{"cn_d3": '
      call write_vector(unit, cn_d3)
      write(unit, "(a)", advance="no") ', "dcn3dr": '
      call write_grad(unit, dcn_d3dr)

      write(unit, "(a)", advance="no") ', "cn_d4": '
      call write_vector(unit, cn_d4)
      write(unit, "(a)", advance="no") ', "dcn_d4dr": '
      call write_grad(unit, dcn_d4dr)

      write(unit, "(a)", advance="no") ', "cn_eeq": '
      call write_vector(unit, cn_eeq)
      write(unit, "(a)", advance="no") ', "dcn_eeqdr": '
      call write_grad(unit, dcn_eeqdr)

      write(unit, "(a)", advance="no") ', "cn_gfn2": '
      call write_vector(unit, cn_gfn2)
      write(unit, "(a)", advance="no") ', "dcn_gfn2dr": '
      call write_grad(unit, dcn_gfn2dr)

      write(unit, "(a)", advance="no") ', "cn_eeq_en": '
      call write_vector(unit, cn_eeq_en)
      write(unit, "(a)", advance="no") ', "dcn_eeq_endr": '
      call write_grad(unit, dcn_eeq_endr)

      write(unit, "(a)", advance="no") ', "cn_eeqbc": '
      call write_vector(unit, cn_eeqbc)
      write(unit, "(a)", advance="no") ', "dcn_eeqbcdr": '
      call write_grad(unit, dcn_eeqbcdr)

      write(unit, "(a)", advance="no") ', "cn_eeqbc_en": '
      call write_vector(unit, cn_eeqbc_en)
      write(unit, "(a)", advance="no") ', "dcn_eeqbc_endr": '
      call write_grad(unit, dcn_eeqbc_endr)

      write(unit, "(a)") "}"
   end subroutine write_json

   !> Write one real(wp) vector as a JSON array, at full double precision.
   subroutine write_vector(unit, vec)
      integer, intent(in) :: unit
      real(wp), intent(in) :: vec(:)

      integer :: k
      character(len=32) :: buffer

      write(unit, "(a)", advance="no") "["
      do k = 1, size(vec)
         if (k > 1) write(unit, "(a)", advance="no") ", "
         write(buffer, "(es24.16e3)") vec(k)
         write(unit, "(a)", advance="no") trim(adjustl(buffer))
      end do
      write(unit, "(a)", advance="no") "]"
   end subroutine write_vector

   !> Write a (3, nat, nat) gradient block as nested JSON of shape
   !> (nat, nat, 3): entry [k][m] holds d(cn of atom k)/d(position of atom
   !> m), matching the index convention of tad-mctc's own gradient
   !> functions (e.g. ``cn_d3_gradient``).
   !>
   !> mctc-lib's own ``dcndr(:, A, B)`` (as filled in by
   !> ``mctc_ncoord_type%ncoord_d``) uses the opposite convention --
   !> d(cn of atom B)/d(position of atom A) -- so the two atom indices are
   !> swapped here to match tad-mctc's convention instead of mctc-lib's.
   subroutine write_grad(unit, dcndr)
      integer, intent(in) :: unit
      real(wp), intent(in) :: dcndr(:, :, :)

      integer :: iat, jat, n

      n = size(dcndr, 2)

      write(unit, "(a)", advance="no") "["
      do iat = 1, n
         if (iat > 1) write(unit, "(a)", advance="no") ", "
         write(unit, "(a)", advance="no") "["
         do jat = 1, n
            if (jat > 1) write(unit, "(a)", advance="no") ", "
            call write_vector(unit, dcndr(:, jat, iat))
         end do
         write(unit, "(a)", advance="no") "]"
      end do
      write(unit, "(a)", advance="no") "]"
   end subroutine write_grad

end program gen_refs_fortran
