
! #############################################################
! #                                                           #
! #                     LIDORT_3p8p3                          #
! #                                                           #
! #    (LInearized Discrete Ordinate Radiative Transfer)      #
! #     --         -        -        -         -              #
! #                                                           #
! #############################################################

! #############################################################
! #                                                           #
! #  Authors :     Robert  J. D. Spurr (1)                    #
! #                Matthew J. Christi                         #
! #                                                           #
! #  Address (1) : RT Solutions, Inc.                         #
! #                9 Channing Street                          #
! #                Cambridge, MA 02138, USA                   #
! #                                                           #
! #  Tel:          (617) 492 1183                             #
! #  Email :       rtsolutions@verizon.net                    #
! #                                                           #
! #  This Version :   LIDORT_3p8p3                            #
! #  Release Date :   31 March 2021                           #
! #                                                           #
! #  Previous LIDORT Versions under Standard GPL 3.0:         #
! #  ------------------------------------------------         #
! #                                                           #
! #      3.7   F90, released        June  2014                #
! #      3.8   F90, released        March 2017                #
! #      3.8.1 F90, released        June  2019                #
! #      3.8.2 F90, limited release May   2020                #
! #                                                           #
! #  Features Summary of Recent LIDORT Versions               #
! #  ------------------------------------------               #
! #                                                           #
! #      NEW: THERMAL SUPPLEMENT INCLUDED    (3.2)            #
! #      NEW: OUTGOING SPHERICITY CORRECTION (3.2)            #
! #      NEW: TOTAL COLUMN JACOBIANS         (3.3)            #
! #      VLIDORT COMPATIBILITY               (3.4)            #
! #      THREADED/OPTIMIZED F90 code         (3.5)            #
! #      EXTERNAL SS / NEW I/O STRUCTURES    (3.6)            #
! #                                                           #
! #      Surface-leaving, BRDF Albedo-scaling     (3.7)       # 
! #      Taylor series, BBF Jacobians, ThreadSafe (3.7)       #
! #      New Water-Leaving Treatment              (3.8)       #
! #      BRDF-Telescoping, enabled                (3.8)       #
! #      Several Performance Enhancements         (3.8)       #
! #      Water-leaving coupled code               (3.8.1)     #
! #      Planetary problem, media properties      (3.8.1)     #
! #      Doublet geometry post-processing         (3.8.2)     #
! #      Reduction zeroing, dynamic memory        (3.8.2)     #
! #                                                           #
! #  Features Summary of This VLIDORT Version                 #
! #  ----------------------------------------                 #
! #                                                           #
! #  3.8.3, released 31 March 2021.                           #
! #    ==> Sphericity Corrections using MS source terms       #
! #    ==> BRDF upgrades, including new snow reflectance      #
! #    ==> SLEAVE Upgrades, extended water-leaving treatment  #
! #                                                           #
! #############################################################

! ###################################################################
! #                                                                 #
! # This is Version 3.8.3 of the LIDORT software library.           #
! # This library comes with the Standard GNU General Public License,#
! # Version 3.0, 29 June 2007. Please read this license carefully.  #
! #                                                                 #
! #      LIDORT Copyright (c) 1999-2023.                            #
! #          Robert Spurr, RT Solutions, Inc.                       #
! #          9 Channing Street, Cambridge, MA 02138, USA.           #
! #                                                                 #
! #                                                                 #
! # This file is part of LIDORT_3p8p3 ( Version 3.8.3. )            #
! #                                                                 #
! # LIDORT_3p8p3 is free software: you can redistribute it          #
! # and/or modify it under the terms of the Standard GNU GPL        #
! # (General Public License) as published by the Free Software      #
! # Foundation, either version 3.0 of this License, or any          #
! # later version.                                                  #
! #                                                                 #
! # LIDORT_3p8p3 is distributed in the hope that it will be         #
! # useful, but WITHOUT ANY WARRANTY; without even the implied      #
! # warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR         #
! # PURPOSE. See the Standard GNU General Public License (GPL)      #
! # for more details.                                               #
! #                                                                 #
! # You should have received a copy of the Standard GNU General     #
! # Public License (GPL) Version 3.0, along with the LIDORT_3p8p3   #
! # code package. If not, see <http://www.gnu.org/licenses/>.       #
! #                                                                 #
! ###################################################################

      program Solar_Tester

!  Upgrade for Version 3.8.1, June 2019
!  -------------------------------------

!  Upgrade for Version 3.8.3, March 2021
!  -------------------------------------

!  Module files for LIDORT. Strict usage, Version 3.8 upwards

      USE BRDF_SUP_AUX_m, Only : BRDF_READ_ERROR
      USE BRDF_SUP_MOD_m

      USE LIDORT_PARS_m
      USE LIDORT_IO_DEFS_m

      USE LIDORT_AUX_m,    Only : LIDORT_READ_ERROR, LIDORT_WRITE_STATUS
      USE LIDORT_INPUTS_m, Only : LIDORT_INPUT_MASTER
      USE LIDORT_MASTERS_m

      USE LIDORT_BRDF_SUP_ACCESSORIES_m

!  Implicit none

      IMPLICIT NONE

!  LIDORT file inputs status structure

      TYPE(LIDORT_Input_Exception_Handling) :: LIDORT_InputStatus

!  LIDORT input structures

      TYPE(LIDORT_Fixed_Inputs)             :: LIDORT_FixIn
      TYPE(LIDORT_Modified_Inputs)          :: LIDORT_ModIn

!  LIDORT supplements i/o structure

      TYPE(LIDORT_Sup_InOut)                :: LIDORT_Sup

!  LIDORT output structure

      TYPE(LIDORT_Outputs)                  :: LIDORT_Out

!  BRDF supplement file inputs status structure

      TYPE(BRDF_Input_Exception_Handling)   :: BRDF_Sup_InputStatus

!  LIDORT BRDF supplement input structure

      TYPE(BRDF_Sup_Inputs)                 :: BRDF_Sup_In

!  LIDORT BRDF supplement output structures

      TYPE(BRDF_Sup_Outputs)                :: BRDF_Sup_Out
      TYPE(BRDF_Output_Exception_Handling)  :: BRDF_Sup_OutputStatus

!  BRDF supplement / LIDORT BRDF-related inputs consistency check status

      TYPE(LIDORT_Exception_Handling)       :: LIDORT_BRDFCheck_Status

!  BRDF supplement variables
!  =========================

      LOGICAL ::          BS_DO_BRDF_SURFACE
      LOGICAL ::          BS_DO_USER_STREAMS
      LOGICAL ::          BS_DO_SURFACE_EMISSION

      INTEGER ::          BS_NSTREAMS
      INTEGER ::          BS_NMOMENTS_INPUT

      INTEGER ::          BS_NBEAMS
      REAL(fpk) ::        BS_BEAM_SZAS ( MAXBEAMS )

      INTEGER ::          BS_N_USER_RELAZMS
      REAL(fpk) ::        BS_USER_RELAZMS ( MAX_USER_RELAZMS )

      INTEGER ::          BS_N_USER_STREAMS
      REAL(fpk) ::        BS_USER_ANGLES_INPUT ( MAX_USER_STREAMS )

      LOGICAL ::          DO_DEBUG_RESTORATION

!  Control shorthand
!  =================

!  Proxies for LIDORT standard input preparation
      
      INTEGER   :: NBEAMS
      INTEGER   :: N_GEOMETRIES, N_USER_LEVELS
      INTEGER   :: NLAYERS, NFINELAYERS, NMOMENTS_INPUT
      REAL(fpk) :: USER_LEVELS (MAX_USER_LEVELS)

      LOGICAL ::   DO_FOCORR, DO_FOCORR_NADIR, DO_FOCORR_OUTGOING
      LOGICAL ::   DO_DELTAM_SCALING, DO_SOLUTION_SAVING, DO_BVP_TELESCOPING

!  Local Optical input Variables
!  =============================

!  Number of tasks

      INTEGER, PARAMETER :: MAXTASKS = 1

!  multilayer Height inputs

      REAL(fpk) :: HEIGHT_GRID( 0:MAXLAYERS )

!  multilayer optical property (bulk) inputs

      REAL(fpk) :: OMEGA_TOTAL_INPUT  ( MAXLAYERS, MAXTASKS )
      REAL(fpk) :: DELTAU_VERT_INPUT  ( MAXLAYERS, MAXTASKS )

!  Phase function Legendre-polynomial expansion coefficients
!   Include all that you require for exact single scatter calculations
!      Phasfunc proxies are new for Version 3.8. Not needed here

      REAL(fpk) :: PHASMOMS_TOTAL_INPUT ( 0:MAXMOMENTS_INPUT, MAXLAYERS, MAXTASKS )
!      REAL(fpk) :: PHASFUNC_UP ( MAXLAYERS, MAX_GEOMETRIES ) 
!      REAL(fpk) :: PHASFUNC_DN ( MAXLAYERS, MAX_GEOMETRIES )
 
!  Lambertian Surface control

      REAL(fpk) :: LAMBERTIAN_ALBEDO (MAXTASKS)

!  Local Output Variables (New for  Version 3.7)
!  ======================

      REAL(fpk), dimension ( MAX_USER_LEVELS, MAX_GEOMETRIES, MAX_DIRECTIONS, MAXTASKS ) :: INTENSITY
      REAL(fpk), dimension ( MAX_USER_LEVELS, MAXBEAMS,       MAX_DIRECTIONS, MAXTASKS ) :: MEAN_INTENSITY
      REAL(fpk), dimension ( MAX_USER_LEVELS, MAXBEAMS,       MAX_DIRECTIONS, MAXTASKS ) :: FLUX_INTEGRAL

!  Other local variables
!  ---------------------

!  Task

      INTEGER          :: TASK

!  Error handling

      LOGICAL          :: OPENFILEFLAG

!  Help variables

      integer          :: n,n6,l,ndum,ldum, v, t, ntasks, ndirs, nmi
      REAL(fpk)        :: kd, gaer, waer, taer, parcel, raywt, aerwt
      REAL(fpk)        :: aersca, aerext, molsca, totsca, totext
      REAL(fpk)        :: molomg(maxlayers),molext(maxlayers)
      REAL(fpk)        :: aermoms(0:maxmoments_input)
      REAL(fpk)        :: raymoms(0:maxmoments_input)
      REAL(fpk)        :: layer_scale


!  2/28/21. Version 3.8.3. Geometry choices, help variables

      INTEGER            :: GEOMETRY_CHOICE
      CHARACTER(Len=150) :: G_inputfile, G_outputfile, B_inputfile, B_outputfile_1, B_outputfile_2
      CHARACTER(Len=7)   :: CGEOM

!  2/28/21. Version 3.8.3. introduce do_debug_input control

      LOGICAL          :: do_debug_input

!mick test
!  Timing test

      INTEGER :: i
      REAL    :: e1,e2

!  Initialize error file output flag

      OPENFILEFLAG = .false.

!  Initialize by-hand input flag
!  2/28/21. Version 3.8.3. Local flag for debug input 

      DO_Debug_input = .false.

!  CHOOSE TOP_LEVEL GEOMETRY
!  =========================

      CGEOM = 'lattice'
      Open(1,file='TOPLEVEL_GEOMETRY_CHOICE',status='old')
      read(1,*)GEOMETRY_CHOICE
      if ( GEOMETRY_CHOICE .le. 0 ) Stop ' GEOMETRY choice must be 1 (lattice), 2 (doublet) or 3 (obsgeom)'
      if ( GEOMETRY_CHOICE .gt. 3 ) Stop ' GEOMETRY choice must be 1 (lattice), 2 (doublet) or 3 (obsgeom)'
      if ( GEOMETRY_CHOICE .eq. 2 ) CGEOM = 'doublet'
      if ( GEOMETRY_CHOICE .eq. 3 ) CGEOM = 'obsgeom'
      close(1)

!  I/O Filenames

      ! G_inputfile  = 'lidort_test/config/'//CGEOM//'/3p8p3_LIDORT_ReadInput.cfg_'//CGEOM
      G_inputfile = '3p8p3_005_test_plot.cfg'
      B_inputfile = trim(G_inputfile)//'.brdf'

      print *, B_inputfile

      G_outputfile = '3p8p3_005_test_plot.all'
      B_outputfile_1 = trim(G_outputfile)//'.brdf1'
      B_outputfile_2 = trim(G_outputfile)//'.brdf2'

!  BRDF SUPPLEMENT CALCULATION
!  ===========================

!  Get the BRDF inputs

      CALL BRDF_INPUTMASTER ( Trim(B_inputfile), &
        BRDF_Sup_In,         & ! Outputs
        BRDF_Sup_InputStatus ) ! Outputs

      IF ( BRDF_Sup_InputStatus%BS_STATUS_INPUTREAD .ne. LIDORT_SUCCESS ) &
        CALL BRDF_READ_ERROR ( '3p8p3_BRDF_ReadInput.log', BRDF_Sup_InputStatus )


!  A normal calculation will require

      BS_NMOMENTS_INPUT = 2 * BRDF_Sup_In%BS_NSTREAMS - 1

      DO_DEBUG_RESTORATION = .FALSE.

!  Linearized BRDF call
!    The output will now be used for the Main LIDORT calculation

      CALL BRDF_MAINMASTER ( &
        DO_DEBUG_RESTORATION, & ! Inputs
        BS_NMOMENTS_INPUT,    & ! Inputs
        BRDF_Sup_In,          & ! Inputs
        BRDF_Sup_Out,         & ! Outputs
        BRDF_Sup_OutputStatus ) ! Output exception handling

!  LIDORT control read input, abort if failed

      CALL LIDORT_input_master ( Trim(G_inputfile), &
          LIDORT_FixIn,        & ! Outputs
          LIDORT_ModIn,        & ! Outputs
          LIDORT_InputStatus )   ! Outputs

      IF ( LIDORT_InputStatus%TS_STATUS_INPUTREAD .ne. LIDORT_SUCCESS ) &
          CALL LIDORT_READ_ERROR ( '3p8p3_LIDORT_ReadInput.log', LIDORT_InputStatus )

!  2/28/21. Version 3.8.3. Tolerance variable and specialist flag
!    -- MUST BE SET BY HAND. These are not configuration file reads

      LIDORT_FixIn%Bool%TS_DO_MSSTS             = .FALSE.
      LIDORT_FixIn%Cont%TS_ASYMTX_TOLERANCE     = 1.0d-20

!  Shorthand

      nbeams        = LIDORT_ModIn%MSunrays%TS_nbeams
      n_user_levels = LIDORT_FixIn%UserVal%TS_n_user_levels
      user_levels   = LIDORT_ModIn%MUserVal%TS_user_levels
      
      LIDORT_FixIn%Bool%TS_DO_BRDF_SURFACE = BRDF_Sup_In%BS_DO_BRDF_SURFACE

      BRDF_Sup_In%BS_WHICH_BRDF = 8

      CALL LIDORT_BRDF_INPUT_CHECK ( &
        BRDF_Sup_In,             & ! Inputs
        LIDORT_FixIn,            & ! Inputs
        LIDORT_ModIn,            & ! Inputs
        LIDORT_BRDFCheck_Status )  ! Outputs
      
        !  Exception handling

      IF ( LIDORT_BRDFCheck_Status%TS_STATUS_INPUTCHECK .ne. LIDORT_SUCCESS ) THEN
        write(*,'(/1X,A)') 'LIDORT/BRDF baseline check:'
        CALL LIDORT_BRDF_INPUT_CHECK_ERROR ( &
          '3p8p3_LIDORT_BRDFcheck.log', LIDORT_BRDFCheck_Status )
      ENDIF

      CALL SET_LIDORT_BRDF_INPUTS ( &
        BRDF_Sup_Out,    & !Inputs
        LIDORT_FixIn, LIDORT_ModIn,       & !Inputs
        LIDORT_Sup)          !Outputs

!  Get the pre-prepared atmosphere. NO AEROSOLS
!    This is a Rayleigh-scattering, Ozone-absorbing atmosphere at 315 nm
!   NLAYERS should be 23
!    Line 1 = Rayleigh phase function expansion coefficient 0, 23 layers
!    Line 2 = Rayleigh phase function expansion coefficient 1, 23 layers
!    Line 3 = Rayleigh phase function expansion coefficient 2, 23 layers
!    Lines 4-26 have the following columns:
!        1. Layer #
!        2. Bottom height of layer in [km]
!        3. Extinction optical thickness of layer
!        4. Single-scattering albedo
!        5-8. These are actually Linearized optical properties. NOT NEEDED.

      nlayers = LIDORT_FixIn%Cont%TS_nlayers
      ! open(45,file='input_atmos_gsit.dat',status='old' )
      ! read(45,'(i5,1p25e18.9)')ldum, (raymoms(0,n),n=1,nlayers)
      ! read(45,'(i5,1p25e18.9)')ldum, (raymoms(1,n),n=1,nlayers)
      ! read(45,'(i5,1p25e18.9)')ldum, (raymoms(2,n),n=1,nlayers)
      ! height_grid(0) = 60.0d0
      ! do n = 1, nlayers
      !   read(45,'(i4,f12.5,1p6e16.7)')ndum,height_grid(n),molext(n),molomg(n),kd,kd,kd,kd
      ! enddo
      ! close(45)

!  Add Aerosols all layers, spread evenly


      nmoments_input = 80
      gaer = 0.8d0 ; waer = 0.95d0 ; taer = 0.5d0 ; aermoms(0) = 1.0d0
      do l = 1, nmoments_input
        aermoms(l) = dble(2*L+1) * gaer ** dble(L)
        raymoms(l) = 0.0
      enddo

      raymoms(0) = 1.0
      raymoms(1) = 0.0
      raymoms(2) = 4.920062061E-01

!  Define additional proxies

      ndirs = max_directions
      nmi   = nmoments_input

!  Fill up optical properties, first task

      ! do n = 1, n6
      !    deltau_vert_input(n,1) = molext(n)
      !    omega_total_input(n,1) = molomg(n)
      !    phasmoms_total_input(0,n,1) = raymoms(0,n)
      !    phasmoms_total_input(1,n,1) = raymoms(1,n)
      !    phasmoms_total_input(2,n,1) = raymoms(2,n)
      !    phasmoms_total_input(3:nmi,n,1) = zero
      ! enddo
      ! parcel = taer / ( height_grid(n6) - height_grid(nlayers) )
      height_grid(0) = dble(nlayers)
      do n = 1, nlayers
         deltau_vert_input(n,1) = 0.01
         layer_scale = (dble(n) - 1) / dble(nlayers - 1)

         omega_total_input(n,1) = 0.9 + layer_scale * (0.99 - 0.9) 
      !    omega_total_input(n,1) = 0.01
         height_grid(n) = height_grid(0) - dble(n) * 1.0
         do L = 0, nmoments_input
            ! layer_scale = 0.01
            phasmoms_total_input(L,n,1) =  aermoms(L) * layer_scale + raymoms(L) * (1.0 - layer_scale)
            ! phasmoms_total_input(L,n,1) =  raymoms(L)
         enddo
      enddo

!  Surface

      lambertian_albedo(1) = 0.3d0

!  Fill up optical properties, other tasks

      ntasks = maxtasks
!       do t = 2, ntasks
!          do n = 1, nlayers
!             deltau_vert_input(n,t) = deltau_vert_input(n,1)
!             omega_total_input(n,t) = omega_total_input(n,1)
!             do l = 0, nmoments_input
!               phasmoms_total_input(l,n,t) = phasmoms_total_input(l,n,1)
!             enddo
!          enddo
!          lambertian_albedo(t) = lambertian_albedo(1)
!      enddo
 
!  Copy local control integers, height grid.
!      SAME for all tasks

      LIDORT_FixIn%Cont%TS_nlayers                   = nlayers
      LIDORT_ModIn%MCont%TS_nmoments_input           = nmoments_input
      LIDORT_FixIn%Chapman%TS_height_grid(0:nlayers) = height_grid(0:nlayers)

!  Start task loop
!  ===============

      do task = 1, ntasks
!        do task = 1, 1
!        do task = 2, 3

!  Copy to optical property type-structure inputs
!    This must now be done for each task

         LIDORT_FixIn%Optical%TS_deltau_vert_input(1:nlayers)          = deltau_vert_input(1:nlayers,task)
         LIDORT_ModIn%MOptical%TS_omega_total_input(1:nlayers)         = omega_total_input(1:nlayers,task)
         LIDORT_FixIn%Optical%TS_phasmoms_total_input(0:nmi,1:nlayers) = phasmoms_total_input(0:nmi,1:nlayers,task)
         LIDORT_FixIn%Optical%TS_lambertian_albedo                     = lambertian_albedo(task)

!  Task 1: No FO correction, no delta-M scaling
!  Task 2: No FO correction, With delta-M scaling
!  Task 3: Ingoing-only FO correction, With delta-M scaling  (SUN in curved atmosphere)
!  Task 4: In/Outgoing  FO correction, With delta-M scaling  (SUN+LOS in curved atmosphere)
!  Task 5: As task 4, but with Solution Saving
!  Task 6: As task 4, but with Solution Saving and BVP Telescoping

!          if ( task .eq. 1 ) then
!             DO_FOCORR          = .false.
!             DO_FOCORR_NADIR    = .false.
!             DO_FOCORR_OUTGOING = .false.
!             DO_DELTAM_SCALING  = .false.
!             DO_SOLUTION_SAVING = .false.
!             DO_BVP_TELESCOPING = .false.
!             NFINELAYERS        = 0
!          else if ( task .eq. 2 ) then
!             DO_FOCORR          = .false.
!             DO_FOCORR_NADIR    = .false.
!             DO_FOCORR_OUTGOING = .false.
!             DO_DELTAM_SCALING  = .true.
!             DO_SOLUTION_SAVING = .false.
!             DO_BVP_TELESCOPING = .false.
!             NFINELAYERS        = 0
!          else if ( task .eq. 3 ) then
!             DO_FOCORR          = .true.
!             DO_FOCORR_NADIR    = .true.
!             DO_FOCORR_OUTGOING = .false.
!             DO_DELTAM_SCALING  = .true.
!             DO_SOLUTION_SAVING = .false.
!             DO_BVP_TELESCOPING = .false.
!             NFINELAYERS        = 0
!          else if ( task .eq. 4 ) then
!             DO_FOCORR          = .true.
!             DO_FOCORR_NADIR    = .false.
!             DO_FOCORR_OUTGOING = .true.
!             DO_DELTAM_SCALING  = .true.
!             DO_SOLUTION_SAVING = .false.
!             DO_BVP_TELESCOPING = .false.
! !          Non zero here
!             NFINELAYERS        = 4
!          else if ( task .eq. 5 ) then
!             DO_FOCORR          = .true.
!             DO_FOCORR_NADIR    = .false.
!             DO_FOCORR_OUTGOING = .true.
!             DO_DELTAM_SCALING  = .true.
!             DO_SOLUTION_SAVING = .true.
!             DO_BVP_TELESCOPING = .false.
! !           Non zero here
!             NFINELAYERS        = 4
!          else if ( task .eq. 6 ) then
            DO_FOCORR          = .true.
            DO_FOCORR_NADIR    = .false.
            DO_FOCORR_OUTGOING = .true.
            DO_DELTAM_SCALING  = .true.
            DO_SOLUTION_SAVING = .true.
            DO_BVP_TELESCOPING = .true.
!           Non zero here
            NFINELAYERS        = 4
      !    endif

!  For special MV-only check

         if ( LIDORT_ModIn%MBool%TS_DO_MVOUT_ONLY ) then
            DO_FOCORR             = .false.
            DO_FOCORR_NADIR       = .false.
            DO_FOCORR_OUTGOING    = .false.
         endif

!  Progress
!          write(*,*)'Doing task # ', task

!  Copy local variables to LIDORT (override config file)

         LIDORT_ModIn%MBool%TS_DO_DELTAM_SCALING  = DO_DELTAM_SCALING
         LIDORT_ModIn%MBool%TS_DO_FOCORR          = DO_FOCORR
         LIDORT_ModIn%MBool%TS_DO_FOCORR_NADIR    = DO_FOCORR_NADIR
         LIDORT_ModIn%MBool%TS_DO_FOCORR_OUTGOING = DO_FOCORR_OUTGOING
         LIDORT_ModIn%MBool%TS_DO_SOLUTION_SAVING = DO_SOLUTION_SAVING
         LIDORT_ModIn%MBool%TS_DO_BVP_TELESCOPING = DO_BVP_TELESCOPING
         LIDORT_FixIn%Cont%TS_NFINELAYERS         = NFINELAYERS

!  LIDORT call
!  2/28/21. Version 3.8.3. Need to set the debug_input flag

         CALL LIDORT_master ( do_debug_input, &
             LIDORT_FixIn, LIDORT_ModIn, LIDORT_Sup, LIDORT_Out )

!  Exception handling, write-up (optional)
!    Will generate file only if errors or warnings are encountered

         CALL LIDORT_WRITE_STATUS ( &
            '3p8p3_LIDORT_Execution.log', LIDORT_ERRUNIT, OPENFILEFLAG, LIDORT_Out%Status )

!  Progress

         write(*,56)'Finished task # ', task, '; number of Fouriers = ',LIDORT_Out%Main%TS_FOURIER_SAVED(1:nbeams)
56       format(a,i3,a,4i4)

!  Write LIDORT intensity to Local output (all tasks)

         n_geometries  = LIDORT_Out%Main%TS_n_geometries
         Intensity(1:n_user_levels,1:n_geometries,1:ndirs,task) = &
             LIDORT_Out%Main%TS_intensity(1:n_user_levels,1:n_geometries,1:ndirs)

!  Write LIDORT Mean-intensity and fluxes to Local output (first two tasks only)

      !    if ( task.lt.3 ) then
      !       Mean_Intensity(1:n_user_levels,1:nbeams,upidx,task)    = &
      !          LIDORT_Out%Main%TS_MeanI_Diffuse(1:n_user_levels,1:nbeams,upidx)
      !       Mean_Intensity(1:n_user_levels,1:nbeams,dnidx,task)    = &
      !          LIDORT_Out%Main%TS_MeanI_Diffuse(1:n_user_levels,1:nbeams,dnidx) &
      !        + LIDORT_Out%Main%TS_DnMeanI_Direct(1:n_user_levels,1:nbeams)
      !       Flux_Integral(1:n_user_levels,1:nbeams,upidx,task)     = &
      !          LIDORT_Out%Main%TS_Flux_Diffuse(1:n_user_levels,1:nbeams,upidx)
      !       Flux_Integral(1:n_user_levels,1:nbeams,dnidx,task)     = &
      !          LIDORT_Out%Main%TS_Flux_Diffuse(1:n_user_levels,1:nbeams,dnidx) &
      !        + LIDORT_Out%Main%TS_DnFlux_Direct(1:n_user_levels,1:nbeams)
      !    endif

!  End task loop

      ENDDO

!  Write final file for all tasks
!  ==============================

      OPEN(36,file = Trim(G_outputfile), status = 'unknown')

!  Intensity output

      if ( .NOT. LIDORT_ModIn%MBool%TS_DO_MVOUT_ONLY ) then
         write(36,'(/T32,a/T32,a/)')'INTENSITIES, 1 task', &
                                    '======================'
      !    write(36,'(a,T32,6(a16,2x)/)')'Geometry    Level/Output', &
      !           'No SSCorr No DM','No SSCorr + DM ','SSNadir  + DM  ',&
      !           'SSOutgoing + DM','#4 + Solsaving ','#5 + BVPTelscpe'
         write(36,'(a,T32,(a16,2x)/)')'Geometry    Level/Output', &
                '#1 + BVPTelscpe'
         do v = 1, n_geometries
            do n = 1, n_user_levels
               write(36,366)v,'Upwelling @',user_levels(n), (Intensity(n,v,upidx,t),t=1,ntasks)
            enddo
            do n = 1, n_user_levels
               write(36,366)v,'Dnwelling @',user_levels(n), (Intensity(n,v,dnidx,t),t=1,ntasks)
            enddo
            write(36,*)' '
         enddo
      endif

!  Mean value output

      ! if ( LIDORT_ModIn%MBool%TS_DO_ADDITIONAL_MVOUT .OR. LIDORT_ModIn%MBool%TS_DO_MVOUT_ONLY ) then
      !    write(36,'(/T32,a/T32,a/)')'ACTINIC + REGULAR FLUXES, tasks 1-2', &
      !                               '==================================='
      !    write(36,'(a,T31,4(a16,2x)/)')' Sun SZA    Level/Output', &
      !           ' Actinic No DM ',' Actinic + DM  ',&
      !           ' Regular No DM ',' Regular + DM  '
      !    do v = 1, nbeams
      !       do n = 1, n_user_levels
      !          write(36,367)v,'Upwelling @',user_levels(n), &
      !                  (Mean_Intensity(n,v,upidx,t),t=1,2), (Flux_Integral(n,v,upidx,t),t=1,2)
      !       enddo
      !       do n = 1, n_user_levels
      !          write(36,367)v,'Dnwelling @',user_levels(n),              &
      !                  (Mean_Intensity(n,v,dnidx,t),t=1,2), (Flux_Integral(n,v,dnidx,t),t=1,2)
      !       enddo
      !       write(36,*)' '
      !    enddo
      ! endif

      close(36)
366   format(i5,T11,a,f6.2,2x,6(1x,1pe16.7,1x))
367   format(i5,T11,a,f6.2,2x,4(1x,1pe16.7,1x))

!  Close error file if it has been opened

      write(*,*)
      IF ( OPENFILEFLAG ) then
        close( LIDORT_ERRUNIT )
        write(*,*)'Main program executed with internal defaults,'// &
                  ' warnings in "3p8p3_LIDORT_Execution.log"'
      ELSE
        write(*,*)'Main program finished successfully'
      ENDIF

!  Finish

      stop
      end program Solar_Tester
