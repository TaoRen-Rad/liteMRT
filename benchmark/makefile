
#
# This file serves as the head Makefile for LIDORT and its "tester" programs
#

# UPGRADE TO LIDORT VERSION 3.7, May   2014
# UPGRADE TO LIDORT VERSION 3.8, March 2017, together with FO code Version 1.5

# UPGRADE TO LIDORT VERSION 3.8.1 JUNE 2019
#      4/9/19.   Removal of TRANSFLUX MASTER Routine
#      4/19/19.  Introduction of mediaprops routines (3 in all)
#      5/1/19.   Supplement and FO codes no compilation changes
#      6/3/19.   Introduce 2 new special tests (Planetary/LWCoupling)

# UPGRADE TO LIDORT VERSION 3.8.3 MARCH 2021. 
#      Doublet Geometry option (from LIDORT 3.8.2, April 2020)
#      MSST option for MS sphericity corrections
#      Use output type structure arrays filled directly
#      Converge routines (including new doublet) now in their own modules


#
# Define some variables
#   2/28/21. Version 3.8.3. Some renaming (in line with VLIDORT scheme)

UTIL_PATH       = ${LIDORT_PATH}/util
SUP_PATH        = ${LIDORT_PATH}/sup
FO_MAIN_PATH    = ${LIDORT_PATH}/lidort_focode
LID_DEF_PATH    = ${LIDORT_PATH}/lidort_def
LID_MAIN_PATH   = ${LIDORT_PATH}/lidort_main
LID_MAIN_PATH_1 = ${LIDORT_PATH}/lidort_main/regular
LID_MAIN_PATH_2 = ${LIDORT_PATH}/lidort_main/linearized
LID_TEST_PATH   = ${LIDORT_PATH}/lidort_test

MOD_PATH = mod
OBJ_PATH = obj

MOD_FILES = $(MOD_PATH)/*.mod
OBJ_FILES = $(OBJ_PATH)/*.o

#
# Define default shell make will use
#

SHELL = /bin/bash

#
# Define FORTRAN90 compiler to use (can be defined here, but usually defined on makefile command line)
#

#  Intel
#FC = ifort

#  gfortran
FC = gfortran

#  g95
#FC = g95

#  NAG
#FC = f95

#
# Define FORTRAN90 compiler flags
#

FFLAGS = -c

# Additional flags for Intel
ifeq ($(FC), ifort)
	FFLAGS := $(FFLAGS) -I$(MOD_PATH) -module $(MOD_PATH)
	FFLAGS_DEBUG = -g -warn all -check all -traceback
	FFLAGS_OPT = -O3
	FFLAGS_OPENMP = -openmp
endif

# Additional flags for gfortran
ifeq ($(FC), gfortran)
	FFLAGS := $(FFLAGS) -I$(MOD_PATH) -J$(MOD_PATH)
	FFLAGS_DEBUG  = -g -C -Wall -fbounds-check -frange-check -ffpe-trap=invalid,zero,overflow
	FFLAGS_Catch  =       -Wall -fbounds-check -frange-check -ffpe-trap=invalid,zero,overflow
	FFLAGS_OPT    = -O3
	FFLAGS_OPENMP = -fopenmp -frecursive
endif

# Additional flags for g95
ifeq ($(FC), g95)
#      older g95
	FFLAGS := $(FFLAGS) -I$(MOD_PATH) -fmod=$(MOD_PATH)
#	FFLAGS_DEBUG = -g -Wall -fbounds-check
	FFLAGS_DEBUG = -g -Wall -Wuninitialized -fbounds-check
	FFLAGS_OPT = -O3
#      g95 v0.92
#	FFLAGS := $(FFLAGS) -I$(MOD_PATH) -fmod=$(MOD_PATH)
#	FFLAGS_DEBUG = -g -Wall -fbounds-check -ftrace=full
endif

# Additional flags for NAG
#ifeq ($(FC), f95)
ifeq ($(FC), $(filter $(FC), f95 nagfor))
	FFLAGS := $(FFLAGS) -mdir $(MOD_PATH) -I$(MOD_PATH)
#	FFLAGS_DEBUG = -w=obs -w=unused -C=array -C=undefined -gline
	FFLAGS_DEBUG = -w=obs -w=unused -C=all -C=undefined -gline
#	FFLAGS_DEBUG = -w=obs -w=unused -g -C=all -C=undefined -gline -mtrace=all -nan
	FFLAGS_OPT = -O3
        FFLAGS_OPENMP = -openmp
endif

#  for compilation detailed
#	FFLAGS := $(FFLAGS) $(FFLAGS_Catch)

# For debug build, use "make DEBUG=t"
ifeq ($(DEBUG), t)
	FFLAGS := $(FFLAGS) $(FFLAGS_DEBUG)
endif

# For optimized build, use "make OPT=t"
ifeq ($(OPT), t)
	FFLAGS := $(FFLAGS) $(FFLAGS_OPT)
endif

# For parallel build using OpenMP, use "make OPENMP=t"
ifeq ($(OPENMP), t)
	FFLAGS := $(FFLAGS) $(FFLAGS_OPENMP)
endif

.SUFFIXES:

#
# Define list of source files
# (Note: ordering is important because of dependencies)
#

BASE_SOURCES =
SOURCES =
L_SOURCES =
LPS_SOURCES =
LCS_SOURCES =

BASE_SOURCES +=   \
   $(LID_DEF_PATH)/lidort_pars.f90

#BASE_SOURCES +=   \
#   $(LID_MAIN_PATH)/lidort_testpts.f90

# For RTS debug build
#ifeq ($(RTS_JAC_DEBUG), t)
#   BASE_SOURCES +=   \
#      $(LID_DEF_PATH)/rts_jac_debug.f90
#endif

SOURCES +=   \
   $(BASE_SOURCES) \
   $(LID_DEF_PATH)/lidort_inputs_def.f90        \
   $(LID_DEF_PATH)/lidort_sup_brdf_def.f90      \
   $(LID_DEF_PATH)/lidort_sup_ss_def.f90        \
   $(LID_DEF_PATH)/lidort_sup_sleave_def.f90    \
   $(LID_DEF_PATH)/lidort_sup_def.f90           \
   $(LID_DEF_PATH)/lidort_outputs_def.f90       \
   $(LID_DEF_PATH)/lidort_io_defs.f90           \
   $(LID_MAIN_PATH_1)/lidort_aux.f90              \
   $(LID_MAIN_PATH_1)/lidort_getplanck.f90  	  \
   $(LID_MAIN_PATH_1)/lidort_geometry.f90         \
   $(LID_MAIN_PATH_1)/lidort_Taylor.f90           \
   $(LID_MAIN_PATH_1)/lidort_inputs.f90           \
   $(LID_MAIN_PATH_1)/lidort_miscsetups.f90       \
   $(LID_MAIN_PATH_1)/lidort_sfo_interface.f90    \
   $(LID_MAIN_PATH_1)/lidort_thermalsup.f90       \
   $(LID_MAIN_PATH_1)/lidort_solutions.f90        \
   $(LID_MAIN_PATH_1)/lidort_bvproblem.f90        \
   $(LID_MAIN_PATH_1)/lidort_PostProcessing.f90   \
   $(LID_MAIN_PATH_1)/lidort_intensity.f90        \
   $(LID_MAIN_PATH_1)/lidort_converge.f90         \
   $(LID_MAIN_PATH_1)/lidort_writemodules.f90     \
   $(LID_MAIN_PATH_1)/lidort_mediaprops.f90       \
   $(LID_MAIN_PATH_1)/lidort_masters_V3p8p3.f90

L_SOURCES += \
   $(SOURCES) \
   $(LID_DEF_PATH)/lidort_lin_inputs_def.f90	\
   $(LID_DEF_PATH)/lidort_lin_sup_brdf_def.f90  \
   $(LID_DEF_PATH)/lidort_lin_sup_ss_def.f90	\
   $(LID_DEF_PATH)/lidort_lin_sup_sleave_def.f90\
   $(LID_DEF_PATH)/lidort_lin_sup_def.f90	\
   $(LID_DEF_PATH)/lidort_lin_outputs_def.f90	\
   $(LID_DEF_PATH)/lidort_lin_io_defs.f90	\
   $(LID_MAIN_PATH_2)/lidort_l_inputs.f90         \
   $(LID_MAIN_PATH_2)/lidort_l_writemodules.f90   \
   $(LID_MAIN_PATH_2)/lidort_la_miscsetups.f90    \
   $(LID_MAIN_PATH_2)/lidort_l_thermalsup.f90     \
   $(LID_MAIN_PATH_2)/lidort_lpc_solutions.f90    \
   $(LID_MAIN_PATH_2)/lidort_lbbf_jacobians.f90   \
   $(LID_MAIN_PATH_2)/lidort_ls_wfsurface.f90     \
   $(LID_MAIN_PATH_2)/lidort_ls_wfsleave.f90

LPS_SOURCES += \
   $(L_SOURCES) \
   $(LID_MAIN_PATH_2)/lidort_lp_miscsetups.f90      \
   $(LID_MAIN_PATH_2)/lidort_sfo_lps_interface.f90  \
   $(LID_MAIN_PATH_2)/lidort_lp_mediaprops.f90      \
   $(LID_MAIN_PATH_2)/lidort_lp_bvproblem.f90       \
   $(LID_MAIN_PATH_2)/lidort_lp_PostProcessing.f90  \
   $(LID_MAIN_PATH_2)/lidort_lp_wfatmos.f90         \
   $(LID_MAIN_PATH_2)/lidort_lps_converge.f90       \
   $(LID_MAIN_PATH_2)/lidort_lps_masters_V3p8p3.f90

LCS_SOURCES += \
   $(L_SOURCES) \
   $(LID_MAIN_PATH_2)/lidort_lc_miscsetups.f90      \
   $(LID_MAIN_PATH_2)/lidort_sfo_lcs_interface.f90  \
   $(LID_MAIN_PATH_2)/lidort_lc_mediaprops.f90      \
   $(LID_MAIN_PATH_2)/lidort_lc_bvproblem.f90       \
   $(LID_MAIN_PATH_2)/lidort_lc_PostProcessing.f90  \
   $(LID_MAIN_PATH_2)/lidort_lc_wfatmos.f90         \
   $(LID_MAIN_PATH_2)/lidort_lcs_converge.f90       \
   $(LID_MAIN_PATH_2)/lidort_lcs_masters_V3p8p3.f90

# (Include supplement source files)
include $(SUP_PATH)/makefile.sup

# (Include first-order source files)
include $(FO_MAIN_PATH)/makefile.fo_lidort

# Main tests
# ----------

#  tests 1-3

SOURCES_SOLAR += \
   $(FO_SOURCES_Scalar) \
   $(SOURCES)           \
   $(LID_TEST_PATH)/3p8p3_solar_tester.f90

SOURCES_SOLAR_LPS += \
   $(FO_SOURCES_L_Scalar) \
   $(L_SOURCES) \
   $(LPS_SOURCES)         \
   $(LID_TEST_PATH)/3p8p3_solar_lps_tester.f90

SOURCES_SOLAR_LCS += \
   $(FO_SOURCES_L_Scalar) \
   $(L_SOURCES) \
   $(LCS_SOURCES)         \
   $(LID_TEST_PATH)/3p8p3_solar_lcs_tester.f90

#  tests 4-6

SOURCES_THERMAL += \
   $(FO_SOURCES_Scalar) \
   $(SOURCES)           \
   $(BRDF_SUP_SOURCES)  \
   $(BRDF_SUP_ACCESS_SOURCES) \
   $(LID_TEST_PATH)/3p8p3_thermal_tester.f90

SOURCES_THERMAL_LPS += \
   $(FO_SOURCES_L_Scalar) \
   $(SOURCES)           \
   $(L_SOURCES) 	  \
   $(LPS_SOURCES)         \
   $(LID_TEST_PATH)/3p8p3_thermal_lps_tester.f90

SOURCES_THERMAL_LCS += \
   $(FO_SOURCES_L_Scalar) \
   $(SOURCES)           \
   $(L_SOURCES) 	  \
   $(LCS_SOURCES)         \
   $(LID_TEST_PATH)/3p8p3_thermal_lcs_tester.f90

#  test 7

SOURCES_PHASFUNC_LC += \
   $(FO_SOURCES_L_Scalar)  \
   $(SOURCES)              \
   $(LCS_SOURCES)          \
   $(PHASFUNC_SUP_SOURCES) \
   $(LID_TEST_PATH)/3p8p3_phasfunc_lc_tester.f90

#  tests 8-9

SOURCES_BRDF_SELF += \
   $(BRDF_LINSUP_SOURCES) \
   $(LID_TEST_PATH)/3p8p3_brdf_self_tester.f90

SOURCES_BRDF_FULL += \
   $(FO_SOURCES_L_Scalar)  \
   $(SOURCES)              \
   $(L_SOURCES)		   \
   $(LCS_SOURCES)          \
   $(BRDF_LINSUP_SOURCES) \
   $(BRDF_LINSUP_ACCESS_SOURCES) \
   $(LID_TEST_PATH)/3p8p3_brdf_full_tester.f90

#  tests 15-17 (OMP)

SOURCES_SOLAR_OMP += \
   $(FO_SOURCES_Scalar) \
   $(SOURCES)           \
   $(LID_TEST_PATH)/3p8p3_solar_OMP_tester.f90

SOURCES_SOLAR_LPS_OMP += \
   $(FO_SOURCES_L_Scalar) \
   $(LPS_SOURCES)         \
   $(LID_TEST_PATH)/3p8p3_solar_lps_OMP_tester.f90

SOURCES_SOLAR_LCS_OMP += \
   $(FO_SOURCES_L_Scalar) \
   $(LCS_SOURCES)         \
   $(LID_TEST_PATH)/3p8p3_solar_lcs_OMP_tester.f90

#  SPECIAL TESTS (Need to use dedicated PARS files first)
#  -------------

# Tests 10-12

SOURCES_SLEAVE_SELF += \
   $(SLEAVE_LINSUP_SOURCES) \
   $(LID_TEST_PATH)/3p8p3_sleave_self_tester.f90

SOURCES_SLEAVE_FULL += \
   $(FO_SOURCES_L_Scalar)    \
   $(SOURCES) 		     \
   $(LCS_SOURCES)            \
   $(LPS_SOURCES)            \
   $(SLEAVE_LINSUP_SOURCES)  \
   $(SLEAVE_LINSUP_ACCESS_SOURCES) \
   $(LID_TEST_PATH)/3p8p3_sleave_full_tester.f90

SOURCES_THERMAL_LBBF += \
   $(FO_SOURCES_L_Scalar)  \
   $(LPS_SOURCES)          \
   $(BRDF_LINSUP_SOURCES) \
   $(BRDF_LINSUP_ACCESS_SOURCES) \
   $(LID_TEST_PATH)/3p8p3_thermal_lbbf_tester.f90

#  tests 13-14 (PLANETARY/WLADJUST).

SOURCES_FULL_PLANETARY += \
   $(FO_SOURCES_L_Scalar) \
   $(L_SOURCES)           \
   $(LCS_SOURCES)         \
   $(LPS_SOURCES)         \
   $(LID_TEST_PATH)/3p8p3_Planetary_tester.f90

SOURCES_LWCOUPLING += \
   $(FO_SOURCES_Scalar) \
   $(SOURCES)           \
   $(SLEAVE_SUP_SOURCES) \
   $(SLEAVE_SUP_ACCESS_SOURCES) \
   $(LID_TEST_PATH)/3p8p3_LWCoupling_tester.f90

# Utilities

SOURCES_UTIL_LID =
SOURCES_UTIL_LID += \
   $(UTIL_PATH)/lidort_diff.f90

SOURCES_003_GSIT_NON_HOMO += \
   $(FO_SOURCES_Scalar) \
   $(SOURCES)           \
   3p8p3_003_gsit_deeper_non_homo.f90

SOURCES_004_GSIT_BRDF += \
   $(FO_SOURCES_L_Scalar)  \
   $(SOURCES)              \
   $(L_SOURCES)		   \
   $(LCS_SOURCES)          \
   $(BRDF_LINSUP_SOURCES) \
   $(BRDF_LINSUP_ACCESS_SOURCES) \
   3p8p3_004_gsit_brdf.f90

SOURCES_005_TEST_PLOT += \
   $(FO_SOURCES_L_Scalar)  \
   $(SOURCES)              \
   $(L_SOURCES)		   \
   $(LCS_SOURCES)          \
   $(BRDF_LINSUP_SOURCES) \
   $(BRDF_LINSUP_ACCESS_SOURCES) \
   3p8p3_005_test_plot.f90

#
# Define pattern rules for creating object files:
#

#.SUFFIXES:

# For lidort main source files
$(OBJ_PATH)/%.o : $(LID_DEF_PATH)/%.f90
	$(FC) $(FFLAGS) $< -o $@
$(OBJ_PATH)/%.o : $(LID_MAIN_PATH_)/%.f90
	$(FC) $(FFLAGS) $< -o $@
$(OBJ_PATH)/%.o : $(LID_MAIN_PATH_1)/%.f90
	$(FC) $(FFLAGS) $< -o $@
$(OBJ_PATH)/%.o : $(LID_MAIN_PATH_2)/%.f90
	$(FC) $(FFLAGS) $< -o $@
$(OBJ_PATH)/%.o : $(LID_TEST_PATH)/%.f90
	$(FC) $(FFLAGS) $< -o $@

# For utility source files
$(OBJ_PATH)/%.o : $(UTIL_PATH)/%.f90
	$(FC) $(FFLAGS) $< -o $@

# For source files in the current directory (ADDED RULE)
$(OBJ_PATH)/%.o : %.f90
	$(FC) $(FFLAGS) $< -o $@


#
# Define object files

#  Main tests
#  ----------

#  tests 1-3

F90SOURCES_SOLAR := $(notdir $(filter %.f90, $(SOURCES_SOLAR)))
F90OBJECTS_SOLAR := $(patsubst %.f90, %.o, $(addprefix $(OBJ_PATH)/, $(F90SOURCES_SOLAR)))

F90SOURCES_SOLAR_LPS := $(notdir $(filter %.f90, $(SOURCES_SOLAR_LPS)))
F90OBJECTS_SOLAR_LPS := $(patsubst %.f90, %.o, $(addprefix $(OBJ_PATH)/, $(F90SOURCES_SOLAR_LPS)))

F90SOURCES_SOLAR_LCS := $(notdir $(filter %.f90, $(SOURCES_SOLAR_LCS)))
F90OBJECTS_SOLAR_LCS := $(patsubst %.f90, %.o, $(addprefix $(OBJ_PATH)/, $(F90SOURCES_SOLAR_LCS)))

#  tests 4-6

F90SOURCES_THERMAL := $(notdir $(filter %.f90, $(SOURCES_THERMAL)))
F90OBJECTS_THERMAL := $(patsubst %.f90, %.o, $(addprefix $(OBJ_PATH)/, $(F90SOURCES_THERMAL)))

F90SOURCES_THERMAL_LPS := $(notdir $(filter %.f90, $(SOURCES_THERMAL_LPS)))
F90OBJECTS_THERMAL_LPS := $(patsubst %.f90, %.o, $(addprefix $(OBJ_PATH)/, $(F90SOURCES_THERMAL_LPS)))

F90SOURCES_THERMAL_LCS := $(notdir $(filter %.f90, $(SOURCES_THERMAL_LCS)))
F90OBJECTS_THERMAL_LCS := $(patsubst %.f90, %.o, $(addprefix $(OBJ_PATH)/, $(F90SOURCES_THERMAL_LCS)))

#  tests 7

F90SOURCES_PHASFUNC_LC := $(notdir $(filter %.f90, $(SOURCES_PHASFUNC_LC)))
F90OBJECTS_PHASFUNC_LC := $(patsubst %.f90, %.o, $(addprefix $(OBJ_PATH)/, $(F90SOURCES_PHASFUNC_LC)))

#  tests 8-9

F90SOURCES_BRDF_SELF := $(notdir $(filter %.f90, $(SOURCES_BRDF_SELF)))
F90OBJECTS_BRDF_SELF := $(patsubst %.f90, %.o, $(addprefix $(OBJ_PATH)/, $(F90SOURCES_BRDF_SELF)))

F90SOURCES_BRDF_FULL := $(notdir $(filter %.f90, $(SOURCES_BRDF_FULL)))
F90OBJECTS_BRDF_FULL := $(patsubst %.f90, %.o, $(addprefix $(OBJ_PATH)/, $(F90SOURCES_BRDF_FULL)))

#  tests 15-17 (OMP)

F90SOURCES_SOLAR_OMP := $(notdir $(filter %.f90, $(SOURCES_SOLAR_OMP)))
F90OBJECTS_SOLAR_OMP := $(patsubst %.f90, %.o, $(addprefix $(OBJ_PATH)/, $(F90SOURCES_SOLAR_OMP)))

F90SOURCES_SOLAR_LPS_OMP := $(notdir $(filter %.f90, $(SOURCES_SOLAR_LPS_OMP)))
F90OBJECTS_SOLAR_LPS_OMP := $(patsubst %.f90, %.o, $(addprefix $(OBJ_PATH)/, $(F90SOURCES_SOLAR_LPS_OMP)))

F90SOURCES_SOLAR_LCS_OMP := $(notdir $(filter %.f90, $(SOURCES_SOLAR_LCS_OMP)))
F90OBJECTS_SOLAR_LCS_OMP := $(patsubst %.f90, %.o, $(addprefix $(OBJ_PATH)/, $(F90SOURCES_SOLAR_LCS_OMP)))

#  Special tests
#  -------------

#  tests 10-12

F90SOURCES_SLEAVE_SELF := $(notdir $(filter %.f90, $(SOURCES_SLEAVE_SELF)))
F90OBJECTS_SLEAVE_SELF := $(patsubst %.f90, %.o, $(addprefix $(OBJ_PATH)/, $(F90SOURCES_SLEAVE_SELF)))

F90SOURCES_SLEAVE_FULL := $(notdir $(filter %.f90, $(SOURCES_SLEAVE_FULL)))
F90OBJECTS_SLEAVE_FULL := $(patsubst %.f90, %.o, $(addprefix $(OBJ_PATH)/, $(F90SOURCES_SLEAVE_FULL)))

F90SOURCES_THERMAL_LBBF := $(notdir $(filter %.f90, $(SOURCES_THERMAL_LBBF)))
F90OBJECTS_THERMAL_LBBF := $(patsubst %.f90, %.o, $(addprefix $(OBJ_PATH)/, $(F90SOURCES_THERMAL_LBBF)))

#  tests 13-14. New 3.8.1  6/3/19

F90SOURCES_FULL_PLANETARY := $(notdir $(filter %.f90, $(SOURCES_FULL_PLANETARY)))
F90OBJECTS_FULL_PLANETARY := $(patsubst %.f90, %.o, $(addprefix $(OBJ_PATH)/, $(F90SOURCES_FULL_PLANETARY)))

F90SOURCES_LWCOUPLING := $(notdir $(filter %.f90, $(SOURCES_LWCOUPLING)))
F90OBJECTS_LWCOUPLING := $(patsubst %.f90, %.o, $(addprefix $(OBJ_PATH)/, $(F90SOURCES_LWCOUPLING)))

#  Utilities

F90SOURCES_UTIL_LID := $(notdir $(filter %.f90, $(SOURCES_UTIL_LID)))
F90OBJECTS_UTIL_LID := $(patsubst %.f90, %.o, $(addprefix $(OBJ_PATH)/, $(F90SOURCES_UTIL_LID)))

# GSIT

F90SOURCES_003_GSIT_NON_HOMO := $(notdir $(filter %.f90, $(SOURCES_003_GSIT_NON_HOMO)))
F90OBJECTS_003_GSIT_NON_HOMO := $(patsubst %.f90, %.o, $(addprefix $(OBJ_PATH)/, $(F90SOURCES_003_GSIT_NON_HOMO)))

F90SOURCES_004_GSIT_BRDF := $(notdir $(filter %.f90, $(SOURCES_004_GSIT_BRDF)))
F90OBJECTS_004_GSIT_BRDF := $(patsubst %.f90, %.o, $(addprefix $(OBJ_PATH)/, $(F90SOURCES_004_GSIT_BRDF)))

F90SOURCES_005_TEST_PLOT := $(notdir $(filter %.f90, $(SOURCES_005_TEST_PLOT)))
F90OBJECTS_005_TEST_PLOT := $(patsubst %.f90, %.o, $(addprefix $(OBJ_PATH)/, $(F90SOURCES_005_TEST_PLOT)))

#
# Define desired targets
#

main: solar    \
      thermal  \
      phasfunc \
      brdf \
      gsit

# Tests 1-3

solar: 3p8p3_solar_tester.exe \
       3p8p3_solar_lps_tester.exe \
       3p8p3_solar_lcs_tester.exe

# Tests 4-6

thermal: 3p8p3_thermal_tester.exe \
         3p8p3_thermal_lps_tester.exe \
         3p8p3_thermal_lcs_tester.exe

# Test 7

phasfunc: 3p8p3_phasfunc_lc_tester.exe

# Tests 8-9

brdf: 3p8p3_brdf_self_tester.exe \
      3p8p3_brdf_full_tester.exe

gsit: 3p8p3_003_gsit_deeper_non_homo.exe \
      3p8p3_004_gsit_brdf.exe \
      3p8p3_005_test_plot.exe

# Tests 10-14 ( SEE BELOW )

# Tests 15-17

OMP: 3p8p3_solar_OMP_tester.exe \
     3p8p3_solar_lps_OMP_tester.exe \
     3p8p3_solar_lcs_OMP_tester.exe

# Selected standard tests with OBSERVATIONAL GEOMETRY (& DOUBLET) options

obsgeo: 3p8p3_solar_tester.exe \
        3p8p3_solar_lps_tester.exe \
        3p8p3_solar_lcs_tester.exe \
        3p8p3_brdf_self_tester.exe \
        3p8p3_brdf_full_tester.exe \
        3p8p3_phasfunc_lc_tester.exe

# Special Tests
# -------------

# Tests 10-14 not included in "main" or "obsgeo" (require different "lidort_pars.f90" settings)

# Tests 10-11

sleave: 3p8p3_sleave_self_tester.exe \
        3p8p3_sleave_full_tester.exe

# Test 12

lbbf:   3p8p3_thermal_lbbf_tester.exe

# Tests 13-14. New 3.8.1, 6/3/19.

Planetary  : 3p8p3_Planetary_tester.exe
LWCoupling : 3p8p3_LWCoupling_tester.exe

# Selected special tests with OBSERVATIONAL GEOMETRY (& DOUBLET) option

obsgeo_sleave_full: 3p8p3_sleave_full_tester.exe

obsgeo_Planetary: 3p8p3_Planetary_tester.exe

obsgeo_LWCoupling: 3p8p3_LWCoupling_tester.exe

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

#  Main tests
#  ----------

#  Tests 1-3

3p8p3_solar_tester.exe: $(F90OBJECTS_SOLAR)
	$(FC) $^ -o $@
3p8p3_solar_lps_tester.exe: $(F90OBJECTS_SOLAR_LPS)
	$(FC) $^ -o $@
3p8p3_solar_lcs_tester.exe: $(F90OBJECTS_SOLAR_LCS)
	$(FC) $^ -o $@

#  Tests 4-6

3p8p3_thermal_tester.exe: $(F90OBJECTS_THERMAL)
	$(FC) $^ -o $@
3p8p3_thermal_lps_tester.exe: $(F90OBJECTS_THERMAL_LPS)
	$(FC) $^ -o $@
3p8p3_thermal_lcs_tester.exe: $(F90OBJECTS_THERMAL_LCS)
	$(FC) $^ -o $@

#  Test 7

3p8p3_phasfunc_lc_tester.exe: $(F90OBJECTS_PHASFUNC_LC)
	$(FC) $^ -o $@

#  Tests 8-9

3p8p3_brdf_self_tester.exe: $(F90OBJECTS_BRDF_SELF)
	$(FC) $^ -o $@
3p8p3_brdf_full_tester.exe: $(F90OBJECTS_BRDF_FULL)
	$(FC) $^ -o $@

#  Tests 21-23

3p8p3_solar_OMP_tester.exe: $(F90OBJECTS_SOLAR_OMP)
	$(FC) $(FFLAGS_OPENMP) $^ -o $@
3p8p3_solar_lps_OMP_tester.exe: $(F90OBJECTS_SOLAR_LPS_OMP)
	$(FC) $(FFLAGS_OPENMP) $^ -o $@
3p8p3_solar_lcs_OMP_tester.exe: $(F90OBJECTS_SOLAR_LCS_OMP)
	$(FC) $(FFLAGS_OPENMP) $^ -o $@

#3p8p3_solar_OMP_tester.exe: $(F90OBJECTS_SOLAR_OMP)
#	$(FC) $^ -o $@
#3p8p3_solar_lps_OMP_tester.exe: $(F90OBJECTS_SOLAR_LPS_OMP)
#	$(FC) $^ -o $@
#3p8p3_solar_lcs_OMP_tester.exe: $(F90OBJECTS_SOLAR_LCS_OMP)
#	$(FC) $^ -o $@

#  specials
#  --------

# Tests 10-12

3p8p3_sleave_self_tester.exe: $(F90OBJECTS_SLEAVE_SELF)
	$(FC) $^ -o $@
3p8p3_sleave_full_tester.exe: $(F90OBJECTS_SLEAVE_FULL)
	$(FC) $^ -o $@
3p8p3_thermal_lbbf_tester.exe: $(F90OBJECTS_THERMAL_LBBF)
	$(FC) $^ -o $@

# Tests 13-14. New 3.8.1    6/3/19.

3p8p3_Planetary_tester.exe: $(F90OBJECTS_FULL_PLANETARY)
	$(FC) $^ -o $@
3p8p3_LWCoupling_tester.exe: $(F90OBJECTS_LWCOUPLING)
	$(FC) $^ -o $@

# Diff utilities

lidort_diff: $(F90OBJECTS_UTIL_LID)
	$(FC) $^ -o $@


3p8p3_003_gsit_deeper_non_homo.exe: $(F90OBJECTS_003_GSIT_NON_HOMO)
	$(FC) $^ -o $@

3p8p3_004_gsit_brdf.exe: $(F90OBJECTS_004_GSIT_BRDF)
	$(FC) $^ -o $@

3p8p3_005_test_plot.exe: $(F90OBJECTS_005_TEST_PLOT)
	$(FC) $^ -o $@

# Make clean

.PHONY: clean
clean:
	rm -f *.o $(OBJ_FILES) *.mod $(MOD_FILES) *.log *.exe

