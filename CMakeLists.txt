#
# Copyright (c) 2015, OK おじさん(岡久卓也)
# Copyright (c) 2015, OK Ojisan(Takuya OKAHISA)
# Copyright (c) 2016 to the present, DEWETRON GmbH
# OTFFT version 8.1
# based on Stockham FFT algorithm
# from OK Ojisan(Takuya OKAHISA), source: http://www.moon.sannet.ne.jp/okahisa/stockham/stockham.html
#

cmake_minimum_required(VERSION 3.22)

set(LIBNAME otfft)
project(otfft)

#
# otfft Standalone target support
if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  #
  # Set used policies

  # (CMP0017, CMP0026, CMP0043 and CMP0058 are NEW since CMake 3.x/4;
  # their OLD behaviour is no longer supported and not needed here.)

  #
  # define PROJECT_ROOT
  get_filename_component(PROJECT_ROOT ../.. ABSOLUTE)

  # expand cmake search path to check for project settings
  set(CMAKE_MODULE_PATH
    ${CMAKE_MODULE_PATH}
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake
  )

  #
  # Speed up compile time
  set(MSVC_BUILD_USING_MP TRUE)

  #
  # Get general cmake settings
  include(CMakeSettings)
  
  #
  # Enable unit tests
  enable_testing()
  include (CTest)

  #
  # Set the output directory so that all artefacts will be at the same place
  SetCommonOutputDirectory()
  message(STATUS "CMAKE_LIBRARY_OUTPUT_DIRECTORY = ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
  message(STATUS "CMAKE_ARCHIVE_OUTPUT_DIRECTORY = ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}")
  message(STATUS "CMAKE_RUNTIME_OUTPUT_DIRECTORY = ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")


  # Support for OpenMP
  if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 3.7)
    message(STATUS "OpenMP disabled")
  else()
    find_package(OpenMP REQUIRED)
  endif()

endif()

if(OPENMP_FOUND)
  message(STATUS "Using OpenMP flags: ${OpenMP_CXX_FLAGS}")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif(OPENMP_FOUND)

#
# compile options
# The AVX-512 backend is opt-in (roadmap R2.1, decision D32): it is chosen at run time only when
# a caller asks for OPTIMIZED_FFT_AVX512, and it has no AVX-512 kernels yet.
option(OTFFT_WITH_AVX512 "Build the AVX-512 backend (64 bit only, needs AVX512F, AVX512DQ and AVX512VL; not chosen automatically)" OFF)

if (CMAKE_SIZEOF_VOID_P EQUAL 8)
  set(OTFFT_WITH_SSE2 TRUE)
  set(OTFFT_WITH_AVX  TRUE)
  set(OTFFT_WITH_AVX2 TRUE)
else()
  set(OTFFT_WITH_SSE2 TRUE)
  set(OTFFT_WITH_AVX  FALSE)
  set(OTFFT_WITH_AVX2 FALSE)
  if (OTFFT_WITH_AVX512)
    message(WARNING "OTFFT_WITH_AVX512 is ignored for a 32 bit build")
    set(OTFFT_WITH_AVX512 FALSE)
  endif()
endif()

#
# Generate otfft_config.h file
configure_file(src/otfft_config.h.in ${CMAKE_CURRENT_LIST_DIR}/inc/otfft_config.h)

#
# Library includes have pedantic warnings switched on
include_directories(
  inc
  src
)

set(PUBLIC_OTFFT_HEADER_FILES
  inc/otfft_fwd.h
  inc/otfft_complex.h
  inc/otfft_config.h
  inc/otfft.h
  inc/otfft_types.h
)

source_group("Public Header Files" FILES
  ${PUBLIC_OTFFT_HEADER_FILES}
)

set(OTFFT_HEADER_FILES
  src/otfft_config.h.in
  src/otfft_avx.h
  src/otfft_avx2.h
  src/otfft_avx512.h
  src/otfft_sse2.h
  src/otfft_avxdif4.h
  src/otfft_avxdif4omp.h
  src/otfft_avxdif8.h
  src/otfft_avxdif8omp.h
  src/otfft_avxdif8exp.h
  src/otfft_avxdif16.h
  src/otfft_avxdif16omp.h
  src/otfft_avxdif16exp.h
  src/otfft_avxdit4.h
  src/otfft_avxdit4omp.h
  src/otfft_avxdit8.h
  src/otfft_avxdit8omp.h
  src/otfft_avxdit8exp.h
  src/otfft_avxdit16.h
  src/otfft_avxdit16omp.h
  src/otfft_avxdit16exp.h
  src/otfft_base.h
  src/otfft_eightstep.h
  src/otfft_factory.h
  src/otfft_if.h
  src/otfft_misc.h
  src/otfft_mixedradix.h
  src/otfft_platform.h
  src/otfft_sixstep.h
  src/otfft_sixstepsq.h
  src/otfft_tuning.h
)

source_group("Header Files" FILES
  ${OTFFT_HEADER_FILES}
)

set(OTFFT_SOURCE_FILES
  src/otfft.cpp
  src/otfft_tuning.cpp
)

set(OTFFT_SOURCE_FILES_ALGO
  src/otfft_algo.cpp
  src/otfft_algo_dif16.cpp
  src/otfft_algo_dif4.cpp
  src/otfft_algo_dif8.cpp
  src/otfft_algo_dit16.cpp
  src/otfft_algo_dit4.cpp
  src/otfft_algo_dit8.cpp
  src/otfft_algo_mixed.cpp
  src/otfft_algo_sixstep.cpp
  src/otfft_algo_twophase.cpp
)

source_group("Source Files" FILES
  ${OTFFT_SOURCE_FILES}
)

add_library(${LIBNAME}_common STATIC
  ${PUBLIC_OTFFT_HEADER_FILES}
  ${OTFFT_HEADER_FILES}
  ${OTFFT_SOURCE_FILES}
)
set_property(TARGET ${LIBNAME}_common APPEND_STRING PROPERTY COMPILE_PDB_NAME ${LIBNAME}_common)

if (NOT OTFFT_WITH_SSE2 AND NOT OTFFT_WITH_AVX AND NOT OTFFT_WITH_AVX2)
  message(FATAL_ERROR "Please choose at least one backend [SSE2, AVX, AVX2]")
endif()

set(OTFFT_BACKENDS "")
if (OTFFT_WITH_SSE2)
  add_library(${LIBNAME}_sse2 STATIC
    ${PUBLIC_OTFFT_HEADER_FILES}
    ${OTFFT_HEADER_FILES}
    ${OTFFT_SOURCE_FILES_ALGO}
    )
  set_property(TARGET ${LIBNAME}_sse2 APPEND_STRING PROPERTY COMPILE_PDB_NAME ${LIBNAME}_sse2)
  list(APPEND OTFFT_BACKENDS ${LIBNAME}_sse2)
endif()

if (OTFFT_WITH_AVX)
  add_library(${LIBNAME}_avx STATIC
    ${PUBLIC_OTFFT_HEADER_FILES}
    ${OTFFT_HEADER_FILES}
    ${OTFFT_SOURCE_FILES_ALGO}
    )
  set_property(TARGET ${LIBNAME}_avx APPEND_STRING PROPERTY COMPILE_PDB_NAME ${LIBNAME}_avx)
  list(APPEND OTFFT_BACKENDS ${LIBNAME}_avx)
endif()

if (OTFFT_WITH_AVX2)
  add_library(${LIBNAME}_avx2 STATIC
    ${PUBLIC_OTFFT_HEADER_FILES}
    ${OTFFT_HEADER_FILES}
    ${OTFFT_SOURCE_FILES_ALGO}
    )
  set_property(TARGET ${LIBNAME}_avx2 APPEND_STRING PROPERTY COMPILE_PDB_NAME ${LIBNAME}_avx2)
  list(APPEND OTFFT_BACKENDS ${LIBNAME}_avx2)
endif()

if (OTFFT_WITH_AVX512)
  add_library(${LIBNAME}_avx512 STATIC
    ${PUBLIC_OTFFT_HEADER_FILES}
    ${OTFFT_HEADER_FILES}
    ${OTFFT_SOURCE_FILES_ALGO}
    )
  set_property(TARGET ${LIBNAME}_avx512 APPEND_STRING PROPERTY COMPILE_PDB_NAME ${LIBNAME}_avx512)
  list(APPEND OTFFT_BACKENDS ${LIBNAME}_avx512)
endif()

MESSAGE("Using backends ${OTFFT_BACKENDS}")

include(mergestaticlibs)
merge_static_libs(${LIBNAME}
  ${LIBNAME}_common
  ${OTFFT_BACKENDS}
)
set_property(TARGET ${LIBNAME} APPEND_STRING PROPERTY COMPILE_PDB_NAME ${LIBNAME})

target_include_directories(${LIBNAME} SYSTEM
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/inc
)

set_library_export_flag(${LIBNAME})

SetFullOptimization()

#
# compiler specific optimization flags

message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID}")

# Treat GNU Compiler and CLang Compiler the same; clang-cl takes the MSVC command line and goes to the MSVC branch
if ((CMAKE_COMPILER_IS_GNUCXX OR ${CMAKE_CXX_COMPILER_ID} MATCHES "Clang") AND NOT MSVC)
  message(STATUS "Optimize for GNU-like compiler")
  if (OTFFT_WITH_SSE2)
    set(OTFFT_BACKENDS_FLAGS
      " ${OTFFT_BACKENDS_FLAGS} -DUSE_SSE2 "
    )
  endif()
  if (OTFFT_WITH_AVX)
    set(OTFFT_BACKENDS_FLAGS
      " ${OTFFT_BACKENDS_FLAGS} -DUSE_AVX "
    )
  endif()
  if (OTFFT_WITH_AVX2)
    set(OTFFT_BACKENDS_FLAGS
      " ${OTFFT_BACKENDS_FLAGS} -DUSE_AVX2 "
    )
  endif()
  if (OTFFT_WITH_AVX512)
    set(OTFFT_BACKENDS_FLAGS
      " ${OTFFT_BACKENDS_FLAGS} -DUSE_AVX512 "
    )
  endif()
  # -fsignaling-nans exists only in GCC; Clang warns that it ignores it
  set(OTFFT_SNAN_FLAG "")
  if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(OTFFT_SNAN_FLAG "-fsignaling-nans")
  endif()
  set(DEFAULT_FLAGS
    " -std=c++11 -frounding-math -fno-finite-math-only ${OTFFT_SNAN_FLAG} -pedantic -Wall -Wextra -fpermissive ${OpenMP_CXX_FLAGS} "
  )
  # -msse2 and -mavx only add features. Some compilers (the Ubuntu 26.04 GCC on the GitHub runners) enable AVX2
  # and FMA by default, so each backend switches off what it must not use (D33).
  set(SSE2_FLAGS
    " ${DEFAULT_FLAGS} -mno-avx -msse2 -DUSE_SSE2 -DOTFFT_NAMESPACE=OTFFT_SSE2 "
  )
  set(AVX_FLAGS
    " ${DEFAULT_FLAGS} -mno-avx2 -mno-fma -mavx -DUSE_AVX -DOTFFT_NAMESPACE=OTFFT_AVX "
  )
  # -mavx2 does not enable FMA in GCC and Clang; the AVX2 backend uses it (otfft_misc.h tests __FMA__,
  # as MSVC defines it for /arch:AVX2) and is chosen at run time only if the CPU has FMA too (D21).
  set(AVX2_FLAGS
    " ${DEFAULT_FLAGS} -mno-avx512f -mavx2 -mfma -DUSE_AVX2 -DOTFFT_NAMESPACE=OTFFT_AVX2 "
  )
  # The AVX-512 backend: the code of the AVX2 backend compiled for AVX-512 (VEX and EVEX encodings, the
  # compiler keeps 256-bit vectors by default); it needs AVX512F, AVX512DQ and AVX512VL at run time.
  set(AVX512_FLAGS
    " ${DEFAULT_FLAGS} -mavx2 -mfma -mavx512f -mavx512dq -mavx512vl -DUSE_AVX512 -DOTFFT_NAMESPACE=OTFFT_AVX512 "
  )
# Intel C++ Compiler
elseif (${CMAKE_CXX_COMPILER_ID} MATCHES "Intel")
  message(STATUS "Optimize for Intel compiler")
  if (OTFFT_WITH_SSE2)
    set(OTFFT_BACKENDS_FLAGS
      " ${OTFFT_BACKENDS_FLAGS} -DUSE_SSE2 "
    )
  endif()
  if (OTFFT_WITH_AVX)
    set(OTFFT_BACKENDS_FLAGS
      " ${OTFFT_BACKENDS_FLAGS} -DUSE_AVX "
    )
  endif()
  if (OTFFT_WITH_AVX2)
    set(OTFFT_BACKENDS_FLAGS
      " ${OTFFT_BACKENDS_FLAGS} -DUSE_AVX2 "
    )
  endif()
  set(DEFAULT_FLAGS
    " -std=c++11 -w2 ${OpenMP_CXX_FLAGS} "
  )
  set(SSE2_FLAGS
    " ${DEFAULT_FLAGS} -msse2 -DUSE_SSE2 -DOTFFT_NAMESPACE=OTFFT_SSE2 "
  )
  set(AVX_FLAGS
    " ${DEFAULT_FLAGS} -mavx -DUSE_AVX -DOTFFT_NAMESPACE=OTFFT_AVX "
  )
  set(AVX2_FLAGS
    " ${DEFAULT_FLAGS} -mavx2 -DUSE_AVX2 -DOTFFT_NAMESPACE=OTFFT_AVX2 "
  )
  if (OTFFT_WITH_AVX512)
    message(FATAL_ERROR "OTFFT_WITH_AVX512 is not supported with the Intel compiler yet")
  endif()
# Microsoft Visual C++ Compiler
elseif (MSVC OR ${CMAKE_CXX_COMPILER_ID} MATCHES "MSVC")
  message(STATUS "Optimize for MSVC compiler")
  if ("${CMAKE_VERSION}" VERSION_EQUAL "3.1" OR "${CMAKE_VERSION}" VERSION_GREATER "3.1")
    message("Configure PDB file generation")
    set(ADDITIONAL_FLAGS
      " /Zi "
    )
  else()
    message(DEPRECATION "Please use CMAKE 3.1 or newer to support PDB file generation")
  endif()
  if (OTFFT_WITH_SSE2)
    set(OTFFT_BACKENDS_FLAGS
      " ${OTFFT_BACKENDS_FLAGS} /DUSE_SSE2 "
    )
  endif()
  if (OTFFT_WITH_AVX)
    set(OTFFT_BACKENDS_FLAGS
      " ${OTFFT_BACKENDS_FLAGS} /DUSE_AVX "
    )
  endif()
  if (OTFFT_WITH_AVX2)
    set(OTFFT_BACKENDS_FLAGS
      " ${OTFFT_BACKENDS_FLAGS} /DUSE_AVX2 "
    )
  endif()
  if (OTFFT_WITH_AVX512)
    set(OTFFT_BACKENDS_FLAGS
      " ${OTFFT_BACKENDS_FLAGS} /DUSE_AVX512 "
    )
  endif()
  if (CMAKE_CL_64)
    set(DEFAULT_FLAGS
      " ${ADDITIONAL_FLAGS} /Oi /favor:INTEL64 ${OpenMP_CXX_FLAGS} "
    )
    # The SSE2 code uses SSE3 intrinsics (_mm_addsub_pd, _mm_hadd_pd); MSVC accepts them anywhere and
    # inc/otfft_complex.h assumes SSE3 on x64, clang-cl needs the feature switched on.
    set(OTFFT_CLANG_CL_SSE3 "")
    if (${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
      set(OTFFT_CLANG_CL_SSE3 "-msse3")
    endif()
    set(SSE2_FLAGS
      " ${DEFAULT_FLAGS} ${OTFFT_CLANG_CL_SSE3} /DUSE_SSE2 /DOTFFT_NAMESPACE=OTFFT_SSE2 "
    )
  else()
    set(DEFAULT_FLAGS
      " ${ADDITIONAL_FLAGS} /Oi /favor:blend ${OpenMP_CXX_FLAGS} "
    )
    set(SSE2_FLAGS
      " ${DEFAULT_FLAGS} /arch:SSE2 /DUSE_SSE2 /DOTFFT_NAMESPACE=OTFFT_SSE2 "
    )
  endif()
  set(AVX_FLAGS
    " ${DEFAULT_FLAGS} /arch:AVX /DUSE_AVX /DOTFFT_NAMESPACE=OTFFT_AVX "
  )
  # clang-cl, unlike MSVC, does not enable FMA with /arch:AVX2 (D21)
  set(OTFFT_CLANG_CL_FMA "")
  if (${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
    set(OTFFT_CLANG_CL_FMA "-mfma")
  endif()
  set(AVX2_FLAGS
    " ${DEFAULT_FLAGS} /arch:AVX2 ${OTFFT_CLANG_CL_FMA} /DUSE_AVX2 /DOTFFT_NAMESPACE=OTFFT_AVX2 "
  )
  set(AVX512_FLAGS
    " ${DEFAULT_FLAGS} /arch:AVX512 ${OTFFT_CLANG_CL_FMA} /DUSE_AVX512 /DOTFFT_NAMESPACE=OTFFT_AVX512 "
  )
endif()

set_property(TARGET ${LIBNAME}_common APPEND_STRING PROPERTY
  COMPILE_FLAGS
  " ${DEFAULT_FLAGS} ${OTFFT_BACKENDS_FLAGS} "
)
if (OTFFT_WITH_SSE2)
  set_property(TARGET ${LIBNAME}_sse2 APPEND_STRING PROPERTY
    COMPILE_FLAGS
    " ${SSE2_FLAGS} "
  )
endif()
if (OTFFT_WITH_AVX)
  set_property(TARGET ${LIBNAME}_avx APPEND_STRING PROPERTY
    COMPILE_FLAGS
    " ${AVX_FLAGS} "
  )
endif()
if (OTFFT_WITH_AVX2)
  set_property(TARGET ${LIBNAME}_avx2 APPEND_STRING PROPERTY
    COMPILE_FLAGS
    " ${AVX2_FLAGS} "
  )
endif()
if (OTFFT_WITH_AVX512)
  set_property(TARGET ${LIBNAME}_avx512 APPEND_STRING PROPERTY
    COMPILE_FLAGS
    " ${AVX512_FLAGS} "
  )
endif()

#
# add this to Visual Studio group app
set_target_properties(${LIBNAME} PROPERTIES FOLDER "3rd_party")

#
# 3rd-party build support
if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  install(TARGETS ${LIBNAME} ARCHIVE
    DESTINATION lib
  )
  install(FILES ${PUBLIC_OTFFT_HEADER_FILES}
    DESTINATION include
  )
endif()

if (UNIX)
  find_program(MEMORYCHECK_COMMAND valgrind)
  set(MEMORYCHECK_COMMAND_OPTIONS "--trace-children=yes --leak-check=full" )
endif(UNIX)

option(OTFFT_BUILD_TESTS "Build the unit tests (doctest)" ON)
if (OTFFT_BUILD_TESTS)
  add_subdirectory(unit_tests)
endif()

option(OTFFT_BUILD_BENCHMARKS "Build the benchmarks (compare with FFTW, PocketFFT, KissFFT)" OFF)
if (OTFFT_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

option(OTFFT_BUILD_TOOLS "Build the tools (ffttune: find the fastest algorithm per length)" OFF)
if (OTFFT_BUILD_TOOLS)
  add_subdirectory(tools)
endif()

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(OTFFT_EXAMPLES_DEFAULT ON)
else()
  set(OTFFT_EXAMPLES_DEFAULT OFF)
endif()
option(OTFFT_BUILD_EXAMPLES "Build the examples (small programs using the public API; ON when otfft is the top-level project)" ${OTFFT_EXAMPLES_DEFAULT})
if (OTFFT_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()
