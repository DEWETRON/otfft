#
# otfft benchmarks: compare OTFFT with FFTW, PocketFFT and KissFFT.
# FFTW is optional (found with pkg-config); PocketFFT and KissFFT are vendored.

find_package(PkgConfig QUIET)
set(FFTW_FOUND FALSE)
if(PkgConfig_FOUND)
  pkg_check_modules(FFTW QUIET IMPORTED_TARGET fftw3)
endif()

# KissFFT, double precision
add_library(kissfft STATIC
  ${PROJECT_SOURCE_DIR}/third_party/kissfft/kiss_fft.c
  ${PROJECT_SOURCE_DIR}/third_party/kissfft/kiss_fftr.c
)
target_compile_definitions(kissfft PUBLIC kiss_fft_scalar=double)
target_include_directories(kissfft SYSTEM PUBLIC ${PROJECT_SOURCE_DIR}/third_party/kissfft)
target_compile_options(kissfft PRIVATE -O3)

add_executable(otfft_bench bench.cpp)
target_include_directories(otfft_bench SYSTEM PRIVATE ${PROJECT_SOURCE_DIR}/third_party/pocketfft)
target_compile_features(otfft_bench PRIVATE cxx_std_17)
target_link_libraries(otfft_bench otfft kissfft)
SetStandardLinkLibrariesForTest(otfft_bench)
SetPlatformSpecificOptimizationFlagsForTarget(otfft_bench)

if(OpenMP_CXX_FOUND)
  target_link_libraries(otfft_bench OpenMP::OpenMP_CXX)
endif()

if(FFTW_FOUND)
  message(STATUS "Benchmarks: FFTW ${FFTW_VERSION} found")
  target_compile_definitions(otfft_bench PRIVATE HAVE_FFTW)
  target_link_libraries(otfft_bench PkgConfig::FFTW)
  # the OpenMP part of FFTW has no pkg-config file of its own
  find_library(FFTW_OMP_LIBRARY NAMES fftw3_omp HINTS ${FFTW_LIBRARY_DIRS} ${FFTW_LIBDIR})
  if(FFTW_OMP_LIBRARY)
    message(STATUS "Benchmarks: FFTW OpenMP threads: ${FFTW_OMP_LIBRARY}")
    target_compile_definitions(otfft_bench PRIVATE HAVE_FFTW_OMP)
    target_link_libraries(otfft_bench ${FFTW_OMP_LIBRARY})
  else()
    message(STATUS "Benchmarks: libfftw3_omp not found, FFTW runs single-threaded")
  endif()
else()
  message(STATUS "Benchmarks: FFTW not found, it will be skipped")
endif()
