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

# Optional second executable that benchmarks the upstream OTFFT sources
# (for example otfft-11.6e/otfft from http://wwwa.pikara.ne.jp/okojisan/otfft-en/download.html)
# instead of this repository's OTFFT. It uses the same harness, the same
# method and the same other libraries, so run it next to otfft_bench.
set(OTFFT_UPSTREAM_DIR "" CACHE PATH "Directory with the upstream OTFFT sources (otfft.cpp, otfft.h, ...)")
if(OTFFT_UPSTREAM_DIR)
  message(STATUS "Benchmarks: upstream OTFFT in ${OTFFT_UPSTREAM_DIR}")
  add_library(otfft_upstream STATIC ${OTFFT_UPSTREAM_DIR}/otfft.cpp)
  target_include_directories(otfft_upstream PUBLIC ${OTFFT_UPSTREAM_DIR})
  target_compile_options(otfft_upstream PRIVATE -O3 -march=native -fopenmp -fPIC)
  target_compile_features(otfft_upstream PRIVATE cxx_std_17)

  add_executable(otfft_bench_upstream bench.cpp)
  target_compile_definitions(otfft_bench_upstream PRIVATE OTFFT_UPSTREAM)
  target_include_directories(otfft_bench_upstream BEFORE PRIVATE ${OTFFT_UPSTREAM_DIR})
  target_include_directories(otfft_bench_upstream SYSTEM PRIVATE ${PROJECT_SOURCE_DIR}/third_party/pocketfft)
  target_compile_options(otfft_bench_upstream PRIVATE -O3 -march=native -fopenmp)
  target_compile_features(otfft_bench_upstream PRIVATE cxx_std_17)
  target_link_libraries(otfft_bench_upstream otfft_upstream kissfft -fopenmp)
  if(FFTW_FOUND)
    target_compile_definitions(otfft_bench_upstream PRIVATE HAVE_FFTW)
    target_link_libraries(otfft_bench_upstream PkgConfig::FFTW)
    if(FFTW_OMP_LIBRARY)
      target_compile_definitions(otfft_bench_upstream PRIVATE HAVE_FFTW_OMP)
      target_link_libraries(otfft_bench_upstream ${FFTW_OMP_LIBRARY})
    endif()
  endif()
endif()
