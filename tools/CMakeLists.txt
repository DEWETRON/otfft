#
# otfft tools (OTFFT_BUILD_TOOLS=ON): ffttune measures which algorithm is fastest
# for each power-of-two length on this machine.
#
# The measuring code is compiled once per backend with the same flags as the
# library's algorithm sources, so it times the classes the library really uses.

set(FFTTUNE_BACKEND_OBJECTS "")
set(FFTTUNE_DEFINITIONS "")

macro(fft_tune_backend NAME UPPER FLAGS)
  add_library(ffttune_${NAME} OBJECT ffttune_backend.cpp ffttune.h)
  target_compile_definitions(ffttune_${NAME} PRIVATE FFTTUNE_MEASURE=ffttune_measure_${NAME})
  set_property(TARGET ffttune_${NAME} APPEND_STRING PROPERTY COMPILE_FLAGS " ${FLAGS} ")
  list(APPEND FFTTUNE_BACKEND_OBJECTS $<TARGET_OBJECTS:ffttune_${NAME}>)
  list(APPEND FFTTUNE_DEFINITIONS FFTTUNE_HAVE_${UPPER})
endmacro()

if(OTFFT_WITH_SSE2)
  fft_tune_backend(sse2 SSE2 "${SSE2_FLAGS}")
endif()
if(OTFFT_WITH_AVX)
  fft_tune_backend(avx AVX "${AVX_FLAGS}")
endif()
if(OTFFT_WITH_AVX2)
  fft_tune_backend(avx2 AVX2 "${AVX2_FLAGS}")
endif()

# ffttune.cpp is C++17; the backend objects use the library's flags (they include
# library headers), see ffttune_backend.cpp.
add_executable(ffttune ffttune.cpp ffttune.h ${FFTTUNE_BACKEND_OBJECTS})
target_compile_definitions(ffttune PRIVATE ${FFTTUNE_DEFINITIONS})
target_compile_features(ffttune PRIVATE cxx_std_17)
target_link_libraries(ffttune otfft)
SetStandardLinkLibrariesForTest(ffttune)
SetPlatformSpecificOptimizationFlagsForTarget(ffttune)
if(OpenMP_CXX_FOUND)
  target_link_libraries(ffttune OpenMP::OpenMP_CXX)
endif()

# smoke test: a short run must finish and print a switch line for the last length
add_test(NAME ffttune.smoke
  COMMAND ffttune --min-log 4 --max-log 12 --repeats 3 --batch-ms 2)
set_tests_properties(ffttune.smoke PROPERTIES PASS_REGULAR_EXPRESSION "case 12: obj = FFT[1-7]::instance")

# the tuning file that --out writes must be accepted by OTFFT::loadTuningFile (D23); ffttune
# checks this itself and fails if the library rejects the file. The second run uses the
# threaded table.
add_test(NAME ffttune.tuning_file
  COMMAND ffttune --min-log 4 --max-log 12 --repeats 3 --batch-ms 2 --out ffttune_test_tuning.txt)
set_tests_properties(ffttune.tuning_file PROPERTIES PASS_REGULAR_EXPRESSION "and the library accepts it")
add_test(NAME ffttune.tuning_file_threaded
  COMMAND ffttune --min-log 4 --max-log 12 --repeats 3 --batch-ms 2 --threads 2 --out ffttune_test_tuning_threaded.txt)
set_tests_properties(ffttune.tuning_file_threaded PROPERTIES PASS_REGULAR_EXPRESSION "\\(threaded table\\) and the library accepts it")
