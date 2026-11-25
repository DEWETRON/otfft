#
# otfft unit tests

set(TEST_NAME otfft.${UNIT_TEST_SUFFIX})

set(UNIT_TEST_HEADERS
  otfft_test_checks.h
  otfft_test_utils.h
  otfft_class_check.h
  otfft_dft_check.h
)

set(UNIT_TEST_SOURCES
  test_module.cpp
  otfft_algorithm_table_test.cpp
  otfft_avx512_test.cpp
  otfft_backend_test.cpp
  otfft_basic_test.cpp
  otfft_bluestein_test.cpp
  otfft_dct_test.cpp
  otfft_exp_test.cpp
  otfft_mixed_avx_test.cpp
  otfft_omp_threshold_test.cpp
  otfft_placement_test.cpp
  otfft_find_peak_test.cpp
  otfft_optimization_test.cpp
  otfft_prime_test.cpp
  otfft_radix35_test.cpp
  otfft_radix7_test.cpp
  otfft_sixstep_test.cpp
  otfft_transform_test.cpp
  otfft_tuning_test.cpp
)

add_executable(${TEST_NAME}
  ${UNIT_TEST_HEADERS}
  ${UNIT_TEST_SOURCES}
)

target_include_directories(${TEST_NAME} SYSTEM PRIVATE
  ${PROJECT_SOURCE_DIR}/third_party
)

# otfft_tuning_test.cpp reads the internal lookup of the loaded tuning (D22)
target_include_directories(${TEST_NAME} PRIVATE
  ${PROJECT_SOURCE_DIR}/src
)

target_link_libraries(${TEST_NAME}
  otfft
)

SetStandardLinkLibrariesForTest(${TEST_NAME})
SetPlatformSpecificOptimizationFlagsForTarget(${TEST_NAME})

#
# add this to Visual Studio group UnitTests
set_target_properties(${TEST_NAME} PROPERTIES FOLDER "3rd_party/unit_tests")

add_test(NAME ${TEST_NAME}
  COMMAND ${TEST_NAME}
)
