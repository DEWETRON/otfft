#
# otfft unit tests

set(TEST_NAME otfft.${UNIT_TEST_SUFFIX})

find_package(Boost
    REQUIRED COMPONENTS
    unit_test_framework
)

include_directories(
  SYSTEM
  ${Boost_INCLUDE_DIRS}
)

set(UNIT_TEST_HEADERS
  otfft_test_utils.h
)

set(UNIT_TEST_SOURCES
  test_module.cpp
  otfft_basic_test.cpp
  otfft_bluestein_test.cpp
  otfft_dct_test.cpp
  otfft_find_peak_test.cpp
  otfft_optimization_test.cpp
  otfft_transform_test.cpp
)

add_definitions(-DBOOST_SPIRIT_THREADSAFE)

add_executable(${TEST_NAME}
  ${UNIT_TEST_HEADERS}
  ${UNIT_TEST_SOURCES}
)

SetBoostUnitTestFlags(${TEST_NAME})

target_link_libraries(${TEST_NAME}
  ${Boost_LIBRARIES}
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
