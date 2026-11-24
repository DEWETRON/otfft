#
# otfft examples (OTFFT_BUILD_EXAMPLES): small programs that use the public API.
# Each one checks its own result and returns 0 on success; ctest runs them.

set(OTFFT_EXAMPLES complex_fft real_fft dct bluestein)

foreach(example ${OTFFT_EXAMPLES})
  add_executable(example_${example} ${example}.cpp)
  target_link_libraries(example_${example} otfft)
  SetStandardLinkLibrariesForTest(example_${example})
  SetPlatformSpecificOptimizationFlagsForTarget(example_${example})
  set_target_properties(example_${example} PROPERTIES FOLDER "3rd_party/examples")
  add_test(NAME example.${example} COMMAND example_${example})
  set_tests_properties(example.${example} PROPERTIES PASS_REGULAR_EXPRESSION "${example}: ok")
endforeach()
