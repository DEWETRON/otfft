#
# Special settings for unit tests
#

if(my_module_UnitTestSettings_included)
  return()
endif(my_module_UnitTestSettings_included)
set(my_module_UnitTestSettings_included true)

#
# Add general suffix to all unit tests names
set(UNIT_TEST_SUFFIX test)
