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

#
# Allow to specialize settings for the unit test projectst
if(WIN32)

  #
  # default use static linking of the boost test
  macro(SetBoostUnitTestFlags TEST_NAME)
    if(3RDPARTY_LINK_SHARED)
      set_property(TARGET ${TEST_NAME}
        APPEND PROPERTY COMPILE_DEFINITIONS
        BOOST_TEST_DYN_LINK
        )
    endif()  
  endmacro()
endif()

if(UNIX)

  if(USE_SYSTEM_BOOST_LIBS)
    #
    # use dynamic linking of the boost test
    macro(SetBoostUnitTestFlags TEST_NAME)
      set_property(TARGET ${TEST_NAME}
        APPEND PROPERTY COMPILE_DEFINITIONS
        BOOST_TEST_DYN_LINK
        )
    endmacro()

  else()
    #
    # default use static linking of the boost test
    macro(SetBoostUnitTestFlags TEST_NAME)
      if(3RDPARTY_LINK_SHARED)
        set_property(TARGET ${TEST_NAME}
          APPEND PROPERTY COMPILE_DEFINITIONS
          BOOST_TEST_DYN_LINK
          )
      endif()  
    endmacro()
  endif()

endif()
