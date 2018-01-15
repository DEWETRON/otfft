#
# Definitions of variables
#

if(my_module_CommonVars_included)
  return()
endif(my_module_CommonVars_included)
set(my_module_CommonVars_included true)

# guess SW_APP_ROOT
if (NOT DEFINED SW_APP_ROOT)
  get_filename_component(PROJECT_ROOT ../.. ABSOLUTE)
endif ()

if (NOT DEFINED LIB_ROOT)
  get_filename_component(LIB_ROOT ${PROJECT_ROOT}/lib ABSOLUTE)
endif ()

if (NOT DEFINED API_ROOT)
  get_filename_component(API_ROOT ${PROJECT_ROOT}/api ABSOLUTE)
endif ()

#
# old 3rdparty
if (NOT DEFINED 3RDPARTY_ROOT)
  get_filename_component(3RDPARTY_ROOT ${PROJECT_ROOT}/3rdparty ABSOLUTE)
else()
  get_filename_component(3RDPARTY_ROOT ${3RDPARTY_ROOT} ABSOLUTE)
endif ()

#
# new 3rdparty
if (NOT DEFINED OPT_ROOT)
  if (EXISTS /opt)
    set(OPT_ROOT "/opt")
  else()
    get_filename_component(OPT_ROOT ${PROJECT_ROOT}/opt ABSOLUTE)
  endif()
else()
  get_filename_component(OPT_ROOT ${OPT_ROOT} ABSOLUTE)
endif ()



if (NOT DEFINED WIN_CPP_RUNTIME)
  get_filename_component(WIN_CPP_RUNTIME ${PROJECT_ROOT}/packaging/Win/bin/CPPRuntime ABSOLUTE)
endif ()

# determine build type: release, debug
if(NOT CMAKE_BUILD_TYPE)
  # default build type is debug
  set(CMAKE_BUILD_TYPE Debug)
endif()

if(CMAKE_BUILD_TYPE STREQUAL None)
  set(BIN_STAGE_INTDIR Debug)
elseif(CMAKE_BUILD_TYPE STREQUAL Debug)
  set(BIN_STAGE_INTDIR Debug)
elseif(CMAKE_BUILD_TYPE STREQUAL Release)
  set(BIN_STAGE_INTDIR Release)
elseif(CMAKE_BUILD_TYPE STREQUAL RelWithDebInfo)
  set(BIN_STAGE_INTDIR RelWithDebInfo)
elseif(CMAKE_BUILD_TYPE STREQUAL MinSizeRel)
  set(BIN_STAGE_INTDIR MinSizeRel)
endif()

if (CMAKE_SCRIPT_DEBUG)
  message("CMAKE_BUILD_TYPE = ${CMAKE_BUILD_TYPE}")
  message("BIN_STAGE_INTDIR = ${BIN_STAGE_INTDIR}")
endif()
