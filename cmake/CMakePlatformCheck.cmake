#
# Checking for OS and Platform
#

if(my_module_PlatformCheck_included)
  return()
endif(my_module_PlatformCheck_included)
set(my_module_PlatformCheck_included true)

#
# Check for 64 bit build
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
  set(BUILD_X64 TRUE)
  set(BUILD_X86 FALSE)
else()
  set(BUILD_X64 FALSE)
  set(BUILD_X86 TRUE)
endif()

if(NOT DEFINED 3RDPARTY_LINK_TYPE)
  set(3RDPARTY_LINK_TYPE "shared")
endif()

if(3RDPARTY_LINK_TYPE STREQUAL "shared")
  set(3RDPARTY_LINK_SHARED TRUE)
  set(3RDPARTY_LINK_SHAREDRT TRUE)
elseif(3RDPARTY_LINK_TYPE STREQUAL "shared_staticrt")
  set(3RDPARTY_LINK_SHARED TRUE)
  set(3RDPARTY_LINK_SHAREDRT FALSE)
elseif(3RDPARTY_LINK_TYPE STREQUAL "static")
  set(3RDPARTY_LINK_SHARED FALSE)
  set(3RDPARTY_LINK_SHAREDRT FALSE)
elseif(3RDPARTY_LINK_TYPE STREQUAL "static_sharedrt")
  set(3RDPARTY_LINK_SHARED FALSE)
  set(3RDPARTY_LINK_SHAREDRT TRUE)
endif()


#
# BUILD Settings can be overruled
# if(UNIX)
#   if(${BUILD_ARCH} X64)
#     set(BUILD_X64 FALSE)
#     set(BUILD_X86 TRUE)
#   else()
#     set(BUILD_X64 FALSE)
#     set(BUILD_X86 TRUE)
#   endif()
# endif()

if (CMAKE_SCRIPT_DEBUG)
  message(STATUS "BUILD_X64 = ${BUILD_X64}")
  message(STATUS "BUILD_X86 = ${BUILD_X86}")

  message(STATUS "3RDPARTY_LINK_TYPE = ${3RDPARTY_LINK_TYPE}")
  message(STATUS "3RDPARTY_LINK_SHARED = ${3RDPARTY_LINK_SHARED}")
  message(STATUS "3RDPARTY_LINK_SHAREDRT = ${3RDPARTY_LINK_SHAREDRT}")
endif()
