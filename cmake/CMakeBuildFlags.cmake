#
# Settings influencing CMake and the build environment
#

if(my_module_BuildFlags_included)
  return()
endif(my_module_BuildFlags_included)
set(my_module_BuildFlags_included true)

# General settings for Visual Studio
if(MSVC)

  # speedup build time: /MP flag
  if (MSVC_BUILD_USING_MP)
    set(CMAKE_CXX_FLAGS                "${CMAKE_CXX_FLAGS} /MP")
    set(CMAKE_C_FLAGS                  "${CMAKE_C_FLAGS} /MP")
  endif()

  #
  # workaround for intel thread inspector
  # segfaults without it in boost/thread/once.hpp
  if(INTEL_INSPECTOR_WORKAROUND)
    set(_INLINE_DEBUG          /Ob1)
    set(_INLINE_MINSIZEREL     /Ob1)
    set(_INLINE_RELEASE        /Ob2)
    set(_INLINE_RELWITHDEBINFO /Ob1)
  else()
    set(_INLINE_DEBUG          /Ob0)
    set(_INLINE_MINSIZEREL     /Ob1)
    set(_INLINE_RELEASE        /Ob2)
    set(_INLINE_RELWITHDEBINFO /Ob1)
  endif()

  if (CMAKE_CXX_FLAGS_RELWITHDEBINFO)
    string(REPLACE "/D NDEBUG" "" CMAKE_CXX_FLAGS_RELWITHDEBINFO ${CMAKE_CXX_FLAGS_RELWITHDEBINFO})
    string(REPLACE "/DNDEBUG" "" CMAKE_CXX_FLAGS_RELWITHDEBINFO ${CMAKE_CXX_FLAGS_RELWITHDEBINFO})
    string(REPLACE "/O2" ""  CMAKE_CXX_FLAGS_RELWITHDEBINFO ${CMAKE_CXX_FLAGS_RELWITHDEBINFO})
    string(REPLACE "/Ob1" ""  CMAKE_CXX_FLAGS_RELWITHDEBINFO ${CMAKE_CXX_FLAGS_RELWITHDEBINFO})
  endif()
  if (CMAKE_C_FLAGS_RELWITHDEBINFO)
    string(REPLACE "/D NDEBUG" "" CMAKE_C_FLAGS_RELWITHDEBINFO ${CMAKE_C_FLAGS_RELWITHDEBINFO})
    string(REPLACE "/DNDEBUG" "" CMAKE_C_FLAGS_RELWITHDEBINFO ${CMAKE_C_FLAGS_RELWITHDEBINFO})
    string(REPLACE "/O2" ""  CMAKE_C_FLAGS_RELWITHDEBINFO ${CMAKE_C_FLAGS_RELWITHDEBINFO})
    string(REPLACE "/Ob1" ""  CMAKE_C_FLAGS_RELWITHDEBINFO ${CMAKE_C_FLAGS_RELWITHDEBINFO})
  endif()

  set(CMAKE_CXX_FLAGS_DEBUG          "${CMAKE_CXX_FLAGS_DEBUG} /D_DEBUG /Zi ${_INLINE_DEBUG} /Od /RTC1")
  set(CMAKE_CXX_FLAGS_MINSIZEREL     "${CMAKE_CXX_FLAGS_MINSIZEREL} /O1 ${_INLINE_MINSIZEREL} /D NDEBUG")
  set(CMAKE_CXX_FLAGS_RELEASE        "${CMAKE_CXX_FLAGS_RELEASE} /O2 ${_INLINE_RELEASE} /Zi /D NDEBUG")
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} /Zi /Od /Oy- ${_INLINE_DEBUG}")

  set(CMAKE_C_FLAGS_DEBUG            "${CMAKE_C_FLAGS_DEBUG} /D_DEBUG /Zi ${_INLINE_DEBUG} /Od /RTC1")
  set(CMAKE_C_FLAGS_MINSIZEREL       "${CMAKE_C_FLAGS_MINSIZEREL} /O1 ${_INLINE_MINSIZEREL} /D NDEBUG")
  set(CMAKE_C_FLAGS_RELEASE          "${CMAKE_C_FLAGS_RELEASE} /O2 ${_INLINE_RELEASE} /Zi /D NDEBUG")
  set(CMAKE_C_FLAGS_RELWITHDEBINFO   "${CMAKE_C_FLAGS_RELWITHDEBINFO} /Zi /Od /Oy- ${_INLINE_DEBUG}")

  if (MSVC12)
    # enable /Zo (Enhance Optimized Debugging)
    set(CMAKE_CXX_FLAGS_RELEASE        "${CMAKE_CXX_FLAGS_RELEASE} /Zo")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} /Zo")
    set(CMAKE_C_FLAGS_RELEASE          "${CMAKE_C_FLAGS_RELEASE} /Zo")
    set(CMAKE_C_FLAGS_RELWITHDEBINFO   "${CMAKE_C_FLAGS_RELWITHDEBINFO} /Zo")
  endif()

  set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /INCREMENTAL:NO /DEBUG /OPT:REF")
  set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} /INCREMENTAL:NO /DEBUG /OPT:REF")

  # set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}   /D _CRTDBG_MAP_ALLOC")
  # set(CMAKE_C_FLAGS_DEBUG   "${CMAKE_C_FLAGS_DEBUG}     /D _CRTDBG_MAP_ALLOC")

  #
  # Disable warning 4786 4250 4503 to silence Arabica
  # Disable warning 4351 4373
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4786 /wd4250 /wd4503 /wd4351 /wd4373")

  #
  # Disable min/max macros
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /DNOMINMAX")

  if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /DBUILD_X64")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /DBUILD_X64")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /DBUILD_X86")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /DBUILD_X86")
  endif()

endif()

# Settings for GCC (UNIX)
if(UNIX)

  # set UNIX flag
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUNIX")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DUNIX")

  if(APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DOSX")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DOSX")
  endif()

  include(CheckCXX11Features)

  if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-multichar ${CXX11_COMPILER_FLAGS} -Wno-unused-variable -Wno-unused-private-field")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wmultichar -Wno-unused-variable")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-multichar ${CXX11_COMPILER_FLAGS}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wmultichar")
  endif()

  if (CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-sign-compare -Wno-unused -Wno-unknown-pragmas -Wno-comment -Wno-parentheses -Wno-switch -Wno-strict-aliasing")
  endif()

  # -> enable 32bit building in gcc64
  # if(CMAKE_SIZEOF_VOID_P EQUAL 8)
  #   if(BUILD_X86)
  #     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m32")
  #     set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -m32")
  #   endif()
  # endif()


  #
  # Allow function pointers to void* assignments
  # if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  #   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpermissive")
  #   set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
  # endif()

  #
  # Position Independent Code
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")

  # set build id via linker
  # if (NOT HG_COMMIT)
  #   execute_process(COMMAND hg id -i
  #     OUTPUT_VARIABLE HG_COMMIT_DATA
  #     WORKING_DIRECTORY ${SW_APP_ROOT}
  #     OUTPUT_STRIP_TRAILING_WHITESPACE)
  #   string(REGEX MATCH "[0-9a-f]+" HG_COMMIT ${HG_COMMIT_DATA})
  # endif()
  #set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--build-id=0x${HG_COMMIT}")
  #set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--build-id=0x${HG_COMMIT}")

  option(USE_GOLD_LINKER "speed up build by using GNU gold linker")
  if(USE_GOLD_LINKER)
    # check if gold linker is available and use it
    if (NOT LD_VERSION)
      execute_process(
          COMMAND ${CMAKE_CXX_COMPILER} -fuse-ld=gold -Wl,--version
          ERROR_QUIET OUTPUT_VARIABLE LD_VERSION)
    endif()
    if("${LD_VERSION}" MATCHES "GNU gold")
      set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-fuse-ld=gold")
      set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-fuse-ld=gold")
      message(STATUS "GNU gold linker will be used.")

      option(USE_GOLD_LINKER_THREADS "speed up build by using multithreaded linking")
      if(USE_GOLD_LINKER_THREADS)
        # enable multithreaded linking
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--threads")
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--threads")
      endif()
    endif()
  endif()

  option(USE_LLD_LINKER "speed up build even more by using LLVM's lld linker")
  if(USE_LLD_LINKER)
    # check if LLD linker is available and use it
    if (NOT LD_VERSION)
      execute_process(
          COMMAND ${CMAKE_CXX_COMPILER} -fuse-ld=lld -Wl,--version
         ERROR_QUIET OUTPUT_VARIABLE LD_VERSION)
    endif()
    if("${LD_VERSION}" MATCHES "LLD")
      set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-fuse-ld=lld")
      set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-fuse-ld=lld")
      message(STATUS "LLVM's lld linker will be used.")
    endif()
  endif()

  if(USE_LLD_LINKER AND USE_GOLD_LINKER)
    message(FATAL_ERROR "Cannot use two different linkers at the same time")
  endif()

  if(UNIX)
    #
    # Convert RPATH setting to RUNPATH
    SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--enable-new-dtags")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--enable-new-dtags")

    
    #
    # Hide non public symbols
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden -fvisibility-inlines-hidden")
    set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -fvisibility=hidden")      

  endif()
  

  #
  # Sometimes it is necessary to have debug info in release
  #set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -g")

  #
  # OSX Clang sets small limits for compiler recursions
  if (APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftemplate-depth=256 -Wno-error-unused-variable -Wno-error-unused-parameter")
  endif()

  #
  # x38 specifics
  if("${BUILD_ARCH}" STREQUAL "x38")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBUILD_X86 -static -m32 -mtune=i586 -mfpmath=387")
    set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS} -DBUILD_X86 -static -m32 -mtune=i586 -mfpmath=387")
  elseif("${BUILD_ARCH}" STREQUAL "pi")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static -O3")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -static -O3")
  else()
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBUILD_X64")
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DBUILD_X64")
    else()
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBUILD_X86")
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DBUILD_X86")
    endif()
  endif()
endif()

if(BUILD_ARCH)
  # so cmake does not complain
endif()


if (APPLE)
  set(CMAKE_MACOSX_RPATH 1)
endif()

#
# Enable target folders in IDEs
set_property(GLOBAL
  PROPERTY USE_FOLDERS ON)


#
# This function enables the following platform specific flags:
#   - Vectorization (SSE, SSE2, SSE3, SSE4)
#   - OpenMP
#   - Use intrinsics where applicable instead of FPU
function(SetPlatformSpecificOptimizationFlagsForTarget TARGET_NAME)
  set(VERBOSE ${ARGC} GREATER 1)

  #
  # compiler specific optimization flags

  if (VERBOSE)
    message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID}")
  endif()

  # Treat GNU Compiler and CLang Compiler the same
  if(CMAKE_COMPILER_IS_GNUCXX OR ${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
    if (VERBOSE)
      message(STATUS "Optimize for GNU-like compiler")
    endif()
    set_property(TARGET ${TARGET_NAME} APPEND_STRING PROPERTY
      COMPILE_FLAGS
      " -msse -msse2 -msse3 -mssse3 -msse4 -ffp-contract=fast -fpermissive ${OpenMP_CXX_FLAGS} "
    )
    if(${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
      set_property(TARGET ${TARGET_NAME} APPEND_STRING PROPERTY
        COMPILE_FLAGS
        " -fvectorize "
      )
    else()
      set_property(TARGET ${TARGET_NAME} APPEND_STRING PROPERTY
        COMPILE_FLAGS
        " -fext-numeric-literals "
        )
    endif()
  # Intel C++ Compiler
  elseif(${CMAKE_CXX_COMPILER_ID} MATCHES "Intel")
    if (VERBOSE)
      message(STATUS "Optimize for Intel compiler")
    endif()
    set_property(TARGET ${TARGET_NAME} APPEND_STRING PROPERTY
      COMPILE_FLAGS
      " -w2 -msse2 -msse3 -mssse3 -msse4 ${OpenMP_CXX_FLAGS} "
    )
  # Microsoft Visual C++ Compiler
  elseif(MSVC OR ${CMAKE_CXX_COMPILER_ID} MATCHES "MSVC")
    if (VERBOSE)
      message(STATUS "Optimize for MSVC compiler")
    endif()
    if(CMAKE_CL_64)
      set_property(TARGET ${TARGET_NAME} APPEND_STRING PROPERTY
        COMPILE_FLAGS
        " /Oi /favor:INTEL64 ${OpenMP_CXX_FLAGS} "
      )
    else()
      # Enable SSE2 for 32bit target
      set_property(TARGET ${TARGET_NAME} APPEND_STRING PROPERTY
        COMPILE_FLAGS
        " /Oi /favor:blend /arch:SSE2 ${OpenMP_CXX_FLAGS} "
      )
    endif()
  endif()
endfunction()
