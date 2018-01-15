#
# Set of custom made practical functions for cmake
#

if(my_module_CmakeFunctions_included)
  return()
endif(my_module_CmakeFunctions_included)
set(my_module_CmakeFunctions_included true)

#
# Debug or trace messages cabe activated with with:
# set(CMAKE_SCRIPT_DEBUG ON)
# Just set it in a CMakeLists.txt file before calling a function or macro!




function(set_library_export_flag TARGETNAME)

  if(WIN32)
    #get type of target
    get_property(_target_type
      TARGET ${TARGETNAME}
      PROPERTY TYPE
      )
    if(${ARGC} EQUAL 2)
      set(_lib_name ${ARGV1})
    else()
      set(_lib_name ${TARGETNAME})
    endif()

    # only add xxxx_LIB for static libs
    # to remove declspec specifiers
    if(_target_type STREQUAL STATIC_LIBRARY)
      message("Setting ${_lib_name}_LIB for ${TARGETNAME}")
      set_property(TARGET ${TARGETNAME}
        APPEND PROPERTY COMPILE_DEFINITIONS
        ${_lib_name}_LIB
        )
    elseif(_target_type STREQUAL SHARED_LIBRARY)
      if(NOT _lib_name STREQUAL ${TARGETNAME})
        message("Setting ${_lib_name}_LIB} for ${TARGETNAME}")
        set_property(TARGET ${TARGETNAME}
          APPEND PROPERTY COMPILE_DEFINITIONS
          ${_lib_name}_LIB
          )
      endif()
    endif()

    # nothing to do for shared libs
    # their export flag is handled by cmake
    endif(WIN32)
endfunction(set_library_export_flag)


#
# Use this macro to set a common output directory for all artifacts
# of a build.
#
macro(SetCommonOutputDirectory)
  if(MSVC)
    if (NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY)
      set(CMAKE_LIBRARY_OUTPUT_DIRECTORY    ${CMAKE_CURRENT_BINARY_DIR})
    endif()
    if (NOT CMAKE_ARCHIVE_OUTPUT_DIRECTORY)
      set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY    ${CMAKE_CURRENT_BINARY_DIR})
    endif()
    if (NOT CMAKE_RUNTIME_OUTPUT_DIRECTORY)
      set(CMAKE_RUNTIME_OUTPUT_DIRECTORY    ${CMAKE_CURRENT_BINARY_DIR})
    endif()
  elseif(APPLE)
    if (XCODE_VERSION)
      if (NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY)
        set(CMAKE_LIBRARY_OUTPUT_DIRECTORY    ${CMAKE_CURRENT_BINARY_DIR})
      endif()
      if (NOT CMAKE_ARCHIVE_OUTPUT_DIRECTORY)
        set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY    ${CMAKE_CURRENT_BINARY_DIR})
      endif()
      if (NOT CMAKE_RUNTIME_OUTPUT_DIRECTORY)
        set(CMAKE_RUNTIME_OUTPUT_DIRECTORY    ${CMAKE_CURRENT_BINARY_DIR})
      endif()
    else()
      if (NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY)
        set(CMAKE_LIBRARY_OUTPUT_DIRECTORY    ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE})
      endif()
      if (NOT CMAKE_ARCHIVE_OUTPUT_DIRECTORY)
        set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY    ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE})
      endif()
      if (NOT CMAKE_RUNTIME_OUTPUT_DIRECTORY)
        set(CMAKE_RUNTIME_OUTPUT_DIRECTORY    ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE})
      endif()
    endif()
  else()
    if (NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY)
      set(CMAKE_LIBRARY_OUTPUT_DIRECTORY    ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE})
    endif()
    if (NOT CMAKE_ARCHIVE_OUTPUT_DIRECTORY)
      set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY    ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE})
    endif()
    if (NOT CMAKE_RUNTIME_OUTPUT_DIRECTORY)
      set(CMAKE_RUNTIME_OUTPUT_DIRECTORY    ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE})
    endif()
  endif()

endmacro()

#
# Set the boost link options
# Windows: static build and static runtime
# Linux: dynamic libs(from OS) and dynamic runtime
macro(SetBoostOptions)
  if (WIN32)
    if(3RDPARTY_LINK_SHARED)
      set(Boost_USE_STATIC_LIBS    OFF)
    else()
      set(Boost_USE_STATIC_LIBS    ON)
    endif()

    set(Boost_USE_MULTITHREADED  ON)

    if(3RDPARTY_LINK_SHAREDRT)
      set(Boost_USE_STATIC_RUNTIME OFF)
    else()
      set(Boost_USE_STATIC_RUNTIME ON)
    endif()
  endif()


  if (UNIX)

    # Handle macosx -> tell the compiler used
    if (APPLE)
      if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        set(Boost_COMPILER "-xgcc42")
      endif()
    else()
      if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        if (NOT DEFINED Boost_COMPILER)
          #set(Boost_COMPILER "-gcc49")
          # want to determine the version of the installed gcc/g++
          # to be able to use the same "versioned" libs with clang
          # limit to major.minor => 4.9.2 --> 4.9
          execute_process(
            COMMAND g++ -dumpversion
            OUTPUT_VARIABLE GCC_VERSION
            )
          string(REGEX REPLACE "\\." "" GCC_VERSION ${GCC_VERSION})
          string(LENGTH ${GCC_VERSION} GCC_VERSION_LEN)
          if (${GCC_VERSION_LEN} GREATER 2)
            string(SUBSTRING ${GCC_VERSION} 0 2 GCC_VERSION)
          endif()
          set(Boost_COMPILER "-gcc${GCC_VERSION}")
        endif()
      endif()
    endif()

    # We may depend on boost of the current linux distribution
    if(USE_SYSTEM_BOOST_LIBS)
      set(Boost_USE_STATIC_LIBS    OFF)
      set(Boost_USE_MULTITHREADED  ON)
      set(Boost_USE_STATIC_RUNTIME OFF)
    else()
      # Or we use our own boost:
      if(3RDPARTY_LINK_SHARED)
        set(Boost_USE_STATIC_LIBS    OFF)
      else()
        set(Boost_USE_STATIC_LIBS    ON)
      endif()

      set(Boost_USE_MULTITHREADED  ON)

      #
      # ignore static runtime for now
      # does not build well in some settings

      # if(3RDPARTY_LINK_SHAREDRT)

      # set(Boost_USE_STATIC_RUNTIME OFF)
      # else()
      #   set(Boost_USE_STATIC_RUNTIME ON)
      # endif()

      set(Boost_USE_STATIC_RUNTIME OFF)

    endif()
  endif()

  if (CMAKE_SCRIPT_DEBUG)
      message(STATUS "BOOST_ROOT = ${BOOST_ROOT}")
      message(STATUS "Boost_USE_STATIC_LIBS    = ${Boost_USE_STATIC_LIBS}")
      message(STATUS "Boost_USE_MULTITHREADED  = ${Boost_USE_MULTITHREADED}")
      message(STATUS "Boost_USE_STATIC_RUNTIME = ${Boost_USE_STATIC_RUNTIME}")
  endif()
endmacro()


macro(SetUnilibSpiritStringConverter LIBNAME)

  # for debugging the "fast" string converter
  # set_property(TARGET ${LIBNAME}
  #   APPEND
  #   PROPERTY COMPILE_DEFINITIONS_DEBUG
  #   USE_SPIRIT_OPTIMIZED_STRING_CONVERTER
  #   )
  # for debugging end

  set_property(TARGET ${LIBNAME}
    APPEND
    PROPERTY COMPILE_DEFINITIONS $<$<CONFIG:Release>:USE_SPIRIT_OPTIMIZED_STRING_CONVERTER>
    )

  set_property(TARGET ${LIBNAME}
    APPEND
    PROPERTY COMPILE_DEFINITIONS $<$<CONFIG:MINSIZEREL>:USE_SPIRIT_OPTIMIZED_STRING_CONVERTER>
    )

  #relWithDebInfo is now built without optimization => use normal string converter
  #set_property(TARGET ${LIBNAME}
  #  APPEND
  #  PROPERTY COMPILE_DEFINITIONS $<$<CONFIG:RELWITHDEBINFO>:USE_SPIRIT_OPTIMIZED_STRING_CONVERTER>
  #  )

endmacro()

#
# This macro adds a build step for the generation of the VersionInfo files.
#macro(AddVersionInfoBuildStep SW_ROOT)
#  add_custom_command(OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/version_info/VersionInfo.h
#    COMMAND python ${SW_ROOT}/build_util/BuildTool/hgwcrev.py --template_file=${CMAKE_CURRENT_SOURCE_DIR}/version_info/version_info.h_template --out_file=${CMAKE_CURRENT_SOURCE_DIR}/version_info/VersionInfo.h
#    COMMAND ${CMAKE_COMMAND} -E remove -f ${CMAKE_CURRENT_SOURCE_DIR}/version_info/computer_info.h
#    COMMAND ${CMAKE_COMMAND} -E echo "#define BUILDPC   ${COMPUTERNAME}" >> ${CMAKE_CURRENT_SOURCE_DIR}/version_info/computer_info.h
#    COMMAND ${CMAKE_COMMAND} -E echo "#define BUILDUSER ${USERNAME}" >> ${CMAKE_CURRENT_SOURCE_DIR}/version_info/computer_info.h
#  )
#endmacro()

#
# This macro adds a build step for the generation of the VersionInfo files.
macro(AddGenerateVersionInfoFile TPL_FILE VER_FILE SW_ROOT)

  if(EXISTS ${TPL_FILE})

    get_filename_component(OUT_DIR ${VER_FILE} PATH)

    add_custom_command(
	OUTPUT ${VER_FILE}
 	COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT_DIR}
 	COMMAND python ${SW_ROOT}/build_util/BuildTool/hgwcrev.py --template_file=${TPL_FILE} --out_file=${VER_FILE}
 	)

    # tell CMake that the file is generated (maybe it would scan the
    # file for dependencies otherwise), and that it is a header file
    # (for whatever reason)

    set_source_files_properties(
        ${VER_FILE}
 	    PROPERTIES
 	    GENERATED TRUE
 	    HEADER_FILE_ONLY TRUE)

else()
    message(FATAL_ERROR "${TPL_FILE} not found")
  endif()
endmacro()

macro(ExtractVersionInfo MAJOR_VERSION_HDR)
  if(EXISTS ${MAJOR_VERSION_HDR})
    file(STRINGS ${MAJOR_VERSION_HDR} CONTENTS)

    foreach(LINE ${CONTENTS})
      if(${LINE} MATCHES ".*#define[ ]*VERSION_[A-Z]*[ ]*[0-9]*")
        message("match = '${LINE}'")
        string(REGEX REPLACE ".*#define[ ]*VERSION_([A-Z]*)[ ]*[0-9]*" "\\1" VERSION_PART ${LINE})
        if(${VERSION_PART} STREQUAL "MAJOR")
          string(REGEX REPLACE ".*#define[ ]*VERSION_[A-Z]*[ ]*([0-9]*)" "\\1" VERSION_MAJOR ${LINE})
        elseif(${VERSION_PART} STREQUAL "MINOR")
          string(REGEX REPLACE ".*#define[ ]*VERSION_[A-Z]*[ ]*([0-9]*)" "\\1" VERSION_MINOR ${LINE})
        elseif(${VERSION_PART} STREQUAL "MICRO")
          string(REGEX REPLACE ".*#define[ ]*VERSION_[A-Z]*[ ]*([0-9]*)" "\\1" VERSION_MICRO ${LINE})
        endif()
      endif()
    endforeach()
  else()
    message(FATAL_ERROR "${MAJOR_VERSION_HDR} not found")
  endif()
endmacro()


#
# This macro adds a build step for version-info substitution based on last change of a specific file
macro(AddGenerateVersionInfoFileLastChange LIBNAME TPL_FILE VER_FILE SW_ROOT)

  if(EXISTS ${TPL_FILE})
    message(STATUS "${TPL_FILE} hgwcrev exact lookup")
    get_filename_component(OUT_DIR ${VER_FILE} PATH)

    add_custom_command(TARGET ${LIBNAME}
      POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT_DIR}
      COMMAND python ${SW_ROOT}/build_util/BuildTool/hgwcrev.py --template_file=${TPL_FILE} --out_file=${VER_FILE} --lookup_template_change
      DEPENDS ALL
      )
  else()
    message(FATAL_ERROR "${TPL_FILE} not found")
  endif()
endmacro()


macro(AddGenerateComputerInfoFile CMP_FILE SW_ROOT)
  get_filename_component(OUT_DIR ${CMP_FILE} PATH)
  add_custom_command(OUTPUT ${CMP_FILE}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT_DIR}
    COMMAND python ${SW_ROOT}/build_util/BuildTool/computer_info.py ${CMP_FILE}
  )

  # tell CMake that the file is generated (maybe it would scan the
  # file for dependencies otherwise), and that it is a header file
  # (for whatever reason)

  set_source_files_properties(
      ${VER_FILE}
      PROPERTIES
          GENERATED TRUE
 	  HEADER_FILE_ONLY TRUE)

endmacro()


#
# This macro adds a build step for lst file generation
# param LIBNAME is the target name
# param LST_NAME is the name of archive (zip) file to be generated
# param DEFINITION_FILE is the path to the spec file necessary for archive generation
# param DEFINITION_FILE_DEST is the dest path to the spec file necessary for archive generation
macro(AddListFileGeneration LIBNAME LST_NAME DEFINITION_FILE DEFINITION_FILE_DEST)

  if(EXISTS ${DEFINITION_FILE})
    set(_DEF_SRC_FILE ${DEFINITION_FILE})
    # also force dest in this case
    set(_DEF_DEST_FILE ${DEFINITION_FILE_DEST})
  else()
    set(_DEF_SRC_FILE ${CMAKE_SOURCE_DIR}/${DEFINITION_FILE})
    set(_DEF_DEST_FILE ${CMAKE_BINARY_DIR}/${DEFINITION_FILE_DEST})
  endif()
  #message(STATUS "---------------------------------------------")
  #message(STATUS "${LIBNAME} = _DEF_SRC_FILE  = ${_DEF_SRC_FILE}")
  #message(STATUS "${LIBNAME} = _DEF_DEST_FILE = ${_DEF_DEST_FILE}")
  #message(STATUS "---------------------------------------------")

  if(MSVC)
    if(${CMAKE_GENERATOR} STREQUAL "Ninja")
      set(LIST_FILE_FULL_PATH "${CMAKE_BINARY_DIR}/${LST_NAME}")
    else()
      set(LIST_FILE_FULL_PATH "${CMAKE_BINARY_DIR}/$(Configuration)/${LST_NAME}")
    endif()
  endif()
  if(UNIX)
    set(LIST_FILE_FULL_PATH "${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}/${LST_NAME}")
  endif()

  add_custom_command(TARGET ${LIBNAME}
    POST_BUILD
    #copy all necessary files from source to dest (noth the entire right place, but easier that way)
    COMMAND ${CMAKE_COMMAND} -E copy ${_DEF_SRC_FILE} ${_DEF_DEST_FILE}
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/board_listing/TDL ${CMAKE_BINARY_DIR}/DwTrionLst/TDL
    #Generate the DwTrion.lst file
    COMMAND python ${SW_API_DLL_ROOT}/build_util/BuildTool/generatelst.py ${LIST_FILE_FULL_PATH} ${_DEF_DEST_FILE}
  )
endmacro()

#
# Add a manual copy step after target build completed
# param LIBNAME is the target name
# param SRC_DIR relative path to a file/directory starting from CMAKE_SOURCE_DIR
# param DEST_DIR relative path to a file/directory starting from CMAKE_BINARY_DIR
macro(AddDirCopyStep LIBNAME SRC_DIR DEST_DIR)
  add_custom_command(TARGET ${LIBNAME}
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/${SRC_DIR} ${CMAKE_BINARY_DIR}/${DEST_DIR}
    )
endmacro()

macro(AddDirCopyStepFullDestPath LIBNAME SRC_DIR DEST_DIR)
  add_custom_command(TARGET ${LIBNAME}
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/${SRC_DIR} ${DEST_DIR}
    )
endmacro()



#
# Enable improved floating point precision
macro(SetFloatingPointPrecision LIBNAME)
  #Floating Point precision Switch
  #modify this section to
  #if(MSVC)
  #   try compile with /Op
  #   try to compile with /fp:precise
  #endif
  #pick the option, that does not throw a error/warning
  #issue an error, if none of them satisfy the above condition
  if(MSVC)
    #message(${CMAKE_CXX_COMPILER_ID})
    if(CMAKE_COMPILER_2005 OR MSVC90 OR MSVC10 OR MSVC11 OR MSVC12 OR MSVC80)
      set( FP_PRECISE /fp:precise)
    elseif( MSVC70 OR MSVC71 )
      set( FP_PRECISE /Op)
    else()
      message( FATAL_ERROR "Unable to determine compiler switch for improved floatingpoint-precision")
    endif()

    #improved floating point consistency
    set_property(TARGET ${LIBNAME}
      APPEND
      PROPERTY COMPILE_FLAGS
      ${FP_PRECISE}
      )
  endif()
endmacro()


#
# Set Debug/Release .. as define
macro(SetBuildConfigFlag LIBNAME)
  if(MSVC)
    if(${CMAKE_GENERATOR} STREQUAL "Ninja")
      set_property(TARGET ${LIBNAME}
        APPEND
        PROPERTY COMPILE_DEFINITIONS
        _BUILDCONFIG="${CMAKE_BUILD_TYPE}"
        _CRT_SECURE_NO_WARNINGS
      )
    else()
      set_property(TARGET ${LIBNAME}
        APPEND
        PROPERTY COMPILE_DEFINITIONS
        _BUILDCONFIG="$(Configuration)"
        _CRT_SECURE_NO_WARNINGS
      )
    endif()
  else()
    set_property(TARGET ${LIBNAME}
      APPEND
      PROPERTY COMPILE_DEFINITIONS
      _BUILDCONFIG="${CMAKE_BUILD_TYPE}"
      )
  endif()
endmacro()


#
# The binary should use static runtime
macro(SetLinkStaticRuntime)

  if (MSVC)
    set(CMAKE_CXX_FLAGS_DEBUG          "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
    set(CMAKE_CXX_FLAGS_MINSIZEREL     "${CMAKE_CXX_FLAGS_MINSIZEREL} /MT")
    set(CMAKE_CXX_FLAGS_RELEASE        "${CMAKE_CXX_FLAGS_RELEASE} /MT")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} /MT")

    set(CMAKE_C_FLAGS_DEBUG            "${CMAKE_C_FLAGS_DEBUG} /MTd")
    set(CMAKE_C_FLAGS_MINSIZEREL       "${CMAKE_C_FLAGS_MINSIZEREL} /MT")
    set(CMAKE_C_FLAGS_RELEASE          "${CMAKE_C_FLAGS_RELEASE} /MT")
    set(CMAKE_C_FLAGS_RELWITHDEBINFO   "${CMAKE_C_FLAGS_RELWITHDEBINFO} /MT")
  else()
    message(WARNING "SetLinkStaticRuntime is not supported if not using MSVC")
  endif()

endmacro()

macro(SetLinkSharedRuntime)
  if (MSVC)
    set(CMAKE_CXX_FLAGS_DEBUG          "${CMAKE_CXX_FLAGS_DEBUG} /MDd")
    set(CMAKE_CXX_FLAGS_MINSIZEREL     "${CMAKE_CXX_FLAGS_MINSIZEREL} /MD")
    set(CMAKE_CXX_FLAGS_RELEASE        "${CMAKE_CXX_FLAGS_RELEASE} /MD")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} /MD")

    set(CMAKE_C_FLAGS_DEBUG            "${CMAKE_C_FLAGS_DEBUG} /MDd")
    set(CMAKE_C_FLAGS_MINSIZEREL       "${CMAKE_C_FLAGS_MINSIZEREL} /MD")
    set(CMAKE_C_FLAGS_RELEASE          "${CMAKE_C_FLAGS_RELEASE} /MD")
    set(CMAKE_C_FLAGS_RELWITHDEBINFO   "${CMAKE_C_FLAGS_RELWITHDEBINFO} /MD")
  endif()
endmacro()

macro(SetFullOptimization)
  set(ADDITIONAL_COMPILE_FLAGS_FOR_RELEASE "")
  if(MSVC)
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
      set(ADDITIONAL_COMPILE_FLAGS_FOR_RELEASE "/GS-")
    else()
      set(ADDITIONAL_COMPILE_FLAGS_FOR_RELEASE "/GS- /arch:SSE2")
    endif()
  endif()
  if(UNIX)
    if (NOT APPLE)
      # gcc
      set(ADDITIONAL_COMPILE_FLAGS_FOR_RELEASE "-msse -msse2 -mfpmath=sse")
    else()
      # clang
      set(ADDITIONAL_COMPILE_FLAGS_FOR_RELEASE "-msse -msse2")
    endif()
  endif()

  set(CMAKE_CXX_FLAGS_MINSIZEREL     "${CMAKE_CXX_FLAGS_MINSIZEREL} ${ADDITIONAL_COMPILE_FLAGS_FOR_RELEASE}")
  set(CMAKE_CXX_FLAGS_RELEASE        "${CMAKE_CXX_FLAGS_RELEASE} ${ADDITIONAL_COMPILE_FLAGS_FOR_RELEASE}")
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} ${ADDITIONAL_COMPILE_FLAGS_FOR_RELEASE}")

  set(CMAKE_C_FLAGS_MINSIZEREL       "${CMAKE_C_FLAGS_MINSIZEREL} ${ADDITIONAL_COMPILE_FLAGS_FOR_RELEASE}")
  set(CMAKE_C_FLAGS_RELEASE          "${CMAKE_C_FLAGS_RELEASE} ${ADDITIONAL_COMPILE_FLAGS_FOR_RELEASE}")
  set(CMAKE_C_FLAGS_RELWITHDEBINFO   "${CMAKE_C_FLAGS_RELWITHDEBINFO} ${ADDITIONAL_COMPILE_FLAGS_FOR_RELEASE}")
endmacro()


#
# Helper functions for install rules
#

#
# SetUsedRuntimeSharedLibraries
# Tries to use a given list of external libraries to determine its dll name
# It looks if the dll exists and returns a list of full path to dll.
#
macro(SetUsedRuntimeSharedLibraries VAR_LIST_NAME BIN_DIR LIB_LIST)

  set(SHARED_LIB_LIST)
  set(is_optimized TRUE)

  foreach(f ${LIB_LIST})
    if(f STREQUAL "optimized")
      set(is_optimized TRUE)
    elseif(f STREQUAL "debug")
      set(is_optimized FALSE)
    else()

      if(is_optimized)
        get_filename_component(LIB_NAME ${f} NAME_WE ${f})

        set(T_VAR_LIST_NAME "${BIN_DIR}/${LIB_NAME}.dll")

        if(EXISTS ${T_VAR_LIST_NAME})
          list(APPEND SHARED_LIB_LIST ${T_VAR_LIST_NAME})
        else()
          message("Warning: ${T_VAR_LIST_NAME} does not exist")
        endif()
      endif()
    endif()
  endforeach()

#  message("SHARED_LIB_LIST = ${SHARED_LIB_LIST}")
  set(${VAR_LIST_NAME} ${SHARED_LIB_LIST})

endmacro()

#
# SetUsedRuntimeSharedLibrariesByTool
# Similar to SetUsedRuntimeSharedLibraries but tries to deduce dependencies
# using ldd or dumpbin
# param VAR_LIST_NAME
# param TARGET_FILE_NAME
# param BIN_DIRS
macro(SetUsedRuntimeSharedLibrariesByTool VAR_LIST_NAME TARGET_FILE_NAME BIN_DIRS)
  include(GetPrerequisites)

  set(SHARED_LIB_LIST)

  # hints where to find dumpbin
  set(gp_cmd_paths
    ${gp_cmd_paths}
    "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Wow6432Node\\Microsoft\\VisualStudio\\10.0;InstallDir]/../../VC/bin"
    )

  set(PRE_REQ)
  set(OPT_RECURSIVE 1)
  set(OPT_SYSTEM    1)

  get_prerequisites(${TARGET_FILE_NAME} PRE_REQ ${OPT_SYSTEM} ${OPT_RECURSIVE} "" "${BIN_DIRS}")

  if (CMAKE_SCRIPT_DEBUG)
    message("PRE_REQ = ${PRE_REQ}")
  endif()

  foreach(f ${PRE_REQ})

    unset(T_VAR_LIST_NAME) #from set()
    unset(T_VAR_LIST_NAME CACHE) #from find_path()
    find_path(T_VAR_LIST_NAME "${f}"
      ${BIN_DIRS}
      NO_DEFAULT_PATH
    )
    set(T_VAR_LIST_NAME "${T_VAR_LIST_NAME}/${f}")

    if (CMAKE_SCRIPT_DEBUG)
      message("${f} => ${T_VAR_LIST_NAME}")
    endif()

    if(EXISTS ${T_VAR_LIST_NAME})
      list(APPEND SHARED_LIB_LIST ${T_VAR_LIST_NAME})
    else()
      message("Warning: ${T_VAR_LIST_NAME} does not exist")
    endif()
  endforeach()

  set(${VAR_LIST_NAME} ${SHARED_LIB_LIST})

  if (CMAKE_SCRIPT_DEBUG)
    message("VAR_LIST_NAME = ${VAR_LIST_NAME}")
  endif()

endmacro()


#
# Similar to InstallRequiredSystemLibraries
# Install all dependent 3rdparty libs
# param TARGET_FILE_NAME
# param DIST_DIR
# param BIN_DIRS
#
macro(InstallRequiredSharedLibraries TARGET_FILE_NAME DIST_DIR BIN_DIRS)

  install(CODE "

   #
   # Begin InstallRequiredSharedLibraries ${TARGET_FILE_NAME} ${BIN_DIRS}
   #

   set(CMAKE_MODULE_PATH
       ${CMAKE_MODULE_PATH}
       ${PROJECT_ROOT}/build_util/cmake
   )

   include(CMakeFunctions)

   #set(CMAKE_SCRIPT_DEBUG ON)

   SetUsedRuntimeSharedLibrariesByTool(3RDPARTY_RUNTIME_LIBS ${TARGET_FILE_NAME} \"${BIN_DIRS}\")

   #message(\"3RDPARTY_RUNTIME_LIBS = \${3RDPARTY_RUNTIME_LIBS}\")
   FILE(INSTALL DESTINATION \"${DIST_DIR}\" TYPE FILE FILES \${3RDPARTY_RUNTIME_LIBS})

   #
   # End InstallRequiredSharedLibraries ${TARGET_FILE_NAME} ${BIN_DIRS}
   #
   "
    COMPONENT RUNTIME)

endmacro()

#
# Copy the python dll to the bin/install dir
# Win32 only!
# param PYTHON_LIB
# param DIST_DIR
macro(InstallRequiredPythonLibraries PYTHON_LIB)

  if(WIN32)
    get_filename_component(PYTHON_NAME ${PYTHON_LIB} NAME_WE)
    set(PYTHON_DLL_NAME "${PYTHON_NAME}.dll")
    #message("PYTHON_DLL_NAME = ${PYTHON_DLL_NAME}")

    #
    # go back to the the root python dir
    get_filename_component(PYTHON_DLL_PATH "${PYTHON_LIB}" PATH)
    get_filename_component(PYTHON_DLL_PATH "${PYTHON_DLL_PATH}" PATH)

    #message("0PYTHON_DLL_PATH = ${PYTHON_DLL_PATH}")

    set(PYTHON_DLL_PATH "${PYTHON_DLL_PATH}/DLLs/${PYTHON_DLL_NAME}")

    #message(FATAL_ERROR "1PYTHON_DLL_PATH = ${PYTHON_DLL_PATH}")

    if(EXISTS ${PYTHON_DLL_PATH})
      #message(FATAL_ERROR "1PYTHON_DLL_PATH = ${PYTHON_DLL_PATH}")
      install(FILES ${PYTHON_DLL_PATH} DESTINATION ${DIST_DIR})
    else()
      message("No python dll found: ${PYTHON_DLL_PATH}")
    endif()
  endif()
endmacro()


#
# Use deploy_qt.py to copy all dependent qt stuff to dest
# param PROJECT_ROOT
# param DEST_DIR
#
macro(InstallQTRuntimeLibraries PROJECT_ROOT DEST_DIR QT_VERSION)

  set(_qt_version "5.9.2")

  if (QT_VERSION)
    set(_qt_version ${QT_VERSION})
  endif()

  if (QT_SPECIAL_VERSION)
    set(_qt_special_build "--special-build ${QT_SPECIAL_VERSION}")
  endif()

  
  if (BUILD_X64)
    install(CODE "
  # InstallQTRuntimeLibraries begin
  #
  # just defer the task to the deploy script
  execute_process(
    COMMAND python \"${PROJECT_ROOT}/build_util/bin/deploy_qt.py\" --arch x64 --build_type Release --dest ${DEST_DIR} --qt-version ${_qt_version} ${_qt_special_build}
    )

  # InstallQTRuntimeLibraries end
   "
    COMPONENT RUNTIME)
  elseif(BUILD_X86)
    install(CODE "
  # InstallQTRuntimeLibraries begin
  #
  # just defer the task to the deploy script
  execute_process(
    COMMAND python \"${PROJECT_ROOT}/build_util/bin/deploy_qt.py\" --arch x86 --build_type Release --dest ${DEST_DIR}  --qt-version ${_qt_version} ${_qt_special_build}
    )

  # InstallQTRuntimeLibraries end
   "
    COMPONENT RUNTIME)
  endif()

endmacro()

#
# Use deploy_qt.py to copy a single lib & its dependencies
# param PROJECT_ROOT
# param DEST_DIR
# param packages ..
macro(InstallQTRuntimeSinglePackage PROJECT_ROOT DEST_DIR PACKAGE QT_VERSION)

  set(_qt_version "5.9.2")

  if (QT_VERSION)
    set(_qt_version ${QT_VERSION})
  endif()

  if (QT_SPECIAL_VERSION)
    set(_qt_special_build "--special-build ${QT_SPECIAL_VERSION}")
  endif()


  if (BUILD_X64)
    install(CODE "
  # InstallQTRuntimeLibraries begin
  #
  # just defer the task to the deploy script
  execute_process(
    COMMAND python \"${PROJECT_ROOT}/build_util/bin/deploy_qt.py\" --arch x64 --build_type Release --qt-module ${PACKAGE} --dest ${DEST_DIR} --qt-version ${_qt_version} ${_qt_special_build}
    )

  # InstallQTRuntimeLibraries end
   "
    COMPONENT RUNTIME)
  elseif(BUILD_X86)
    install(CODE "
  # InstallQTRuntimeLibraries begin
  #
  # just defer the task to the deploy script
  execute_process(
    COMMAND python \"${PROJECT_ROOT}/build_util/bin/deploy_qt.py\" --arch x86 --build_type Release --qt-module ${PACKAGE} --dest ${DEST_DIR} --qt-version ${_qt_version} ${_qt_special_build}
    )

  # InstallQTRuntimeLibraries end
   "
    COMPONENT RUNTIME)
  endif()

endmacro()


#
# Settings for an app with static wx
#
# Out:
#   DEP_STATIC_LIBS
#   WX_LINK_TYPE
#   WX_CPP_DEFINE
macro(SetAppBuildStaticWX)
  set(WX_LINK_TYPE static)
  set(WX_CPP_DEFINE)
  SetLinkStaticRuntime()

  if(WIN32)
    find_package(ZLIB REQUIRED)
    find_package(JPEG REQUIRED)
    find_package(TIFF REQUIRED)
    find_package(PNG REQUIRED)

    set(DEP_STATIC_LIBS
      ${TIFF_LIBRARIES}
      ${JPEG_LIBRARIES}
      ${PNG_LIBRARIES}
      Comctl32
      Rpcrt4
      )

  endif()

  if(UNIX)
    find_package(ZLIB REQUIRED)
    find_package(PNG REQUIRED)

    set(DEP_STATIC_LIBS
      #${TIFF_LIBRARIES}
      #${JPEG_LIBRARIES}
      ${PNG_LIBRARIES}
      )
  endif()

endmacro()

#
# Settings for an app with shared  wx
#
# Out:
#   DEP_STATIC_LIBS
#   WX_LINK_TYPE
#   WX_CPP_DEFINE
macro(SetAppBuildSharedWX)
  set(WX_LINK_TYPE shared)
  set(WX_CPP_DEFINE "WXUSINGDLL")
  SetLinkSharedRuntime()
  set(DEP_STATIC_LIBS "")
endmacro()


macro(FindDasyLab DASYLAB_DIR)

  set(DASYLAB_REGISTRY_KEYS
    "[HKEY_LOCAL_MACHINE\\SOFTWARE\\DASYLab\\14.0;Path]"
    "[HKEY_LOCAL_MACHINE\\SOFTWARE\\National Instruments\\DASYLab\\13.0;Path]"
    "[HKEY_LOCAL_MACHINE\\SOFTWARE\\National Instruments\\DASYLab\\12.0;Path]"
    "[HKEY_LOCAL_MACHINE\\SOFTWARE\\National Instruments\\DASYLab\\11.0;Path]"
    )

  foreach(REG_ENTRY ${DASYLAB_REGISTRY_KEYS})
    get_filename_component(TEMP_DASYLAB_DIR "${REG_ENTRY}" ABSOLUTE)

    if (NOT ${TEMP_DASYLAB_DIR} STREQUAL /registry)
      #message("FindDasyLab: DASYLAB_DIR = ${TEMP_DASYLAB_DIR}")
      set(DASYLAB_DIR ${TEMP_DASYLAB_DIR})
      break()
    endif()

  endforeach()

endmacro()


#
# Standard link libraries for unit tests
# param TEST_NAME
# not defined params:
# param WIN_LIBS List of additional libs for windows
# param LIN_LIBS List of additional libs for linux
# param OSX_LIBS List of additional libs for osx
macro(SetStandardLinkLibrariesForTest TEST_NAME)

  # copy extra arguments to be usable as list
  set(extra_macro_args ${ARGN})
  list(LENGTH extra_macro_args num_extra_macro_args)

  set(WIN_LIBS "")
  set(LIN_LIBS "")
  set(OSX_LIBS "")

  if (num_extra_macro_args GREATER 0)
    list(GET extra_macro_args 0 T_WIN_LIBS)
    string(REPLACE " " ";" WIN_LIBS ${T_WIN_LIBS})
  endif()

  if (num_extra_macro_args GREATER 1)
    list(GET extra_macro_args 1 T_LIN_LIBS)
    string(REPLACE " " ";" LIN_LIBS ${T_LIN_LIBS})
  endif()

  if (num_extra_macro_args GREATER 2)
    list(GET extra_macro_args 2 T_OSX_LIBS)
    string(REPLACE " " ";" OSX_LIBS ${T_OSX_LIBS})
  endif()

  if(WIN32)
    #
    # Libraries for windows
    target_link_libraries(${TEST_NAME}
      ws2_32.lib
      ${WIN_LIBS}
      )
  endif()

  if(UNIX)
    if(APPLE)
      #
      # Libraries for apple/darwin/osx
      if(NOT ${OSX_LIBS} STREQUAL "")
	    target_link_libraries(${TEST_NAME}
	      ${OSX_LIBS}
	      )
      endif()
    else()
      #
      # Libraries for Linux
      target_link_libraries(${TEST_NAME}
	    dl
	    rt
	    ${LIN_LIBS}
        pthread
	    )
    endif()

  endif()

endmacro()

#
# Check if the given library exists.
#  - checks if the variable containing the lib is set.
macro(TargetLinkLibrariesChecked TARGET)

  set(_dep_libs "")

  foreach(_lib ${ARGN})
    if(NOT ${_lib})
      message(FATAL_ERROR "${_lib} not set")
    else()
      list(APPEND _dep_libs ${${_lib}})
    endif()
  endforeach()

  target_link_libraries(${TARGET} ${_dep_libs})

  if (CMAKE_SCRIPT_DEBUG)
    message(STATUS "${TARGET} ${_dep_libs}")
  endif()
endmacro()

#
# "Extract" boost dlls from boost libs
# Note: Macro only takes release libs!!!
macro(GetDllFromLib DLL_LIST)
  foreach(_lib ${ARGN})
    if(${_lib} MATCHES "(mt-gd)" OR ${_lib} MATCHES "(optimized)" OR ${_lib} MATCHES "(debug)")
      # do nothing ...
    else()
      string(REPLACE ".lib" ".dll" _dll ${_lib})
      list(APPEND DLL_LIST ${_dll})
    endif()
  endforeach()
endmacro()


#
# replace __ varviables (from us) with their correct cmake counterpart
macro(UpdateUnderscoredPaths VERSION_SOURCE_FILES)
  unset(VERSION_SOURCE_FILES)
  foreach(_p ${ARGN})
    string(REPLACE "__CMAKE_CURRENT_BINARY_DIR__" "${CMAKE_CURRENT_BINARY_DIR}" _r ${_p})
    list(APPEND VERSION_SOURCE_FILES ${_r})
  endforeach()

  list(REMOVE_DUPLICATES VERSION_SOURCE_FILES)
endmacro()

#
# Assign a stable GUID to visual studio 2013 projects
macro(SetProjectGuid LIBNAME GUID)
# http://stackoverflow.com/questions/28959488/how-to-enable-incremental-builds-for-vs-2013-with-cmake-and-long-target-names
  set("${LIBNAME}_GUID_CMAKE" ${GUID} CACHE INTERNAL "remove this and Visual Studio will mess up incremental builds")
endmacro()


#
# make sure a target is built exactly once
macro(AddUniqueTargetFromSubdirectory TARGET_NAME SOURCE_DIR BINARY_DIR)
  if(NOT TARGET ${TARGET_NAME})
    add_subdirectory(${SOURCE_DIR} ${BINARY_DIR})
  endif()
endmacro()


#
# Configure the used Qt Version
macro(SetupQtVersion)
  if(NOT QT_VERSION)
    set(QT_VERSION "5.9.2")
  endif()

  if (WIN32)
    if (QT_USE_ANGLE)
      set(QT_SPECIAL_VERSION "angle")
      message("Using ANGLE build of Qt")
    endif()

    if ("${QT_VERSION}" STREQUAL "5.6.0")
      set(QT_SPECIAL_VERSION "angle")
      message("Using ANGLE build of Qt")
    endif()
  endif()
  message(STATUS "QT_VERSION = ${QT_VERSION}")
endmacro()


#
# Setup a common major version.h file
macro(SetupMajorVersionFile MAJOR_VERSION_FILE)

  if(NOT EXISTS ${MAJOR_VERSION_FILE})
    message(FATAL_ERROR "Could not find ${MAJOR_VERSION_FILE}")
  else()
    get_filename_component(MAJOR_VERSION_FILE_PATH_DIR ${MAJOR_VERSION_FILE} DIRECTORY)
    include_directories(${MAJOR_VERSION_FILE_PATH_DIR})
  endif()

endmacro()
