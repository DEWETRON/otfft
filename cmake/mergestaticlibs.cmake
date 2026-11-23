# Copyright Modelon AB 2012
# Copyright DEWETRON GmbH 2017

# This program is free software: you can redistribute it and/or modify
# it under the terms of the BSD style license.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# FMILIB_License.txt file for more details.

# You should have received a copy of the FMILIB_License.txt file
# along with this program. If not, contact Modelon AB <http://www.modelon.com>.

# Merge_static_libs(outlib lib1 lib2 ... libn) merges a number of static
# libs into a single static library
function(merge_static_libs outlib)
  set(libs ${ARGV})
  list(REMOVE_AT libs 0)
  # Create a dummy file that the target will depend on
  set(dummyfile ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${outlib}_dummy.c)
  file(WRITE ${dummyfile} "const char * dummy = \"${dummyfile}\";")

  add_library(${outlib} STATIC ${dummyfile})

  # Library files, resolved per configuration at generate time
  set(libfiles "")
  foreach(lib ${libs})
    get_target_property(libtype ${lib} TYPE)
    if(NOT libtype STREQUAL "STATIC_LIBRARY")
      message(FATAL_ERROR "Merge_static_libs can only process static libraries")
    endif()
    list(APPEND libfiles "$<TARGET_FILE:${lib}>")
  endforeach()
  message(STATUS "will be merging ${libs}")

  if(MSVC)
    # lib.exe does the merging: pass the other libraries as extra inputs
    string(REPLACE ";" " " flags "${libfiles}")
    set_target_properties(${outlib} PROPERTIES STATIC_LIBRARY_FLAGS "${flags}")

  elseif(APPLE)
    # Use OSX's libtool to merge archives
    add_custom_command(TARGET ${outlib} POST_BUILD
      COMMAND /usr/bin/libtool -static -o $<TARGET_FILE:${outlib}>.merged
              $<TARGET_FILE:${outlib}> ${libfiles}
      COMMAND ${CMAKE_COMMAND} -E rename
              $<TARGET_FILE:${outlib}>.merged $<TARGET_FILE:${outlib}>
      COMMAND_EXPAND_LISTS)

  else()
    # general UNIX - add the libraries with an ar MRI script
    set(mri "${CMAKE_CURRENT_BINARY_DIR}/${outlib}_merge_$<CONFIG>.mri")
    set(content "OPEN $<TARGET_FILE:${outlib}>\n")
    foreach(f ${libfiles})
      string(APPEND content "ADDLIB ${f}\n")
    endforeach()
    string(APPEND content "SAVE\nEND\n")
    file(GENERATE OUTPUT "${mri}" CONTENT "${content}")
    add_custom_command(TARGET ${outlib} POST_BUILD
      COMMAND ${CMAKE_AR} -M < "${mri}"
      VERBATIM)
  endif()

  file(WRITE ${dummyfile}.base "const char* ${outlib}_sublibs=\"${libs}\";")
  add_custom_command(
    OUTPUT  ${dummyfile}
    COMMAND ${CMAKE_COMMAND}  -E copy ${dummyfile}.base ${dummyfile}
    DEPENDS ${libs}
    WORKING_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
  add_dependencies(${outlib} ${libs})

endfunction()
