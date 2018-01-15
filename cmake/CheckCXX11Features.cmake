#
# C++ compiler feature tests
#

if(my_module_CXX11Features_included)
  return()
endif(my_module_CXX11Features_included)
set(my_module_CXX11Features_included true)

#
### Check for needed compiler flags
#
include(CheckCXXCompilerFlag)

check_cxx_compiler_flag("-std=gnu++11" _HAS_GNU11_FLAG)
if (NOT _HAS_GNU11_FLAG)
    check_cxx_compiler_flag("-std=c++11" _HAS_CXX11_FLAG)
    if (NOT _HAS_CXX11_FLAG)
        check_cxx_compiler_flag("-std=c++0x" _HAS_CXX0X_FLAG)
    endif ()
endif()

if (_HAS_GNU11_FLAG)
    set(CXX11_COMPILER_FLAGS "-std=gnu++11")
elseif (_HAS_CXX11_FLAG)
    set(CXX11_COMPILER_FLAGS "-std=c++11")
elseif (_HAS_CXX0X_FLAG)
    set(CXX11_COMPILER_FLAGS "-std=c++0x")
endif ()


function(cxx11_check_feature FEATURE_NAME RESULT_VAR)
    if (NOT DEFINED ${RESULT_VAR})
        set(_bindir "${CMAKE_CURRENT_BINARY_DIR}/cxx11_${FEATURE_NAME}")

        set(_SRCFILE_BASE ${CMAKE_CURRENT_LIST_DIR}/CheckCXX11Features/cxx11-test-${FEATURE_NAME})
        set(_LOG_NAME "\"${FEATURE_NAME}\"")
        message(STATUS "Checking C++11 support for ${_LOG_NAME}")

        set(_SRCFILE "${_SRCFILE_BASE}.cpp")
        set(_SRCFILE_FAIL "${_SRCFILE_BASE}_fail.cpp")
        set(_SRCFILE_FAIL_COMPILE "${_SRCFILE_BASE}_fail_compile.cpp")


        if (CROSS_COMPILING)
            try_compile(${RESULT_VAR} "${_bindir}" "${_SRCFILE}"
                        COMPILE_DEFINITIONS "${CXX11_COMPILER_FLAGS}")
            if (${RESULT_VAR} AND EXISTS ${_SRCFILE_FAIL})
                try_compile(${RESULT_VAR} "${_bindir}_fail" "${_SRCFILE_FAIL}"
                            COMPILE_DEFINITIONS "${CXX11_COMPILER_FLAGS}")
            endif (${RESULT_VAR} AND EXISTS ${_SRCFILE_FAIL})
        else (CROSS_COMPILING)
            try_run(_RUN_RESULT_VAR _COMPILE_RESULT_VAR
                    "${_bindir}" "${_SRCFILE}"
                    COMPILE_DEFINITIONS "${CXX11_COMPILER_FLAGS}")
            if (_COMPILE_RESULT_VAR AND NOT _RUN_RESULT_VAR)
                set(${RESULT_VAR} TRUE)
            else (_COMPILE_RESULT_VAR AND NOT _RUN_RESULT_VAR)
                set(${RESULT_VAR} FALSE)
            endif (_COMPILE_RESULT_VAR AND NOT _RUN_RESULT_VAR)
            if (${RESULT_VAR} AND EXISTS ${_SRCFILE_FAIL})
                try_run(_RUN_RESULT_VAR _COMPILE_RESULT_VAR
                        "${_bindir}_fail" "${_SRCFILE_FAIL}"
                         COMPILE_DEFINITIONS "${CXX11_COMPILER_FLAGS}")
                if (_COMPILE_RESULT_VAR AND _RUN_RESULT_VAR)
                    set(${RESULT_VAR} TRUE)
                else (_COMPILE_RESULT_VAR AND _RUN_RESULT_VAR)
                    set(${RESULT_VAR} FALSE)
                endif (_COMPILE_RESULT_VAR AND _RUN_RESULT_VAR)
            endif (${RESULT_VAR} AND EXISTS ${_SRCFILE_FAIL})
        endif (CROSS_COMPILING)
        if (${RESULT_VAR} AND EXISTS ${_SRCFILE_FAIL_COMPILE})
            try_compile(_TMP_RESULT "${_bindir}_fail_compile" "${_SRCFILE_FAIL_COMPILE}"
                        COMPILE_DEFINITIONS "${CXX11_COMPILER_FLAGS}")
            if (_TMP_RESULT)
                set(${RESULT_VAR} FALSE)
            else (_TMP_RESULT)
                set(${RESULT_VAR} TRUE)
            endif (_TMP_RESULT)
        endif (${RESULT_VAR} AND EXISTS ${_SRCFILE_FAIL_COMPILE})

        if (${RESULT_VAR})
            message(STATUS "Checking C++11 support for ${_LOG_NAME}: works")
        else (${RESULT_VAR})
            message(STATUS "Checking C++11 support for ${_LOG_NAME}: not supported")
        endif (${RESULT_VAR})
        set(${RESULT_VAR} ${${RESULT_VAR}} CACHE INTERNAL "C++11 support for ${_LOG_NAME}")
    endif (NOT DEFINED ${RESULT_VAR})
endfunction(cxx11_check_feature)



#cxx11_check_feature("__func__" HAS_CXX11_FUNC)
#cxx11_check_feature("auto" HAS_CXX11_AUTO)
#cxx11_check_feature("auto_ret_type" HAS_CXX11_AUTO_RET_TYPE)
#cxx11_check_feature("class_override_final" HAS_CXX11_CLASS_OVERRIDE)
cxx11_check_feature("constexpr" HAS_CXX11_CONSTEXPR)
#cxx11_check_feature("cstdint" HAS_CXX11_CSTDINT_H)
#cxx11_check_feature("decltype" HAS_CXX11_DECLTYPE)
#cxx11_check_feature("initializer_list" HAS_CXX11_INITIALIZER_LIST)
#cxx11_check_feature("lambda" HAS_CXX11_LAMBDA)
#cxx11_check_feature("long_long" HAS_CXX11_LONG_LONG)
#cxx11_check_feature("nullptr" HAS_CXX11_NULLPTR)
#cxx11_check_feature("regex" HAS_CXX11_LIB_REGEX)
#cxx11_check_feature("rvalue-references" HAS_CXX11_RVALUE_REFERENCES)
#cxx11_check_feature("sizeof_member" HAS_CXX11_SIZEOF_MEMBER)
#cxx11_check_feature("static_assert" HAS_CXX11_STATIC_ASSERT)
#cxx11_check_feature("variadic_templates" HAS_CXX11_VARIADIC_TEMPLATES)
