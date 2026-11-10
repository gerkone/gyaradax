cmake_minimum_required(VERSION 3.26)

# hipcc must be the compiler before project()
find_program(HIPCC hipcc HINTS /opt/rocm/bin ENV ROCM_PATH PATH_SUFFIXES bin REQUIRED)
set(CMAKE_CXX_COMPILER ${HIPCC} CACHE FILEPATH "HIP compiler")

project(gyaradax_hip LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(GPU_ARCHITECTURES "native" CACHE STRING "Target gfx architectures (semicolon-separated, or 'native')")
if(GPU_ARCHITECTURES STREQUAL "native")
    find_program(ROCMINFO rocminfo HINTS /opt/rocm/bin REQUIRED)
    execute_process(COMMAND ${ROCMINFO} OUTPUT_VARIABLE _info ERROR_QUIET)
    string(REGEX MATCHALL "Name: +gfx[0-9a-f]+" _names "${_info}")
    string(REGEX MATCHALL "gfx[0-9a-f]+" _archs "${_names}")
    list(REMOVE_DUPLICATES _archs)
    if(NOT _archs)
        message(FATAL_ERROR "No GPU found; set -DGPU_ARCHITECTURES=gfx942")
    endif()
else()
    set(_archs ${GPU_ARCHITECTURES})
endif()
list(JOIN _archs "," _offload)

execute_process(
    COMMAND python3 -c
        "import sys, jaxlib, pathlib; print(jaxlib.__version__); print(pathlib.Path(jaxlib.__file__).parent / 'include')"
    OUTPUT_VARIABLE JAX_INFO OUTPUT_STRIP_TRAILING_WHITESPACE RESULT_VARIABLE _jax_exit)
if(NOT _jax_exit EQUAL 0)
    message(FATAL_ERROR "Could not locate JAX FFI headers in the current environment.")
endif()
string(REPLACE "\n" ";" JAX_INFO_LIST "${JAX_INFO}")
list(GET JAX_INFO_LIST 0 JAXLIB_VERSION)
list(GET JAX_INFO_LIST 1 JAX_INCLUDE_DIR)

find_package(hipfft CONFIG REQUIRED HINTS /opt/rocm/lib/cmake /opt/rocm)

message(STATUS "Found jaxlib:       ${JAXLIB_VERSION}")
message(STATUS "Found JAX include:  ${JAX_INCLUDE_DIR}")
message(STATUS "Found hipcc:        ${HIPCC}")
message(STATUS "Target GPU arch:    ${_offload}")

set(CUDA_KERNELS ${CMAKE_CURRENT_SOURCE_DIR}/../cuda_kernels)
set(GYX_SOURCES
    ${CUDA_KERNELS}/kernels/linear_rhs_fused.cu
    ${CUDA_KERNELS}/kernels/linear_rhs_es.cu
    ${CUDA_KERNELS}/kernels/collision_rhs.cu
    ${CUDA_KERNELS}/kernels/field_moments.cu
    kernels/hip_bracket_v6.cu)
set_source_files_properties(${GYX_SOURCES} PROPERTIES LANGUAGE CXX COMPILE_OPTIONS "-x;hip")

add_library(gyaradax_hip SHARED ${GYX_SOURCES})
target_compile_options(gyaradax_hip PRIVATE --offload-arch=${_offload} -Wno-return-type)
# register-window linear rhs on amd
target_compile_definitions(gyaradax_hip PRIVATE GYX_LIN_ES_REG GYX_LIN_ES_XCD=8 GYX_LIN_ES_VCH=16 GYX_LIN_NT)
target_link_options(gyaradax_hip PRIVATE --hip-link --offload-arch=${_offload})
target_include_directories(gyaradax_hip PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/compat
    ${CUDA_KERNELS}/kernels
    ${CUDA_KERNELS}/lto_callbacks
    ${JAX_INCLUDE_DIR})
target_link_libraries(gyaradax_hip PRIVATE hip::hipfft)

install(TARGETS gyaradax_hip LIBRARY DESTINATION ${CMAKE_CURRENT_SOURCE_DIR} COMPONENT runtime)
