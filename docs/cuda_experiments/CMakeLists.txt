cmake_minimum_required(VERSION 3.26)

project(gyaradax_cuda_experiments LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(GPU_ARCHITECTURES "native" CACHE STRING
    "Target GPU compute capabilities (semicolon-separated, or 'native')")
set(CMAKE_CUDA_ARCHITECTURES ${GPU_ARCHITECTURES})

# Match the production CUDA build: resolve JAX FFI headers from the active
# python3 environment rather than hard-coding a site-packages path.
execute_process(
    COMMAND python3 -c
        "import sys, jaxlib, pathlib; print(jaxlib.__version__); print(pathlib.Path(jaxlib.__file__).parent / 'include'); print(sys.executable)"
    OUTPUT_VARIABLE JAX_INFO
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _jax_exit
)

if(NOT _jax_exit EQUAL 0)
    message(FATAL_ERROR "Could not locate JAX FFI headers in the current environment.")
endif()

string(REPLACE "\n" ";" JAX_INFO_LIST "${JAX_INFO}")
list(GET JAX_INFO_LIST 0 JAXLIB_VERSION)
list(GET JAX_INFO_LIST 1 JAX_INCLUDE_DIR)
list(GET JAX_INFO_LIST 2 Python3_EXECUTABLE)

message(STATUS "Found Python:       ${Python3_EXECUTABLE}")
message(STATUS "Found jaxlib:       ${JAXLIB_VERSION}")
message(STATUS "Found JAX include:  ${JAX_INCLUDE_DIR}")

find_package(CUDAToolkit REQUIRED)
message(STATUS "Target GPU Arch:    ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Found CUDA compiler: ${CMAKE_CUDA_COMPILER}")
message(STATUS "Found CUDA toolkit:  ${CUDAToolkit_VERSION}")

file(GLOB EXPERIMENT_KERNELS CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/*.cu)

if(NOT EXPERIMENT_KERNELS)
    message(FATAL_ERROR "No CUDA experiment kernels found under kernels/*.cu")
endif()

add_library(gyaradax_cuda_experiments SHARED ${EXPERIMENT_KERNELS})

set_target_properties(gyaradax_cuda_experiments PROPERTIES
    PREFIX "lib"
    OUTPUT_NAME "gyaradax_cuda_experiments"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

target_include_directories(gyaradax_cuda_experiments PRIVATE
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels
    ${JAX_INCLUDE_DIR})

target_link_libraries(gyaradax_cuda_experiments PRIVATE
    CUDA::cudart)
