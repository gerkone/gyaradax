cmake_minimum_required(VERSION 3.26)

project(gyaradax_cuda LANGUAGES CUDA CXX)

# ── Standards ───────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ── GPU architectures ─────────────────────────────────────────────
set(GPU_ARCHITECTURES "native" CACHE STRING
    "Target GPU compute capabilities (semicolon-separated, or 'native')")
set(CMAKE_CUDA_ARCHITECTURES ${GPU_ARCHITECTURES})

# ── Find Python and auto-detect JAX FFI headers ────────────────────
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

# ── CUDAToolkit ───────────────────────────────────────────────────
find_package(CUDAToolkit REQUIRED)
message(STATUS "Target GPU Arch:    ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Found CUDA compiler: ${CMAKE_CUDA_COMPILER}")
message(STATUS "Found CXX compiler:  ${CMAKE_CXX_COMPILER}")
message(STATUS "Found CUDA toolkit:  ${CUDAToolkit_VERSION}")

# ── LTO fatbin tooling ────────────────────────────────────────────

# cuFFT LTO callback archs. Override on older toolkits: compute_100 needs CUDA >= 12.8.
set(LTO_ARCHITECTURES "80;90;100" CACHE STRING
    "Compute capabilities to emit cuFFT LTO callback fatbins for")
set(GENCODE_FLAGS_LTO "")
foreach(_lto_arch IN LISTS LTO_ARCHITECTURES)
    list(APPEND GENCODE_FLAGS_LTO
        --generate-code arch=compute_${_lto_arch},code=lto_${_lto_arch})
endforeach()
list(APPEND GENCODE_FLAGS_LTO --ptxas-options=-v)

get_filename_component(CUDA_COMPILER_DIR ${CMAKE_CUDA_COMPILER} DIRECTORY)
set(BIN2C_EXECUTABLE ${CUDA_COMPILER_DIR}/bin2c)

function(add_lto_fatbin_target NAME FILEPATH)
    add_custom_command(
        OUTPUT  ${CMAKE_BINARY_DIR}/${NAME}.fatbin
                ${CMAKE_BINARY_DIR}/${NAME}_fatbin.h
        DEPENDS ${FILEPATH}
                ${CMAKE_CURRENT_SOURCE_DIR}/lto_callbacks/bracket_v5_z2z_pack.cuh
                ${CMAKE_CURRENT_SOURCE_DIR}/lto_callbacks/bracket_v6_info.cuh
        COMMAND ${CMAKE_CUDA_COMPILER} ${GENCODE_FLAGS_LTO} -dc -fatbin
                ${FILEPATH} -o ${CMAKE_BINARY_DIR}/${NAME}.fatbin
        COMMAND ${BIN2C_EXECUTABLE} --name ${NAME}_fatbin --type longlong
                ${CMAKE_BINARY_DIR}/${NAME}.fatbin
                > ${CMAKE_BINARY_DIR}/${NAME}_fatbin.h
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
endfunction()

add_lto_fatbin_target(bracket_v5_z2z_load_cb
    ${CMAKE_CURRENT_SOURCE_DIR}/lto_callbacks/bracket_v5_z2z_load_cb.cu)
add_lto_fatbin_target(bracket_v5_d2z_load_cb
    ${CMAKE_CURRENT_SOURCE_DIR}/lto_callbacks/bracket_v5_d2z_load_cb.cu)
add_lto_fatbin_target(bracket_v5_store_cb
    ${CMAKE_CURRENT_SOURCE_DIR}/lto_callbacks/bracket_v5_store_cb.cu)

set(LTO_HEADERS
    ${CMAKE_BINARY_DIR}/bracket_v5_z2z_load_cb_fatbin.h
    ${CMAKE_BINARY_DIR}/bracket_v5_d2z_load_cb_fatbin.h
    ${CMAKE_BINARY_DIR}/bracket_v5_store_cb_fatbin.h)


# ── Shared library ────────────────────────────────────────────────
add_library(gyaradax_cuda SHARED
    kernels/cufft_graph_bracket_fp64.cu
    kernels/cufft_graph_bracket_true_fp32.cu
    lto_callbacks/bracket_v5_lto_fatbins.cu
    kernels/apply_parallel.cu
    kernels/apply_parallel_dual.cu
    kernels/apply_vpar_stencil.cu
    kernels/apply_vpar_dual_stencil.cu
    kernels/linear_rhs_fused.cu
    kernels/field_moments.cu
    ${LTO_HEADERS})

set_target_properties(gyaradax_cuda PROPERTIES
    PREFIX "lib"
    OUTPUT_NAME "gyaradax_cuda")

target_compile_options(gyaradax_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas -v>)

target_include_directories(gyaradax_cuda PRIVATE
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels
    ${CMAKE_CURRENT_SOURCE_DIR}/lto_callbacks
    ${CMAKE_BINARY_DIR}
    ${JAX_INCLUDE_DIR})

target_link_libraries(gyaradax_cuda PRIVATE
    CUDA::cufft
    CUDA::cudart
    CUDA::nvJitLink)


# ── Optional cuFFTDx (pip install nvidia-mathdx): v6 bracket ──────
set(CUFFTDX_INCLUDE_DIR "" CACHE PATH "cuFFTDx include directory; empty = detect from Python")
if(NOT CUFFTDX_INCLUDE_DIR)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c
            "import os, nvidia.mathdx as m; print(os.path.join(list(m.__path__)[0], 'include'))"
        OUTPUT_VARIABLE _dx_inc
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _dx_exit
        ERROR_QUIET)
    if(_dx_exit EQUAL 0)
        set(CUFFTDX_INCLUDE_DIR ${_dx_inc})
    endif()
endif()

if(CUFFTDX_INCLUDE_DIR AND EXISTS "${CUFFTDX_INCLUDE_DIR}/cufftdx.hpp")
    # cuFFTDx SM<> values of the kernel architectures (90 -> 900, 103 -> 1030)
    if(GPU_ARCHITECTURES STREQUAL "native")
        execute_process(COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
            OUTPUT_VARIABLE _caps OUTPUT_STRIP_TRAILING_WHITESPACE)
        string(REPLACE "." "" _caps "${_caps}")
        string(REPLACE "\n" ";" _dx_arch_list "${_caps}")
    else()
        set(_dx_arch_list ${GPU_ARCHITECTURES})
    endif()
    set(_dx_archs "")
    foreach(_a IN LISTS _dx_arch_list)
        string(REGEX MATCH "^[0-9]+" _n "${_a}")
        math(EXPR _v "${_n} * 10")
        list(APPEND _dx_archs ${_v})
    endforeach()
    list(REMOVE_DUPLICATES _dx_archs)
    string(REPLACE ";" ", " _dx_archs_csv "${_dx_archs}")
    # nvcc splits -D values at commas, so the list goes through a generated header
    file(WRITE ${CMAKE_BINARY_DIR}/gyx_dx_archs.h "#define GYX_DX_ARCHS ${_dx_archs_csv}\n")

    target_sources(gyaradax_cuda PRIVATE kernels/cufft_bracket_v6.cu)
    target_include_directories(gyaradax_cuda PRIVATE
        ${CUFFTDX_INCLUDE_DIR}
        ${CUFFTDX_INCLUDE_DIR}/../external/cutlass/include)
    message(STATUS "cuFFTDx:            ${CUFFTDX_INCLUDE_DIR} (v6 bracket for SM ${_dx_archs_csv})")
else()
    message(STATUS "cuFFTDx:            not found, v6 bracket disabled (pip install nvidia-mathdx)")
endif()


# ── Install: place .so next to this CMakeLists.txt ────────────────
# cmake --install _build  →  gyaradax/backends/cuda_kernels/libgyaradax_cuda.so
install(TARGETS gyaradax_cuda
    LIBRARY DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}
    COMPONENT runtime)
