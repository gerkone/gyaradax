cmake_minimum_required(VERSION 3.18)
project(
  test_lto_project
  DESCRIPTION "cuFFT LTO callbacks + CUDA kernels for gyaradax"
  LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if("${CMAKE_BUILD_TYPE}" STREQUAL "")
  set(CMAKE_BUILD_TYPE Release)
endif()

# ── Architecture detection ──────────────────────────────────────
# Default: build for A100 + B300. Override with:
#   cmake .. -DGPU_ARCHITECTURES="80"            (A100 only)
#   cmake .. -DGPU_ARCHITECTURES="100"            (Blackwell B200)
#   cmake .. -DGPU_ARCHITECTURES="103"            (Blackwell B300 Ultra)
#   cmake .. -DGPU_ARCHITECTURES="80;90;103"      (A100 + Hopper + B300)
if(NOT DEFINED GPU_ARCHITECTURES)
  set(GPU_ARCHITECTURES "80;103" CACHE STRING "Target GPU architectures (semicolon-separated)")
endif()

set(CMAKE_CUDA_ARCHITECTURES ${GPU_ARCHITECTURES})

# ── LTO fatbins need per-architecture gencode flags ─────────────
# cuFFT LTO callbacks require arch-specific lto_XX code, not just PTX.
# We generate one fatbin per callback containing all target architectures.
set(GENCODE_FLAGS_LTO "")
foreach(ARCH ${GPU_ARCHITECTURES})
  list(APPEND GENCODE_FLAGS_LTO
    --generate-code arch=compute_${ARCH},code=lto_${ARCH})
endforeach()
list(APPEND GENCODE_FLAGS_LTO --ptxas-options=-v)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Find bin2c relative to the detected nvcc compiler path
get_filename_component(CUDA_COMPILER_DIR ${CMAKE_CUDA_COMPILER} DIRECTORY)
set(BIN2C_EXECUTABLE ${CUDA_COMPILER_DIR}/bin2c)

# ── Fatbin build rules ──────────────────────────────────────────
function(add_lto_fatbin_target NAME FILENAME)
    add_custom_command(
        OUTPUT  ${CMAKE_BINARY_DIR}/${NAME}.fatbin ${CMAKE_BINARY_DIR}/${NAME}_fatbin.h
        DEPENDS ${CMAKE_SOURCE_DIR}/${FILENAME}
        COMMAND ${CMAKE_CUDA_COMPILER} ${GENCODE_FLAGS_LTO} -dc -fatbin
                ${CMAKE_SOURCE_DIR}/${FILENAME}
                -o ${CMAKE_BINARY_DIR}/${NAME}.fatbin
        COMMAND ${BIN2C_EXECUTABLE} --name ${NAME}_fatbin --type longlong
                ${CMAKE_BINARY_DIR}/${NAME}.fatbin
                > ${CMAKE_BINARY_DIR}/${NAME}_fatbin.h
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
endfunction()

add_lto_fatbin_target(bracket_load_cb              bracket_load_cb.cu)
add_lto_fatbin_target(bracket_d2z_load_cb           bracket_d2z_load_cb.cu)
add_lto_fatbin_target(bracket_z2z_load_cb           bracket_z2z_load_cb.cu)
add_lto_fatbin_target(bracket_d2z_load_v2_cb        bracket_d2z_load_v2_cb.cu)
add_lto_fatbin_target(bracket_store_cb              bracket_store_cb.cu)
add_lto_fatbin_target(bracket_d2z_z2z_load_cb       bracket_d2z_z2z_load_cb.cu)
add_lto_fatbin_target(bracket_z2z_merged_load_cb    bracket_z2z_merged_load_cb.cu)
add_lto_fatbin_target(bracket_d2z_load_cb_merged    bracket_d2z_load_cb_merged.cu)

set(LTO_HEADERS
    ${CMAKE_BINARY_DIR}/bracket_load_cb_fatbin.h
    ${CMAKE_BINARY_DIR}/bracket_d2z_load_cb_fatbin.h
    ${CMAKE_BINARY_DIR}/bracket_z2z_load_cb_fatbin.h
    ${CMAKE_BINARY_DIR}/bracket_d2z_load_v2_cb_fatbin.h
    ${CMAKE_BINARY_DIR}/bracket_store_cb_fatbin.h
    ${CMAKE_BINARY_DIR}/bracket_d2z_z2z_load_cb_fatbin.h
    ${CMAKE_BINARY_DIR}/bracket_z2z_merged_load_cb_fatbin.h
    ${CMAKE_BINARY_DIR}/bracket_d2z_load_cb_merged_fatbin.h)

# ── Shared library target ───────────────────────────────────────
add_library(lto_bracket SHARED
    ${PROJECT_SOURCE_DIR}/cufft_lto_bracket.cu
    ${PROJECT_SOURCE_DIR}/cufft_bracket.cu
    ${PROJECT_SOURCE_DIR}/apply_parallel.cu
    ${PROJECT_SOURCE_DIR}/apply_parallel_dual.cu
    ${PROJECT_SOURCE_DIR}/apply_vpar_stencil.cu
    ${PROJECT_SOURCE_DIR}/apply_vpar_dual_stencil.cu
    ${PROJECT_SOURCE_DIR}/linear_rhs_fused.cu
    ${PROJECT_SOURCE_DIR}/linear_rhs_vtiled.cu
    ${PROJECT_SOURCE_DIR}/compute_scale_factors.cu
    ${LTO_HEADERS})

set_target_properties(lto_bracket PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
    PREFIX "lib")

target_compile_options(lto_bracket PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>: -Xptxas -v>)

# ── JAX FFI headers ─────────────────────────────────────────────
set(JAX_INCLUDE_DIR "/system/apps/userenv/volkmann/jax_env/lib/python3.14/site-packages/jaxlib/include"
    CACHE PATH "Path to JAX FFI headers")

target_include_directories(lto_bracket PRIVATE
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_BINARY_DIR}
    ${JAX_INCLUDE_DIR})

target_link_libraries(lto_bracket PRIVATE
    CUDA::cufft
    CUDA::cudart
    CUDA::nvJitLink)