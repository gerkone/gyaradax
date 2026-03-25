cmake_minimum_required(VERSION 3.18)

project(
  test_lto_project
  DESCRIPTION "cuFFT LTO test for test_lto.cpp"
  LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if("${CMAKE_BUILD_TYPE}" STREQUAL "")
  set(CMAKE_BUILD_TYPE Release)
endif()

# Explicitly target A100 (sm_80) for LTO compilation
set(CUDA_LTO_ARCHITECTURE 80)
set(GENCODE_FLAGS_LTO --generate-code arch=compute_${CUDA_LTO_ARCHITECTURE},code=lto_${CUDA_LTO_ARCHITECTURE})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Find bin2c relative to the detected nvcc compiler path
get_filename_component(CUDA_COMPILER_DIR ${CMAKE_CUDA_COMPILER} DIRECTORY)
set(BIN2C_EXECUTABLE ${CUDA_COMPILER_DIR}/bin2c)

# --- Fatbin Rules ---

function(add_lto_fatbin_target NAME FILENAME)
    add_custom_command(
        OUTPUT  ${CMAKE_BINARY_DIR}/${NAME}.fatbin ${CMAKE_BINARY_DIR}/${NAME}_fatbin.h
        DEPENDS ${CMAKE_SOURCE_DIR}/${FILENAME}
        COMMAND ${CMAKE_CUDA_COMPILER} ${GENCODE_FLAGS_LTO} -dc -fatbin ${CMAKE_SOURCE_DIR}/${FILENAME} -o ${CMAKE_BINARY_DIR}/${NAME}.fatbin
        COMMAND ${BIN2C_EXECUTABLE} --name ${NAME}_fatbin --type longlong ${CMAKE_BINARY_DIR}/${NAME}.fatbin > ${CMAKE_BINARY_DIR}/${NAME}_fatbin.h
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
endfunction()

add_lto_fatbin_target(bracket_load_cb          bracket_load_cb.cu)
add_lto_fatbin_target(bracket_d2z_load_cb      bracket_d2z_load_cb.cu)      # v1
add_lto_fatbin_target(bracket_z2z_load_cb      bracket_z2z_load_cb.cu)      # v2-packing
add_lto_fatbin_target(bracket_d2z_load_v2_cb   bracket_d2z_load_v2_cb.cu)   # v2-bracket
add_lto_fatbin_target(bracket_store_cb          bracket_store_cb.cu)

# Ensure headers are generated
set(LTO_HEADERS 
    ${CMAKE_BINARY_DIR}/bracket_load_cb_fatbin.h
    ${CMAKE_BINARY_DIR}/bracket_d2z_load_cb_fatbin.h
    ${CMAKE_BINARY_DIR}/bracket_z2z_load_cb_fatbin.h
    ${CMAKE_BINARY_DIR}/bracket_d2z_load_v2_cb_fatbin.h
    ${CMAKE_BINARY_DIR}/bracket_store_cb_fatbin.h)

# Add shared library target for the JAX FFI (LTO v0, v1, v2)
add_library(lto_bracket SHARED
            ${PROJECT_SOURCE_DIR}/cufft_lto_bracket.cu
            ${LTO_HEADERS})

set_target_properties(lto_bracket PROPERTIES 
                      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
                      PREFIX "lib")

# Set JAX FFI headers path (detected via probe)
set(JAX_INCLUDE_DIR "/system/apps/userenv/volkmann/jax_env/lib/python3.14/site-packages/jaxlib/include")

target_include_directories(lto_bracket
                           PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
                           ${CMAKE_SOURCE_DIR}
                           ${CMAKE_BINARY_DIR}
                           ${JAX_INCLUDE_DIR})

target_link_libraries(lto_bracket PRIVATE CUDA::cufft CUDA::cudart CUDA::nvJitLink)

# Shared library target for JAX FFI is the primary output
