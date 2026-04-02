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
# Override with: cmake -DGPU_ARCHITECTURES="80;90;100;103"                                      
# Use "native" to auto-detect the GPU on the build machine.                                                                                                                               
set(GPU_ARCHITECTURES "103" CACHE STRING                                                 
    "Target GPU compute capabilities (semicolon-separated, or 'native')")                                                                                                                 
set(CMAKE_CUDA_ARCHITECTURES ${GPU_ARCHITECTURES})


# ── Find Python and auto-detect JAX FFI headers ────────────────────                                                                                                                     
find_package(Python3 REQUIRED COMPONENTS Interpreter)                                       
execute_process(                                                                            
    COMMAND ${Python3_EXECUTABLE} -c                                                        
        "import jaxlib, pathlib; print(jaxlib.__version__); print(pathlib.Path(jaxlib.__file__).parent / 'include')"                                                                                                 
    OUTPUT_VARIABLE JAX_INFO                                                                
    OUTPUT_STRIP_TRAILING_WHITESPACE                                                        
    RESULT_VARIABLE _jax_exit                                                               
)
                                                                                      
if(NOT _jax_exit EQUAL 0)
    message(FATAL_ERROR
        "Could not locate JAX FFI headers. "
        "Please ensure 'jaxlib' is installed in the current Python environment:\n"
        "Python path used: ${Python3_EXECUTABLE}")
endif()

# Split the multi-line output
string(REPLACE "\n" ";" JAX_INFO_LIST "${JAX_INFO}")
list(GET JAX_INFO_LIST 0 JAXLIB_VERSION)
list(GET JAX_INFO_LIST 1 JAX_INCLUDE_DIR)                                                  

message(STATUS "Found Python:       ${Python3_EXECUTABLE}")                                 
message(STATUS "Found jaxlib:       ${JAXLIB_VERSION}")                                     
# ── CUDAToolkit ───────────────────────────────────────────────────                                                                                                                      
find_package(CUDAToolkit REQUIRED)
message(STATUS "Target GPU Arch:    ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Found CUDA compiler: ${CMAKE_CUDA_COMPILER}")
message(STATUS "Found CXX compiler:  ${CMAKE_CXX_COMPILER}")
message(STATUS "Found CUDA toolkit:  ${CUDAToolkit_VERSION} (at ${CUDAToolkit_LIBRARY_DIR})")


# ── LTO fatbin tooling ────────────────────────────────────────────                                                                                                                           


# ── Shared library ────────────────────────────────────────────────                                                                                                                      
add_library(gyaradax_cuda SHARED                                                            
    kernels/cufft_bracket.cu                                                          
    kernels/apply_parallel.cu                                                               
    kernels/apply_parallel_dual.cu                                                          
    kernels/apply_vpar_stencil.cu                                                           
    kernels/apply_vpar_dual_stencil.cu                                                      
    kernels/linear_rhs_fused.cu                                                             
    kernels/linear_rhs_vtiled.cu                                                            
    kernels/compute_scale_factors.cu)

set_target_properties(gyaradax_cuda PROPERTIES                                              
    PREFIX "lib"                                                                            
    OUTPUT_NAME "gyaradax_cuda")                                                            

target_compile_options(gyaradax_cuda PRIVATE                                                
    $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas -v>)                                                 

target_include_directories(gyaradax_cuda PRIVATE                                            
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}                                               
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels                                                     
    ${JAX_INCLUDE_DIR})                                                                     

target_link_libraries(gyaradax_cuda PRIVATE                                                 
    CUDA::cufft                                                                             
    CUDA::cudart)


# ── Install: place .so next to this CMakeLists.txt ────────────────                                                                                                                      
# cmake --install _build/cuda  →  gyaradax/backends/cuda/libgyaradax_cuda.so                                                                                                              
install(TARGETS gyaradax_cuda                                                               
    LIBRARY DESTINATION ${CMAKE_CURRENT_SOURCE_DIR})  