// hip and hipfft names on cuda and cufft, for building the hip bracket with nvcc
#pragma once
#include <cuda_runtime.h>
#include <cufft.h>
#define hipStream_t cudaStream_t
#define hipError_t cudaError_t
#define hipSuccess cudaSuccess
#define hipGetErrorString cudaGetErrorString
#define hipGetDevice cudaGetDevice
#define hipGetLastError cudaGetLastError
#define hipMalloc cudaMalloc
#define hipFree cudaFree
#define hipMemcpy cudaMemcpy
#define hipMemcpyHostToDevice cudaMemcpyHostToDevice
#define hipfftHandle cufftHandle
#define hipfftType cufftType
#define hipfftResult cufftResult
#define hipfftComplex cufftComplex
#define hipfftDoubleComplex cufftDoubleComplex
#define HIPFFT_C2C CUFFT_C2C
#define HIPFFT_Z2Z CUFFT_Z2Z
#define HIPFFT_SUCCESS CUFFT_SUCCESS
#define HIPFFT_FORWARD CUFFT_FORWARD
#define HIPFFT_BACKWARD CUFFT_INVERSE
#define hipfftExecC2C cufftExecC2C
#define hipfftExecZ2Z cufftExecZ2Z
#define hipfftCreate cufftCreate
#define hipfftMakePlanMany cufftMakePlanMany
#define hipfftDestroy cufftDestroy
#define hipfftSetStream cufftSetStream
#define hipErrorInvalidValue cudaErrorInvalidValue
#define hipEvent_t cudaEvent_t
#define hipEventCreate cudaEventCreate
#define hipEventDestroy cudaEventDestroy
#define hipEventRecord cudaEventRecord
#define hipEventSynchronize cudaEventSynchronize
#define hipEventElapsedTime cudaEventElapsedTime
#define hipDeviceGetAttribute cudaDeviceGetAttribute
#define hipDeviceAttributeMaxSharedMemoryPerBlock cudaDevAttrMaxSharedMemoryPerBlockOptin
#define hipFuncSetAttribute cudaFuncSetAttribute
#define hipFuncAttributeMaxDynamicSharedMemorySize cudaFuncAttributeMaxDynamicSharedMemorySize
#define __builtin_nontemporal_store(v, p) (*(p) = (v))
