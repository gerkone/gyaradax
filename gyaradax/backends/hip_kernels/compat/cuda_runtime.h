// cuda runtime names on hip, for building ../cuda_kernels unchanged with hipcc
#pragma once

#include <hip/hip_runtime.h>
#include <cmath>
#include <cstddef>

#define cudaStream_t hipStream_t
#define cudaError_t hipError_t
#define cudaSuccess hipSuccess
#define cudaErrorUnknown hipErrorUnknown
#define cudaGetLastError hipGetLastError
#define cudaGetErrorString hipGetErrorString
#define cudaGetDevice hipGetDevice
#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemsetAsync hipMemsetAsync
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaFuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize

template <class F>
inline hipError_t cudaFuncSetAttribute(F* fn, hipFuncAttribute attr, int value) {
    return hipFuncSetAttribute(reinterpret_cast<const void*>(fn), attr, value);
}
