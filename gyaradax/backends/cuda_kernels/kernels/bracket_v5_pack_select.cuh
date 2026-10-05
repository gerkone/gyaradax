// Explicit pack kernel and C2C load-callback selection shared by the FP32/FP64 bracket pipelines.
#pragma once

#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>

#include "bracket_v5_z2z_pack.cuh"

__device__ __forceinline__ void v5_pack_store(float2* p, int gb, int i, int j, const V5Z2zInfo* ci) {
    *p = v5_z2z_true_fp32_pack_at(gb, i, j, ci);
}

__device__ __forceinline__ void v5_pack_store(double2* p, int gb, int i, int j, const V5Z2zInfo* ci) {
    *p = v5_z2z_pack_at(gb, i, j, ci);
}

// explicit pack, same math as the C2C load callback; threadIdx.x runs along a plane row
template <typename T>
__global__ void v5_z2z_pack_kernel(T* ws, const V5Z2zInfo ci, int n_rows) {
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= n_rows) return;
    const int gb = row / ci.mrad;
    const int i  = row - gb * ci.mrad;
    T* out = ws + (size_t)row * ci.mphi;
    for (int j = threadIdx.x; j < ci.mphi; j += blockDim.x)
        v5_pack_store(&out[j], gb, i, j, &ci);
}

template <typename T>
static cudaError_t v5_launch_pack(T* ws, const V5Z2zInfo& ci, cudaStream_t stream) {
    const int n_rows = (ci.b_df + ci.b_phi) * ci.mrad;
    const int bx = ci.mphi < 1024 ? ((ci.mphi + 31) / 32) * 32 : 1024;
    const int by = bx < 256 ? 256 / bx : 1;
    v5_z2z_pack_kernel<T><<<(n_rows + by - 1) / by, dim3(bx, by), 0, stream>>>(ws, ci, n_rows);
    return cudaGetLastError();
}

static inline cufftResult v5_exec_inverse(cufftHandle plan, float2* ws) {
    return cufftExecC2C(plan, (cufftComplex*)ws, (cufftComplex*)ws, CUFFT_INVERSE);
}

static inline cufftResult v5_exec_inverse(cufftHandle plan, double2* ws) {
    return cufftExecZ2Z(plan, (cufftDoubleComplex*)ws, (cufftDoubleComplex*)ws, CUFFT_INVERSE);
}

// whether cuFFT applied the load callback, probed on a NaN-filled workspace
template <typename T>
static cudaError_t v5_callback_applied(cufftHandle plan, T* ws, size_t n, size_t plane,
                                       cudaStream_t stream, bool* applied) {
    T probe[2];
    cudaError_t err = cudaMemsetAsync(ws, 0xFF, n * sizeof(T), stream);
    if (err != cudaSuccess) return err;
    if (v5_exec_inverse(plan, ws) != CUFFT_SUCCESS) return cudaErrorUnknown;
    if ((err = cudaMemcpyAsync(&probe[0], ws, sizeof(T), cudaMemcpyDeviceToHost, stream)) != cudaSuccess)
        return err;
    if ((err = cudaMemcpyAsync(&probe[1], ws + n - plane, sizeof(T), cudaMemcpyDeviceToHost, stream)) != cudaSuccess)
        return err;
    if ((err = cudaStreamSynchronize(stream)) != cudaSuccess) return err;
    *applied = isfinite(probe[0].x) && isfinite(probe[0].y) && isfinite(probe[1].x) && isfinite(probe[1].y);
    return cudaSuccess;
}

// mean time (ms) of the inverse transform, preceded by the explicit pack when pack is set
template <typename T>
static cudaError_t v5_time_inverse(cufftHandle plan, T* ws, const V5Z2zInfo& ci, bool pack,
                                   cudaStream_t stream, float* ms) {
    const int reps = 3;
    cudaEvent_t e0, e1;
    cudaError_t err;
    if ((err = cudaEventCreate(&e0)) != cudaSuccess) return err;
    if ((err = cudaEventCreate(&e1)) != cudaSuccess) return err;
    for (int r = -1; r < reps; ++r) {
        if (r == 0 && (err = cudaEventRecord(e0, stream)) != cudaSuccess) return err;
        if (pack && (err = v5_launch_pack(ws, ci, stream)) != cudaSuccess) return err;
        if (v5_exec_inverse(plan, ws) != CUFFT_SUCCESS) return cudaErrorUnknown;
    }
    if ((err = cudaEventRecord(e1, stream)) != cudaSuccess) return err;
    if ((err = cudaEventSynchronize(e1)) != cudaSuccess) return err;
    if ((err = cudaEventElapsedTime(ms, e0, e1)) != cudaSuccess) return err;
    *ms /= reps;
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    return cudaSuccess;
}
