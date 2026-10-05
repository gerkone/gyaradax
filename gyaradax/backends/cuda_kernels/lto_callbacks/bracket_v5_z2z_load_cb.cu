// Z2Z/C2C inverse load callback for v5 layout.
// Fuses the pack step into the FFT kernel:
//   - Hermitian gather from packed [b_df|b_phi, nkx, nky] spectrum
//   - fy + i*fx 2-for-1 packing in FP64
//   - Phi broadcast: batches b_df..b_df+b_phi-1 cycle over b_phi phi elements
//   - Hermitian symmetrisation at ky=0
//
// Two variants compiled into one fatbin:
//   d_v5_z2z_fp32_load  — for C2C (returns cufftComplex,       mixed precision v5)
//   d_v5_z2z_fp64_load  — for Z2Z (returns cufftDoubleComplex, FP64 v5)
// Both compute in FP64; fp32 variant casts on return.

#include <cufft.h>
#include <cufftXt.h>
#include <cuda_runtime.h>

#include "bracket_v5_z2z_pack.cuh"
#include "bracket_v6_info.cuh"

__device__ cufftComplex d_v5_z2z_fp32_load(
    void *dataIn, unsigned long long offset,
    void *callerInfo, void *sharedPointer)
{
    double2 r = v5_z2z_pack(offset, (const V5Z2zInfo*)callerInfo);
    return make_float2((float)r.x, (float)r.y);
}

__device__ cufftDoubleComplex d_v5_z2z_fp64_load(
    void *dataIn, unsigned long long offset,
    void *callerInfo, void *sharedPointer)
{
    return v5_z2z_pack(offset, (const V5Z2zInfo*)callerInfo);
}

__device__ cufftJITCallbackLoadC d_v5_z2z_fp32_load_addr = d_v5_z2z_fp32_load;
__device__ cufftJITCallbackLoadZ d_v5_z2z_fp64_load_addr = d_v5_z2z_fp64_load;

__device__ cufftComplex d_v5_z2z_true_fp32_load(
    void *dataIn, unsigned long long offset,
    void *callerInfo, void *sharedPointer)
{
    return v5_z2z_true_fp32_pack(offset, (const V5Z2zInfo*)callerInfo);
}

__device__ cufftJITCallbackLoadC d_v5_z2z_true_fp32_load_addr = d_v5_z2z_true_fp32_load;

// v6 column pass: pack read straight into the retained-column layout
__device__ cufftComplex d_v6_col_true_fp32_load(
    void *dataIn, unsigned long long offset,
    void *callerInfo, void *sharedPointer)
{
    const V6ColInfo* ci = (const V6ColInfo*)callerInfo;
    int gb, i, j;
    v6_col_split(offset, ci, &gb, &i, &j);
    return v5_z2z_true_fp32_pack_at(gb, i, j, &ci->z);
}

__device__ cufftDoubleComplex d_v6_col_fp64_load(
    void *dataIn, unsigned long long offset,
    void *callerInfo, void *sharedPointer)
{
    const V6ColInfo* ci = (const V6ColInfo*)callerInfo;
    int gb, i, j;
    v6_col_split(offset, ci, &gb, &i, &j);
    return v5_z2z_pack_at(gb, i, j, &ci->z);
}

__device__ cufftJITCallbackLoadC d_v6_col_true_fp32_load_addr = d_v6_col_true_fp32_load;
__device__ cufftJITCallbackLoadZ d_v6_col_fp64_load_addr      = d_v6_col_fp64_load;
