// D2Z store callback for v5 layout with zero-mode masking.
// Scatters D2Z dense output [batch, mrad, mphi_half] to packed [batch, nkx, nky].
// Uses inverse_jind[dense_m] to map dense kx index -> packed index;
// elements with inverse_jind[i_dense] < 0 or j >= nky are dropped.
// Zero-mode masking: (ixzero, iyzero) forced to zero.

#include <cufft.h>
#include <cufftXt.h>
#include <cuda_runtime.h>

#include "bracket_v6_info.cuh"

struct V5StoreInfo {
    double2*    out_packed;      // [batch, nkx, nky] -- FFI output buffer
    const int*  inverse_jind;    // [mrad] dense -> packed, -1 if absent
    int mrad, mphiw3, nkx, nky;
    int ixzero, iyzero;
};

__device__ void d_v5_store_cb(
    void *dataOut, unsigned long long offset,
    cufftDoubleComplex element,
    void *callerInfo, void *sharedPointer)
{
    const V5StoreInfo* si = (const V5StoreInfo*)callerInfo;
    int batch_idx = (int)(offset / ((unsigned long long)si->mrad * si->mphiw3));
    int i_dense   = (int)((offset / si->mphiw3) % si->mrad);
    int j         = (int)(offset % si->mphiw3);

    if (j >= si->nky) return;

    int i_pack = si->inverse_jind[i_dense];
    if (i_pack < 0) return;

    // Zero-mode masking
    if (i_pack == si->ixzero && j == si->iyzero)
        element = {0.0, 0.0};

    si->out_packed[((unsigned long long)batch_idx * si->nkx + i_pack) * si->nky + j] = element;
}

__device__ void d_v5_store_fp32_cb(
    void *dataOut, unsigned long long offset,
    cufftComplex element,
    void *callerInfo, void *sharedPointer)
{
    const V5StoreInfo* si = (const V5StoreInfo*)callerInfo;
    int batch_idx = (int)(offset / ((unsigned long long)si->mrad * si->mphiw3));
    int i_dense   = (int)((offset / si->mphiw3) % si->mrad);
    int j         = (int)(offset % si->mphiw3);

    if (j >= si->nky) return;

    int i_pack = si->inverse_jind[i_dense];
    if (i_pack < 0) return;

    // Zero-mode masking
    double2 elem_d = {0.0, 0.0};
    if (i_pack == si->ixzero && j == si->iyzero) {
        elem_d = {0.0, 0.0};
    } else {
        elem_d = {(double)element.x, (double)element.y};
    }

    si->out_packed[((unsigned long long)batch_idx * si->nkx + i_pack) * si->nky + j] = elem_d;
}

__device__ cufftJITCallbackStoreZ d_v5_store_cb_addr       = d_v5_store_cb;
__device__ cufftJITCallbackStoreC d_v5_store_fp32_cb_addr  = d_v5_store_fp32_cb;

// v6 column pass: [mrad][n_df][nky] -> packed [n_df, nkx, nky], dropping dealiased kx rows
template <class T>
__device__ __forceinline__ static void v6_store(unsigned long long offset, T element, const V6StoreInfo* si)
{
    const unsigned long long row = (unsigned long long)si->n_df * si->nky;
    const int i = (int)(offset / row);
    const int i_pack = si->inverse_jind[i];
    if (i_pack < 0) return;
    const int c = (int)(offset - (unsigned long long)i * row);
    const int gb = c / si->nky, k = c - gb * si->nky;
    double2 v = make_double2((double)element.x, (double)element.y);
    if (i_pack == si->ixzero && k == si->iyzero) v = make_double2(0.0, 0.0);
    si->out_packed[((unsigned long long)gb * si->nkx + i_pack) * si->nky + k] = v;
}

__device__ void d_v6_store_fp32_cb(
    void *dataOut, unsigned long long offset, cufftComplex element,
    void *callerInfo, void *sharedPointer)
{
    v6_store(offset, element, (const V6StoreInfo*)callerInfo);
}

__device__ void d_v6_store_fp64_cb(
    void *dataOut, unsigned long long offset, cufftDoubleComplex element,
    void *callerInfo, void *sharedPointer)
{
    v6_store(offset, element, (const V6StoreInfo*)callerInfo);
}

__device__ cufftJITCallbackStoreC d_v6_store_fp32_cb_addr = d_v6_store_fp32_cb;
__device__ cufftJITCallbackStoreZ d_v6_store_fp64_cb_addr = d_v6_store_fp64_cb;
