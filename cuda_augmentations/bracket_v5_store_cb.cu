// D2Z store callback for v5 layout with zero-mode masking.
// Scatters D2Z dense output [batch, mrad, mphi_half] to packed [batch, nkx, nky].
// Uses inverse_jind[dense_m] to map dense kx index -> packed index;
// elements with inverse_jind[i_dense] < 0 or j >= nky are dropped.
// Zero-mode masking: (ixzero, iyzero) forced to zero.

#include <cufft.h>
#include <cufftXt.h>
#include <cuda_runtime.h>

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

__device__ cufftJITCallbackStoreZ d_v5_store_cb_addr = d_v5_store_cb;
