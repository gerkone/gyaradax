#include <cufft.h>
#include <cufftXt.h>
#include <cuda_runtime.h>

// Sparse store callback: scatters D2Z output [batch, mrad, mphiw3]
// directly into the packed output buffer [batch, nkx, nky].
// Uses inverse_jind[i_dense] to map dense kx index -> packed index;
// elements with inverse_jind[i_dense] < 0 or j >= nky are dropped.
struct StoreInfo {
    double2*    out_packed;      // [batch, nkx, nky]  — FFI output buffer
    const int*  inverse_jind;    // [mrad]  dense->packed, -1 if absent
    int mrad, mphiw3, nkx, nky;
};

__device__ void d_store_cb_ptr(void *dataOut, unsigned long long offset,
                                cufftDoubleComplex element,
                                void *callerInfo, void *sharedPointer) {
    const StoreInfo* si = (const StoreInfo*)callerInfo;
    int batch_idx = (int)(offset / ((unsigned long long)si->mrad * si->mphiw3));
    int i_dense   = (int)((offset / si->mphiw3) % si->mrad);
    int j         = (int)(offset % si->mphiw3);

    if (j >= si->nky) return;

    int i_pack = si->inverse_jind[i_dense];
    if (i_pack < 0) return;

    si->out_packed[((unsigned long long)batch_idx * si->nkx + i_pack) * si->nky + j] = element;
}

__device__ cufftJITCallbackStoreZ d_store_cb_ptr_addr = d_store_cb_ptr;
