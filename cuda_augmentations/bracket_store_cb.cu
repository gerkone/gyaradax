#include <cufft.h>
#include <cufftXt.h>
#include <cuda_runtime.h>

struct StoreInfo {
    double2* out_packed;
    int mrad, mphi_half, nkx, nky;
};

__device__ void d_store_cb_ptr(void *dataOut, unsigned long long offset, cufftDoubleComplex element, void *callerInfo, void *sharedPointer) {
    const StoreInfo* si = (const StoreInfo*)callerInfo;
    int batch_idx = (int)(offset / ((unsigned long long)si->mrad * si->mphi_half));
    int i_dense = (int)((offset / si->mphi_half) % si->mrad);
    int j_dense = (int)(offset % si->mphi_half);

    // Spectral packing: only keep [0:nkx, 0:nky]
    if (i_dense < si->nkx && j_dense < si->nky) {
        si->out_packed[((unsigned long long)batch_idx * si->nkx + i_dense) * si->nky + j_dense] = element;
    }
}

__device__ cufftJITCallbackStoreZ d_store_cb_ptr_addr = d_store_cb_ptr;
