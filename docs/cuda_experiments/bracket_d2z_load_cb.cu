#include <cufft.h>
#include <cufftXt.h>
#include <cuda_runtime.h>

struct BracketD2zInfo {
    const double *py, *fx, *px, *fy;
    const double* dum_s;
    int nspec, mrad, mphi;
    double scale;
};

__device__ double d_bracket_d2z_load(void *dataIn, unsigned long long offset, void *callerInfo, void *sharedPointer) {
    const BracketD2zInfo* ci = (const BracketD2zInfo*)callerInfo;
    int batch_idx = (int)(offset / ((unsigned long long)ci->mrad * ci->mphi));
    double dum = ci->dum_s[batch_idx % ci->nspec];
    
    // JAX norm="backward" on irfft2 gives 1/N. Two of them give 1/N^2.
    // cuFFT Z2D is 1.0. So to match JAX, we need 1/N^2.
    // BUT JAX finally multiplies by N. So Total = 1/N.
    double factor = dum * ci->scale;
    
    return factor * (ci->py[offset] * ci->fx[offset] - ci->px[offset] * ci->fy[offset]);
}

__device__ cufftJITCallbackLoadD d_bracket_d2z_load_addr = d_bracket_d2z_load;
