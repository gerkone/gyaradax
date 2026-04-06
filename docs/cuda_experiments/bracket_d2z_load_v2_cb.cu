#include <cufft.h>
#include <cufftXt.h>
#include <cuda_runtime.h>

struct BracketD2zV2Info {
    const double2 *ws0, *ws1;
    const double* dum_s;
    int nspec, mrad, mphi;
    double scale;
};

__device__ double d_bracket_d2z_v2_load(void *dataIn, unsigned long long offset, void *callerInfo, void *sharedPointer) {
    const BracketD2zV2Info* ci = (const BracketD2zV2Info*)callerInfo;
    int batch_idx = (int)(offset / ((unsigned long long)ci->mrad * ci->mphi));
    
    // GA = ws0.x, GB = ws0.y. GC = ws1.x, GD = ws1.y.
    // Bracket = GA*GB - GC*GD
    double2 val0 = ci->ws0[offset];
    double2 val1 = ci->ws1[offset];
    
    // sign: py*fx - px*fy (matches JAX bracket)
    return ci->scale * (val0.x * val0.y - val1.x * val1.y);
}

__device__ cufftJITCallbackLoadD d_bracket_d2z_v2_load_addr = d_bracket_d2z_v2_load;
