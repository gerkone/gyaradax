#include <cufft.h>
#include <cufftXt.h>
#include <cuda_runtime.h>

struct ScaleFactors {
    double alpha0, beta0;
    double alpha1, beta1;
    double inv_a0, inv_b0;
    double inv_a1, inv_b1;
};

struct BracketD2zInfoMerged {
    const double2* ws;     
    const double*  dum_s;
    const ScaleFactors* sf;
    size_t df_offset;      // Number of elements (mrad * mphi * field_boundary)
    int nspec, mrad, mphi;
    double scale;
};

__device__ double d_bracket_d2z_merged_load(void *dataIn, unsigned long long offset,
                                            void *callerInfo, void *sharedPointer) {
    const BracketD2zInfoMerged* ci = (const BracketD2zInfoMerged*)callerInfo;
    int batch_idx = (int)(offset / ((unsigned long long)ci->mrad * ci->mphi));
    double dum    = ci->dum_s[batch_idx % ci->nspec];
    
    double2 v0 = ci->ws[offset];
    double2 v1 = ci->ws[offset + ci->df_offset];
    
    return ci->scale * dum * (v0.x * ci->sf->alpha0 * v0.y * ci->sf->beta0 - 
                              v1.x * ci->sf->alpha1 * v1.y * ci->sf->beta1);
}

__device__ cufftJITCallbackLoadD d_bracket_d2z_merged_load_addr = d_bracket_d2z_merged_load;
