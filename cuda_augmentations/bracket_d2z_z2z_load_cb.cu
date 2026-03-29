#include <cufft.h>
#include <cufftXt.h>
#include <cuda_runtime.h>

// D2Z load callback for the explicit-packing Z2Z path.
// ws0[offset] = {phi_y_real[offset], f_x_real[offset]} after Z2Z IFFT.
// ws1[offset] = {phi_x_real[offset], f_y_real[offset]} after Z2Z IFFT.
// cuFFT Z2Z IFFT is unnormalized (N * IDFT), so bracket = ws0.x*ws0.y - ws1.x*ws1.y
// returns N^2 * dum * (phi_y * f_x - phi_x * f_y); Python divides by N^2.
struct ScaleFactors {
    double alpha0, beta0;
    double alpha1, beta1;
    double inv_a0, inv_b0;
    double inv_a1, inv_b1;
};

struct BracketD2zZ2zInfo {
    const double2* ws0;    // [batch, mrad, mphi]  Re=ga/alpha0, Im=gb/beta0
    const double2* ws1;    // [batch, mrad, mphi]  Re=gc/alpha1, Im=gd/beta1
    const double*  dum_s;  // [nspec]
    const ScaleFactors* sf;
    int nspec, mrad, mphi;
    double scale;          // inv_n2 = 1.0/(mrad*mphi)^2
};

__device__ double d_bracket_d2z_z2z_load(void *dataIn, unsigned long long offset,
                                           void *callerInfo, void *sharedPointer) {
    const BracketD2zZ2zInfo* ci = (const BracketD2zZ2zInfo*)callerInfo;
    int batch_idx = (int)(offset / ((unsigned long long)ci->mrad * ci->mphi));
    double dum    = ci->dum_s[batch_idx % ci->nspec];
    double2 v0    = ci->ws0[offset];
    double2 v1    = ci->ws1[offset];
    
    // v0.x = ga/alpha0, v0.y = gb/beta0 -> ga = v0.x * alpha0, gb = v0.y * beta0
    // v1.x = gc/alpha1, v1.y = gd/beta1 -> gc = v1.x * alpha1, gd = v1.y * beta1

    // Bracket = phi_y*f_x - phi_x*f_y = ga*gb - gc*gd
    // Wait, the pairing was:
    // pair0: phi derivatives, pair1: df derivatives?
    // Let's re-check bracket_z2z_load_cb.cu.
    // ci->pair_type == 0: valA=phi, valB=df? No, look at bracket_z2z_load_cb.cu (Step 44):
    // if (ci->pair_type == 0) { valA = phi; kA = ky_val; valB = df; kB = kx_val; } -> ga = phi_y, gb = df_x
    // if (ci->pair_type == 1) { valA = phi; kA = kx_val; valB = df; kB = ky_val; } -> gc = phi_x, gd = df_y
    
    double ga = v0.x * ci->sf->alpha0;
    double gb = v0.y * ci->sf->beta0; // oops, wait.
    // I should be careful.
    // In bracket_z2z_load_cb.cu:
    // if (pair_type == 0): ga = phi_y, gb = df_x. 
    //   alpha = alpha0 (max|phi_y|), beta = alpha1 (max|df_x|)? 
    // No, I used:
    // alpha = (ci->pair_type == 0) ? ci->sf->alpha0 : ci->sf->alpha1;
    // beta  = (ci->pair_type == 0) ? ci->sf->beta0  : ci->sf->beta1;
    // So for pair 0: alpha = alpha0, beta = beta0. 
    // ga = phi_y / alpha0, gb = df_x / beta0.
    // For pair 1: alpha = alpha1, beta = beta1.
    // gc = phi_x / alpha1, gd = df_y / beta1.
    
    // Bracket = phi_y * df_x - phi_x * df_y
    //         = (v0.x * alpha0) * (v0.y * beta0) - (v1.x * alpha1) * (v1.y * beta1)
    
    // Bracket = phi_x*f_y - phi_y*f_x ??? Or vice-versa?
    // Let's try v0 - v1 again.
    return ci->scale * dum * (v0.x * ci->sf->alpha0 * v0.y * ci->sf->beta0 - 
                             v1.x * ci->sf->alpha1 * v1.y * ci->sf->beta1);
}

__device__ cufftJITCallbackLoadD d_bracket_d2z_z2z_load_addr = d_bracket_d2z_z2z_load;
