#include <cufft.h>
#include <cufftXt.h>
#include <cuda_runtime.h>

struct CallbackInfo {
    const double2* df_packed;
    const double2* phi_packed;
    const double*  kx;
    const double*  ky;
    const int*     inverse_jind;
    int mrad, mphi_half, nkx, nky, gradient_type, n_df_batches, n_phi_batches;
};

__device__ double2 d_get_gradient(const CallbackInfo* ci, int batch_idx, int i_dense, int j_dense, int g_type) {
    int i_pack = ci->inverse_jind[i_dense];
    if (i_pack < 0 || j_dense >= ci->nky) return make_double2(0.0, 0.0);

    // 0=phi_y, 2=phi_x → phi;  1=f_x, 3=f_y → df
    const double2* packed = (g_type == 0 || g_type == 2) ? ci->phi_packed : ci->df_packed;
    int b_idx = (g_type == 0 || g_type == 2) ? (batch_idx % ci->n_phi_batches) : (batch_idx % ci->n_df_batches);
    
    double2 val = packed[((unsigned long long)b_idx * ci->nkx + i_pack) * ci->nky + j_dense];
    double k = (g_type == 0 || g_type == 3) ? ci->ky[j_dense] : ci->kx[i_pack];
    
    // G = i * k * val = (-k*val.y, k*val.x)
    return make_double2(-k * val.y, k * val.x);
}

__device__ cufftDoubleComplex d_load_cb_ptr(void *dataIn, unsigned long long offset, void *callerInfo, void *sharedPointer) {
    const CallbackInfo* ci = (const CallbackInfo*)callerInfo;
    int batch_idx = (int)(offset / ((unsigned long long)ci->mrad * ci->mphi_half));
    int i_dense = (int)((offset / ci->mphi_half) % ci->mrad);
    int j_dense = (int)(offset % ci->mphi_half);

    double2 G = d_get_gradient(ci, batch_idx, i_dense, j_dense, ci->gradient_type);
    return make_double2(G.x, G.y);
}

__device__ cufftJITCallbackLoadZ d_load_cb_ptr_addr = d_load_cb_ptr;
