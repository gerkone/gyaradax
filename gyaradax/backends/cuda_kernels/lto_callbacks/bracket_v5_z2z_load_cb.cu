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

struct V5Z2zInfo {
    const double2* df_packed;    // [b_df,  nkx, nky]
    const double2* phi_packed;   // [b_phi, nkx, nky]
    const double*  kx;
    const double*  ky;
    const int*     inverse_jind; // [mrad] dense_m → packed_kx_idx, -1 = absent
    int mrad, mphi, nkx, nky, b_df, b_phi;
};

__device__ static double2 v5_get(
    const double2* f, int b, int kxi, int js, int nkx, int nky)
{
    if (kxi < 0 || js >= nky) return make_double2(0.0, 0.0);
    return f[(size_t)b * nkx * nky + (size_t)kxi * nky + js];
}

__device__ static double2 v5_z2z_pack(
    unsigned long long offset, const V5Z2zInfo* ci)
{
    int plane = ci->mrad * ci->mphi;
    int gb    = (int)(offset / (unsigned long long)plane);
    int i     = (int)((offset / ci->mphi) % ci->mrad);
    int j     = (int)(offset % ci->mphi);

    bool is_phi = (gb >= ci->b_df);
    int  lb     = is_phi ? (gb - ci->b_df) % ci->b_phi : gb;
    const double2* field = is_phi ? ci->phi_packed : ci->df_packed;

    bool mirror = (j > ci->mphi / 2);
    int  j_src  = mirror ? ci->mphi - j : j;
    int  m_src  = mirror ? (ci->mrad - i) % ci->mrad : i;

    int kxi = ci->inverse_jind[m_src];
    if (kxi < 0 || j_src >= ci->nky) return make_double2(0.0, 0.0);

    double2 val = v5_get(field, lb, kxi, j_src, ci->nkx, ci->nky);
    double  kxv = ci->kx[kxi];
    double  kyv = ci->ky[j_src];

    // Hermitian symmetrisation at ky=0
    if ((j_src == 0 || j_src == ci->mphi / 2) && !mirror) {
        int m_pair  = (ci->mrad - i) % ci->mrad;
        int kx_pair = ci->inverse_jind[m_pair];
        if (kx_pair >= 0 && m_pair != i) {
            double2 vp = v5_get(field, lb, kx_pair, j_src, ci->nkx, ci->nky);
            val.x = 0.5 * (val.x + vp.x);
            val.y = 0.5 * (val.y - vp.y);
        }
    }

    double fy_re = -kyv * val.y, fy_im = kyv * val.x;
    double fx_re = -kxv * val.y, fx_im = kxv * val.x;

    if (!mirror)
        return make_double2(fy_re - fx_im, fy_im + fx_re);
    else
        return make_double2(fy_re + fx_im, fx_re - fy_im);
}

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

// ── True FP32 Z2Z Load Callback (Early Cast Optimization) ───────────────────
// Casts input double2 and kx/ky to float2/float IMMEDIATELY, then performs
// all arithmetic (Hermitian symmetry, derivative packing) in FP32.
// This reduces register pressure by ~50% and eliminates FP64 ALU usage.

__device__ static float2 v5_z2z_true_fp32_pack(
    unsigned long long offset, const V5Z2zInfo* ci)
{
    int plane = ci->mrad * ci->mphi;
    int gb    = (int)(offset / (unsigned long long)plane);
    int i     = (int)((offset / ci->mphi) % ci->mrad);
    int j     = (int)(offset % ci->mphi);

    bool is_phi = (gb >= ci->b_df);
    int  lb     = is_phi ? (gb - ci->b_df) % ci->b_phi : gb;
    const double2* field = is_phi ? ci->phi_packed : ci->df_packed;

    bool mirror = (j > ci->mphi / 2);
    int  j_src  = mirror ? ci->mphi - j : j;
    int  m_src  = mirror ? (ci->mrad - i) % ci->mrad : i;

    int kxi = ci->inverse_jind[m_src];
    if (kxi < 0 || j_src >= ci->nky) return make_float2(0.0f, 0.0f);

    // CAST IMMEDIATELY: double2 -> float2
    double2 val_d = v5_get(field, lb, kxi, j_src, ci->nkx, ci->nky);
    float2 val = make_float2((float)val_d.x, (float)val_d.y);
    
    // Cast kx/ky to FP32 immediately
    float kxv = (float)ci->kx[kxi];
    float kyv = (float)ci->ky[j_src];

    // Hermitian symmetrisation at ky=0 (in FP32)
    if ((j_src == 0 || j_src == ci->mphi / 2) && !mirror) {
        int m_pair  = (ci->mrad - i) % ci->mrad;
        int kx_pair = ci->inverse_jind[m_pair];
        if (kx_pair >= 0 && m_pair != i) {
            double2 vp_d = v5_get(field, lb, kx_pair, j_src, ci->nkx, ci->nky);
            float2 vp = make_float2((float)vp_d.x, (float)vp_d.y);
            val.x = 0.5f * (val.x + vp.x);
            val.y = 0.5f * (val.y - vp.y);
        }
    }

    // Derivative packing: fy + i*fx in FP32
    float fy_re = -kyv * val.y, fy_im = kyv * val.x;
    float fx_re = -kxv * val.y, fx_im = kxv * val.x;

    if (!mirror)
        return make_float2(fy_re - fx_im, fy_im + fx_re);
    else
        return make_float2(fy_re + fx_im, fx_re - fy_im);
}

__device__ cufftComplex d_v5_z2z_true_fp32_load(
    void *dataIn, unsigned long long offset,
    void *callerInfo, void *sharedPointer)
{
    return v5_z2z_true_fp32_pack(offset, (const V5Z2zInfo*)callerInfo);
}

__device__ cufftJITCallbackLoadC d_v5_z2z_true_fp32_load_addr = d_v5_z2z_true_fp32_load;
